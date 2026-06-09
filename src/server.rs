//! The server that accepts connections and serves commands.
//!
//! `Server` is the [`EventHandler`] driven by the [`EventLoop`]. It owns the
//! listener, the shared state, and the live connections.
//!
//! Most commands run inline on the event-loop thread — they touch only the
//! in-memory maps and finish in nanoseconds. `VSEARCH` is different: walking the
//! graph can take milliseconds, long enough to stall every other client. So a
//! search is handed to a [`SearchPool`] of worker threads, which read the
//! registry under a shared lock and send the encoded reply back. The loop is
//! woken from its `poll` by a self-pipe registered like any other descriptor.
//!
//! To keep a single connection's replies in request order, a connection with a
//! search in flight is marked busy and no further commands from it are
//! dispatched until the search completes. Other connections are unaffected,
//! which is the whole point: one slow search no longer blocks the world.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::fd::{AsRawFd, RawFd};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

use crate::command;
use crate::config::Config;
use crate::event_loop::{Event, EventHandler, EventLoop, Operation};
use crate::resp;
use crate::state::State;
use crate::vector::VectorRegistry;

/// How many bytes to read from a socket per `read` call.
const READ_CHUNK: usize = 16 * 1024;

/// Returns whether `name` is the `VSEARCH` command (the offloaded one).
fn is_search(name: &[u8]) -> bool {
    name.eq_ignore_ascii_case(b"VSEARCH")
}

/// A search to run off the event-loop thread.
struct Job {
    conn_id: u64,
    fd: RawFd,
    argv: Vec<Vec<u8>>,
}

/// A finished search, ready to write back to its connection.
struct Done {
    conn_id: u64,
    fd: RawFd,
    reply: Vec<u8>,
}

/// A pool of threads that execute searches against the shared registry.
struct SearchPool {
    jobs: Sender<Job>,
    results: Receiver<Done>,
    /// Read end of the self-pipe; readable whenever a result is queued.
    wake_read: RawFd,
    /// Write end; workers write one byte here to wake the event loop.
    wake_write: RawFd,
}

impl SearchPool {
    fn new(registry: Arc<RwLock<VectorRegistry>>, threads: usize) -> io::Result<Self> {
        // Self-pipe: workers write a byte, the event loop wakes on the read end.
        let mut fds = [0 as libc::c_int; 2];
        // SAFETY: `fds` is a valid two-element array for pipe(2) to fill.
        if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let (wake_read, wake_write) = (fds[0], fds[1]);
        // SAFETY: standard non-blocking flag toggle on a fd we own.
        unsafe {
            let flags = libc::fcntl(wake_read, libc::F_GETFL);
            libc::fcntl(wake_read, libc::F_SETFL, flags | libc::O_NONBLOCK);
        }

        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (res_tx, res_rx) = mpsc::channel::<Done>();
        let job_rx = Arc::new(Mutex::new(job_rx));

        for _ in 0..threads.max(1) {
            let job_rx = Arc::clone(&job_rx);
            let res_tx = res_tx.clone();
            let registry = Arc::clone(&registry);
            thread::spawn(move || {
                loop {
                    // Hold the receiver lock only to dequeue, not while searching,
                    // so workers process concurrently.
                    let job = {
                        let rx = job_rx.lock().unwrap();
                        match rx.recv() {
                            Ok(job) => job,
                            Err(_) => break, // sender dropped: shut down
                        }
                    };

                    let reply = {
                        let reg = registry.read().unwrap_or_else(|e| e.into_inner());
                        command::run_search(&job.argv[1..], &reg).encode()
                    };

                    if res_tx
                        .send(Done {
                            conn_id: job.conn_id,
                            fd: job.fd,
                            reply,
                        })
                        .is_err()
                    {
                        break;
                    }
                    let byte = 1u8;
                    // SAFETY: one byte to the pipe write end; EAGAIN (pipe already
                    // has pending wakes) is fine to ignore.
                    unsafe {
                        libc::write(wake_write, &byte as *const u8 as *const libc::c_void, 1);
                    }
                }
            });
        }

        Ok(Self {
            jobs: job_tx,
            results: res_rx,
            wake_read,
            wake_write,
        })
    }

    /// Drains the self-pipe so it is not perpetually readable.
    fn drain_wake(&self) {
        let mut buf = [0u8; 256];
        loop {
            // SAFETY: non-blocking read into a local buffer; loop until drained.
            let n = unsafe {
                libc::read(
                    self.wake_read,
                    buf.as_mut_ptr() as *mut libc::c_void,
                    buf.len(),
                )
            };
            if n <= 0 {
                break;
            }
        }
    }
}

impl Drop for SearchPool {
    fn drop(&mut self) {
        // SAFETY: both ends are fds this pool created and owns.
        unsafe {
            libc::close(self.wake_read);
            libc::close(self.wake_write);
        }
    }
}

/// A connected client and the bytes read from it but not yet parsed.
struct Connection {
    /// The client's TCP stream.
    stream: TcpStream,

    /// Bytes received but not yet consumed by a complete command, which may
    /// arrive split across reads.
    buffer: Vec<u8>,

    /// A monotonic id distinguishing this connection from a later one that
    /// happens to reuse the same file descriptor.
    id: u64,

    /// True while a search is in flight; no commands are dispatched until it
    /// completes, so replies stay in request order.
    busy: bool,
}

/// The event-driven server, holding the listener, shared state, and connections.
pub struct Server {
    /// The listening socket new clients connect to.
    listener: TcpListener,

    /// The shared state every command operates on.
    state: State,

    /// Live connections, keyed by file descriptor.
    connections: HashMap<RawFd, Connection>,

    /// Worker pool that executes searches off the event-loop thread.
    pool: SearchPool,

    /// Source of unique connection ids.
    next_id: u64,
}

impl Server {
    /// Binds the listening socket from `config` in non-blocking mode and starts
    /// the search worker pool.
    pub fn bind(config: Config) -> io::Result<Self> {
        let listener = TcpListener::bind(config.addr())?;
        listener.set_nonblocking(true)?;

        let state = State::new(config);
        let threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let pool = SearchPool::new(Arc::clone(&state.vectors), threads)?;

        Ok(Self {
            listener,
            state,
            connections: HashMap::new(),
            pool,
            next_id: 0,
        })
    }

    /// Accepts every pending connection, registering each with the event loop.
    fn accept(&mut self, event_loop: &mut EventLoop) -> io::Result<()> {
        loop {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(true)?;
                    stream.set_nodelay(true)?;

                    let fd = stream.as_raw_fd();
                    event_loop.subscribe(Event {
                        fd,
                        op: Operation::Read,
                    })?;

                    let id = self.next_id;
                    self.next_id += 1;
                    self.connections.insert(
                        fd,
                        Connection {
                            stream,
                            buffer: Vec::new(),
                            id,
                            busy: false,
                        },
                    );
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Reads available bytes from `fd` into its buffer, then drives command
    /// processing. Drops the connection on end-of-stream or read error.
    fn handle_client(&mut self, fd: RawFd) {
        let mut close = false;

        if let Some(conn) = self.connections.get_mut(&fd) {
            let mut chunk = [0u8; READ_CHUNK];
            match conn.stream.read(&mut chunk) {
                Ok(0) => close = true,
                Ok(n) => conn.buffer.extend_from_slice(&chunk[..n]),
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                Err(_) => close = true,
            }
        }

        if close {
            self.connections.remove(&fd);
        } else {
            self.process_commands(fd);
        }
    }

    /// Parses and serves complete commands from `fd`'s buffer until it is empty,
    /// incomplete, or the connection goes busy waiting on a search.
    fn process_commands(&mut self, fd: RawFd) {
        loop {
            let conn = match self.connections.get_mut(&fd) {
                Some(conn) if !conn.busy => conn,
                _ => return, // gone, or busy waiting on a search
            };

            match resp::Request::parse(&conn.buffer) {
                Ok(resp::Request::Command { argv, consumed }) => {
                    conn.buffer.drain(..consumed);
                    if is_search(&argv[0]) {
                        // Offload: stop serving this connection until it returns.
                        conn.busy = true;
                        let job = Job {
                            conn_id: conn.id,
                            fd,
                            argv,
                        };
                        if self.pool.jobs.send(job).is_err() {
                            self.connections.remove(&fd);
                        }
                        return;
                    }
                    let reply = command::dispatch(&argv, &mut self.state);
                    if conn.stream.write_all(&reply.encode()).is_err() {
                        self.connections.remove(&fd);
                        return;
                    }
                }
                Ok(resp::Request::Empty { consumed }) => {
                    conn.buffer.drain(..consumed);
                }
                Ok(resp::Request::Incomplete) => return,
                Err(e) => {
                    let reply = resp::Value::Error(format!("ERR {e}"));
                    let _ = conn.stream.write_all(&reply.encode());
                    self.connections.remove(&fd);
                    return;
                }
            }
        }
    }

    /// Handles the self-pipe waking: writes back every finished search and
    /// resumes the connections that were waiting on one.
    fn on_search_results(&mut self) {
        self.pool.drain_wake();

        let mut resume = Vec::new();
        while let Ok(done) = self.pool.results.try_recv() {
            // Match the live connection by id, guarding against fd reuse by a
            // newer connection while this search was running.
            let outcome = match self.connections.get_mut(&done.fd) {
                Some(conn) if conn.id == done.conn_id && conn.busy => {
                    match conn.stream.write_all(&done.reply) {
                        Ok(()) => {
                            conn.busy = false;
                            Some(true)
                        }
                        Err(_) => Some(false),
                    }
                }
                _ => None, // stale result for a closed/replaced connection
            };
            match outcome {
                Some(true) => resume.push(done.fd),
                Some(false) => {
                    self.connections.remove(&done.fd);
                }
                None => {}
            }
        }

        for fd in resume {
            self.process_commands(fd);
        }
    }
}

impl EventHandler for Server {
    fn register(&mut self, event_loop: &mut EventLoop) -> io::Result<()> {
        event_loop.subscribe(Event {
            fd: self.listener.as_raw_fd(),
            op: Operation::Read,
        })?;
        // Wake the loop whenever a worker has a search result ready.
        event_loop.subscribe(Event {
            fd: self.pool.wake_read,
            op: Operation::Read,
        })
    }

    fn handle(&mut self, event: Event, event_loop: &mut EventLoop) -> io::Result<()> {
        if event.fd == self.listener.as_raw_fd() {
            self.accept(event_loop)
        } else if event.fd == self.pool.wake_read {
            self.on_search_results();
            Ok(())
        } else {
            self.handle_client(event.fd);
            Ok(())
        }
    }
}
