//! engram, a low-level associative vector memory engine speaking the Redis
//! wire protocol. It keeps the classic key-value store and adds named vector
//! indices with approximate nearest-neighbour search.

mod command;
mod config;
mod event_loop;
mod persist;
mod resp;
mod server;
mod state;
mod vector;

use config::Config;
use event_loop::EventLoop;
use server::Server;

fn main() -> std::io::Result<()> {
    let config = match Config::load() {
        Ok(config) => config,
        Err(e) => {
            eprintln!("config error: {e}");
            std::process::exit(1);
        }
    };

    let mut event_loop = EventLoop::new(config.max_clients)?;
    let mut server = Server::bind(config)?;
    event_loop.run(&mut server)
}
