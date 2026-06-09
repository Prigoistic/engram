//! Durability for the vector indices.
//!
//! Persistence follows the append-log-plus-snapshot shape: every mutation is
//! appended to a [`Wal`] and flushed, so a crash loses nothing committed; a
//! [`snapshot`] periodically captures the full live state and lets the log be
//! truncated so replay stays bounded. On startup [`Persist::open`] rebuilds the
//! registry from the snapshot and then replays the log on top.
//!
//! Replay applies operations straight to the [`VectorRegistry`], bypassing the
//! log, so recovery never re-logs what it reads.

mod mmap;
mod snapshot;
mod wal;

use std::io;
use std::path::PathBuf;

use crate::vector::{self, Metric, VectorRegistry};
use snapshot::SnapBlock;
use wal::Wal;

const SNAPSHOT_FILE: &str = "engram.snap";
const WAL_FILE: &str = "engram.wal";

const OP_NEW: u8 = 1;
const OP_ADD: u8 = 2;
const OP_DEL: u8 = 3;

/// The persistence handle: a data directory and its open write-ahead log.
pub struct Persist {
    dir: PathBuf,
    wal: Wal,
}

impl Persist {
    /// Opens the data directory, recovers the registry from snapshot + log, and
    /// returns the handle alongside the recovered registry. Creates the
    /// directory if it does not exist.
    pub fn open(dir: &str) -> io::Result<(Self, VectorRegistry)> {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir)?;

        let mut registry = VectorRegistry::new();

        // 1. Base state from the most recent snapshot.
        for block in snapshot::read(&dir.join(SNAPSHOT_FILE))? {
            apply_snapshot_block(&mut registry, block);
        }
        // 2. Everything logged since that snapshot.
        for record in Wal::replay(&dir.join(WAL_FILE))? {
            if let Some(op) = Op::decode(&record) {
                op.apply(&mut registry);
            }
        }

        let wal = Wal::open(&dir.join(WAL_FILE))?;
        Ok((Self { dir, wal }, registry))
    }

    /// Logs the creation of an index.
    pub fn log_new(&mut self, name: &[u8], dim: usize, metric: Metric) -> io::Result<()> {
        self.wal
            .append(&Op::encode_new(name, dim as u32, metric.to_u8()))
    }

    /// Logs an inserted vector. `vec` is the packed-f32 payload as received.
    pub fn log_add(&mut self, name: &[u8], key: &[u8], vec: &[u8]) -> io::Result<()> {
        self.wal.append(&Op::encode_add(name, key, vec))
    }

    /// Logs a deletion.
    pub fn log_del(&mut self, name: &[u8], key: &[u8]) -> io::Result<()> {
        self.wal.append(&Op::encode_del(name, key))
    }

    /// Writes a fresh snapshot of `registry` and truncates the log. The
    /// snapshot is committed (via rename) before the log is cleared, so a crash
    /// in between leaves a redundant-but-harmless log to replay.
    pub fn save(&mut self, registry: &VectorRegistry) -> io::Result<()> {
        let blocks: Vec<SnapBlock> = registry
            .iter()
            .map(|(name, index)| SnapBlock {
                name: name.clone(),
                dim: index.dim() as u32,
                metric: index.metric().to_u8(),
                entries: index
                    .entries()
                    .into_iter()
                    .map(|(key, v)| (key.to_vec(), vector::encode(v)))
                    .collect(),
            })
            .collect();

        snapshot::write(&self.dir.join(SNAPSHOT_FILE), &blocks)?;
        self.wal.truncate()
    }
}

fn apply_snapshot_block(registry: &mut VectorRegistry, block: SnapBlock) {
    let metric = match Metric::from_u8(block.metric) {
        Some(m) => m,
        None => return,
    };
    let dim = block.dim as usize;
    registry.create(block.name.clone(), dim, metric);
    if let Some(index) = registry.get_mut(&block.name) {
        for (key, raw) in block.entries {
            if let Some(v) = vector::decode(&raw)
                && v.len() == dim
            {
                index.add(key, v);
            }
        }
    }
}

/// A logged mutation.
enum Op {
    New {
        name: Vec<u8>,
        dim: u32,
        metric: u8,
    },
    Add {
        name: Vec<u8>,
        key: Vec<u8>,
        vec: Vec<u8>,
    },
    Del {
        name: Vec<u8>,
        key: Vec<u8>,
    },
}

fn put_lp(buf: &mut Vec<u8>, bytes: &[u8]) {
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

/// Reads a length-prefixed slice from `data` at `*pos`, advancing `*pos`.
fn get_lp<'a>(data: &'a [u8], pos: &mut usize) -> Option<&'a [u8]> {
    let end = pos.checked_add(4)?;
    if end > data.len() {
        return None;
    }
    let len = u32::from_le_bytes(data[*pos..end].try_into().unwrap()) as usize;
    let start = end;
    let stop = start.checked_add(len)?;
    if stop > data.len() {
        return None;
    }
    *pos = stop;
    Some(&data[start..stop])
}

impl Op {
    fn encode_new(name: &[u8], dim: u32, metric: u8) -> Vec<u8> {
        let mut buf = vec![OP_NEW];
        put_lp(&mut buf, name);
        buf.extend_from_slice(&dim.to_le_bytes());
        buf.push(metric);
        buf
    }

    fn encode_add(name: &[u8], key: &[u8], vec: &[u8]) -> Vec<u8> {
        let mut buf = vec![OP_ADD];
        put_lp(&mut buf, name);
        put_lp(&mut buf, key);
        put_lp(&mut buf, vec);
        buf
    }

    fn encode_del(name: &[u8], key: &[u8]) -> Vec<u8> {
        let mut buf = vec![OP_DEL];
        put_lp(&mut buf, name);
        put_lp(&mut buf, key);
        buf
    }

    fn decode(record: &[u8]) -> Option<Op> {
        let (&opcode, rest) = record.split_first()?;
        let mut pos = 0;
        match opcode {
            OP_NEW => {
                let name = get_lp(rest, &mut pos)?.to_vec();
                let dim_end = pos.checked_add(4)?;
                if dim_end > rest.len() {
                    return None;
                }
                let dim = u32::from_le_bytes(rest[pos..dim_end].try_into().unwrap());
                let metric = *rest.get(dim_end)?;
                Some(Op::New { name, dim, metric })
            }
            OP_ADD => {
                let name = get_lp(rest, &mut pos)?.to_vec();
                let key = get_lp(rest, &mut pos)?.to_vec();
                let vec = get_lp(rest, &mut pos)?.to_vec();
                Some(Op::Add { name, key, vec })
            }
            OP_DEL => {
                let name = get_lp(rest, &mut pos)?.to_vec();
                let key = get_lp(rest, &mut pos)?.to_vec();
                Some(Op::Del { name, key })
            }
            _ => None,
        }
    }

    fn apply(self, registry: &mut VectorRegistry) {
        match self {
            Op::New { name, dim, metric } => {
                if let Some(m) = Metric::from_u8(metric) {
                    registry.create(name, dim as usize, m);
                }
            }
            Op::Add { name, key, vec } => {
                if let Some(index) = registry.get_mut(&name)
                    && let Some(v) = vector::decode(&vec)
                    && v.len() == index.dim()
                {
                    index.add(key, v);
                }
            }
            Op::Del { name, key } => {
                if let Some(index) = registry.get_mut(&name) {
                    index.remove(&key);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_new_roundtrips() {
        let enc = Op::encode_new(b"mem", 8, 1);
        match Op::decode(&enc) {
            Some(Op::New { name, dim, metric }) => {
                assert_eq!(name, b"mem");
                assert_eq!(dim, 8);
                assert_eq!(metric, 1);
            }
            _ => panic!("decode failed"),
        }
    }

    #[test]
    fn op_add_roundtrips() {
        let enc = Op::encode_add(b"mem", b"k", &[1, 2, 3, 4]);
        match Op::decode(&enc) {
            Some(Op::Add { name, key, vec }) => {
                assert_eq!(name, b"mem");
                assert_eq!(key, b"k");
                assert_eq!(vec, vec![1, 2, 3, 4]);
            }
            _ => panic!("decode failed"),
        }
    }

    #[test]
    fn op_del_roundtrips() {
        let enc = Op::encode_del(b"mem", b"k");
        match Op::decode(&enc) {
            Some(Op::Del { name, key }) => {
                assert_eq!(name, b"mem");
                assert_eq!(key, b"k");
            }
            _ => panic!("decode failed"),
        }
    }

    #[test]
    fn decode_rejects_truncated() {
        let enc = Op::encode_add(b"mem", b"k", &[1, 2, 3, 4]);
        assert!(Op::decode(&enc[..enc.len() - 2]).is_none());
        assert!(Op::decode(&[]).is_none());
        assert!(Op::decode(&[99]).is_none());
    }
}
