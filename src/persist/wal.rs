//! A write-ahead log of mutation records.
//!
//! Each record is framed as `[crc32 u32][len u32][payload len bytes]`, all
//! little-endian, and the file is flushed to disk after every append. On
//! replay, records are read until one is short or fails its checksum, which is
//! exactly the signature of a write torn by a crash; everything up to that
//! point is durable and is returned.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

/// IEEE CRC-32 of `data` (bit-reversed, polynomial `0xEDB88320`).
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// An append-only, flushed write-ahead log.
pub struct Wal {
    file: File,
}

impl Wal {
    /// Opens (creating if needed) the log at `path` for appending.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(path)?;
        Ok(Self { file })
    }

    /// Appends one record and flushes it to stable storage.
    pub fn append(&mut self, payload: &[u8]) -> io::Result<()> {
        let mut frame = Vec::with_capacity(8 + payload.len());
        frame.extend_from_slice(&crc32(payload).to_le_bytes());
        frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        frame.extend_from_slice(payload);
        self.file.write_all(&frame)?;
        // Durability: ensure the bytes reach the device before we report success.
        self.file.sync_data()
    }

    /// Discards every record. Called after a snapshot has captured the full
    /// state, so the log can start fresh.
    pub fn truncate(&mut self) -> io::Result<()> {
        self.file.set_len(0)?;
        self.file.sync_data()
    }

    /// Reads every intact record from the log at `path`, stopping at the first
    /// truncated or corrupt record (a crash-torn tail). Returns an empty list
    /// if the file does not exist.
    pub fn replay(path: &Path) -> io::Result<Vec<Vec<u8>>> {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let mut records = Vec::new();
        let mut pos = 0;
        while pos + 8 <= bytes.len() {
            let crc = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
            let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
            let start = pos + 8;
            let end = start + len;
            if end > bytes.len() {
                break; // truncated tail
            }
            let payload = &bytes[start..end];
            if crc32(payload) != crc {
                break; // corrupt tail
            }
            records.push(payload.to_vec());
            pos = end;
        }
        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("engram_wal_{}_{}.log", name, std::process::id()))
    }

    #[test]
    fn crc32_known_value() {
        // "123456789" => 0xCBF43926, the canonical CRC-32 check value.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn append_then_replay_roundtrips() {
        let path = tmp("roundtrip");
        std::fs::remove_file(&path).ok();
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append(b"alpha").unwrap();
            wal.append(b"beta").unwrap();
            wal.append(b"").unwrap();
            wal.append(b"gamma").unwrap();
        }
        let recs = Wal::replay(&path).unwrap();
        assert_eq!(
            recs,
            vec![
                b"alpha".to_vec(),
                b"beta".to_vec(),
                vec![],
                b"gamma".to_vec()
            ]
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn replay_stops_at_torn_tail() {
        let path = tmp("torn");
        std::fs::remove_file(&path).ok();
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append(b"good1").unwrap();
            wal.append(b"good2").unwrap();
        }
        // Simulate a crash mid-write: a header promising 100 bytes, then 3.
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&crc32(b"x").to_le_bytes()).unwrap();
            f.write_all(&100u32.to_le_bytes()).unwrap();
            f.write_all(b"abc").unwrap();
        }
        let recs = Wal::replay(&path).unwrap();
        assert_eq!(recs, vec![b"good1".to_vec(), b"good2".to_vec()]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn replay_stops_at_bad_crc() {
        let path = tmp("badcrc");
        std::fs::remove_file(&path).ok();
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append(b"ok").unwrap();
        }
        {
            // A full frame whose crc does not match the payload.
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&0xDEAD_BEEFu32.to_le_bytes()).unwrap();
            f.write_all(&4u32.to_le_bytes()).unwrap();
            f.write_all(b"junk").unwrap();
        }
        assert_eq!(Wal::replay(&path).unwrap(), vec![b"ok".to_vec()]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn truncate_clears_records() {
        let path = tmp("trunc");
        std::fs::remove_file(&path).ok();
        let mut wal = Wal::open(&path).unwrap();
        wal.append(b"a").unwrap();
        wal.truncate().unwrap();
        wal.append(b"b").unwrap();
        assert_eq!(Wal::replay(&path).unwrap(), vec![b"b".to_vec()]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn replay_missing_file_is_empty() {
        let path = tmp("missing");
        std::fs::remove_file(&path).ok();
        assert!(Wal::replay(&path).unwrap().is_empty());
    }
}
