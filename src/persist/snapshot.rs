//! A point-in-time snapshot of every index.
//!
//! The snapshot is the compaction target for the [`super::wal::Wal`]: it holds
//! only live entries, so tombstones from deletes and overwrites are dropped
//! when it is written. It is written to a temp file and `rename`d into place so
//! a reader never sees a half-written snapshot, and it is read back through a
//! memory map.
//!
//! Layout (all integers little-endian):
//! ```text
//! magic "EGSNAPv1" | nblocks u32
//! per block: name(lp) | dim u32 | metric u8 | count u32
//!            count * ( key(lp) | vector(lp) )      // vector = packed f32 bytes
//! ```
//! where `lp` is a `u32` length followed by that many bytes.

use std::io::{self, Write};
use std::path::Path;

use super::mmap::MmapRo;

const MAGIC: &[u8; 8] = b"EGSNAPv1";

/// One index's worth of snapshot data.
pub struct SnapBlock {
    pub name: Vec<u8>,
    pub dim: u32,
    pub metric: u8,
    /// Live `(key, packed-f32-vector-bytes)` pairs.
    pub entries: Vec<(Vec<u8>, Vec<u8>)>,
}

fn put_lp(buf: &mut Vec<u8>, bytes: &[u8]) {
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

/// Atomically writes `blocks` to the snapshot at `path`.
pub fn write(path: &Path, blocks: &[SnapBlock]) -> io::Result<()> {
    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
    for b in blocks {
        put_lp(&mut buf, &b.name);
        buf.extend_from_slice(&b.dim.to_le_bytes());
        buf.push(b.metric);
        buf.extend_from_slice(&(b.entries.len() as u32).to_le_bytes());
        for (key, vec) in &b.entries {
            put_lp(&mut buf, key);
            put_lp(&mut buf, vec);
        }
    }

    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&buf)?;
        f.sync_data()?;
    }
    // rename is atomic on the same filesystem: readers see old or new, never half.
    std::fs::rename(&tmp, path)
}

/// A bounds-checked forward reader over a byte slice.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> Option<u32> {
        let end = self.pos.checked_add(4)?;
        if end > self.data.len() {
            return None;
        }
        let v = u32::from_le_bytes(self.data[self.pos..end].try_into().unwrap());
        self.pos = end;
        Some(v)
    }

    fn u8(&mut self) -> Option<u8> {
        let b = *self.data.get(self.pos)?;
        self.pos += 1;
        Some(b)
    }

    fn lp(&mut self) -> Option<&'a [u8]> {
        let len = self.u32()? as usize;
        let end = self.pos.checked_add(len)?;
        if end > self.data.len() {
            return None;
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Some(slice)
    }
}

/// Parses the snapshot bytes into blocks, or `None` if it is malformed.
fn parse(bytes: &[u8]) -> Option<Vec<SnapBlock>> {
    if bytes.len() < 12 || &bytes[..8] != MAGIC {
        return None;
    }
    let mut r = Reader {
        data: bytes,
        pos: 8,
    };
    let nblocks = r.u32()?;
    let mut blocks = Vec::with_capacity(nblocks as usize);
    for _ in 0..nblocks {
        let name = r.lp()?.to_vec();
        let dim = r.u32()?;
        let metric = r.u8()?;
        let count = r.u32()?;
        let mut entries = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let key = r.lp()?.to_vec();
            let vec = r.lp()?.to_vec();
            entries.push((key, vec));
        }
        blocks.push(SnapBlock {
            name,
            dim,
            metric,
            entries,
        });
    }
    Some(blocks)
}

/// Reads the snapshot at `path` through a memory map. Returns an empty list if
/// the file is absent, and an error if it exists but is unreadable or corrupt.
pub fn read(path: &Path) -> io::Result<Vec<SnapBlock>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let map = MmapRo::open(path)?;
    parse(map.bytes()).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "corrupt snapshot"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("engram_snap_{}_{}.bin", name, std::process::id()))
    }

    #[test]
    fn write_then_read_roundtrips() {
        let path = tmp("roundtrip");
        std::fs::remove_file(&path).ok();

        let blocks = vec![
            SnapBlock {
                name: b"mem".to_vec(),
                dim: 2,
                metric: 1,
                entries: vec![
                    (b"a".to_vec(), vec![1, 2, 3, 4, 5, 6, 7, 8]),
                    (b"b".to_vec(), vec![9, 10, 11, 12, 13, 14, 15, 16]),
                ],
            },
            SnapBlock {
                name: b"other".to_vec(),
                dim: 1,
                metric: 0,
                entries: vec![],
            },
        ];
        write(&path, &blocks).unwrap();

        let got = read(&path).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].name, b"mem");
        assert_eq!(got[0].dim, 2);
        assert_eq!(got[0].metric, 1);
        assert_eq!(got[0].entries.len(), 2);
        assert_eq!(got[0].entries[1].0, b"b");
        assert_eq!(got[0].entries[1].1, vec![9, 10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(got[1].name, b"other");
        assert!(got[1].entries.is_empty());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_missing_is_empty() {
        let path = tmp("missing");
        std::fs::remove_file(&path).ok();
        assert!(read(&path).unwrap().is_empty());
    }

    #[test]
    fn corrupt_magic_is_error() {
        let path = tmp("corrupt");
        std::fs::write(&path, b"NOTMAGIC....").unwrap();
        assert!(read(&path).is_err());
        std::fs::remove_file(&path).ok();
    }
}
