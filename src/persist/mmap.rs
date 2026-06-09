//! A minimal read-only memory map over `libc::mmap`.
//!
//! Used to load a snapshot: the file is mapped into the address space once and
//! its bytes are read straight from the mapping, letting the OS page the data
//! in on demand instead of issuing read syscalls. The mapping is private and
//! read-only, and is unmapped on drop.

use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::ptr;
use std::slice;

/// A read-only `mmap` of a whole file.
pub struct MmapRo {
    ptr: *const u8,
    len: usize,
    // Kept alive for the lifetime of the mapping. The mapping itself survives
    // closing the fd, but holding the file is tidy and keeps the path open.
    _file: File,
}

impl MmapRo {
    /// Maps the entire file at `path` read-only. An empty file maps to an empty
    /// slice with no actual mapping.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Ok(Self {
                ptr: ptr::null(),
                len: 0,
                _file: file,
            });
        }

        // SAFETY: a fresh, page-aligned mapping of `len` readable bytes backed
        // by a valid fd. We check for MAP_FAILED before trusting the pointer.
        let addr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        Ok(Self {
            ptr: addr as *const u8,
            len,
            _file: file,
        })
    }

    /// The mapped bytes.
    pub fn bytes(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: `ptr` covers `len` readable bytes for the lifetime of
            // `self`, and is non-null and page-aligned when `len > 0`.
            unsafe { slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

impl Drop for MmapRo {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            // SAFETY: `ptr`/`len` are exactly the region returned by `mmap`.
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.len);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn maps_and_reads_back() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("engram_mmap_test_{}.bin", std::process::id()));
        let data: Vec<u8> = (0..4096u32).map(|i| i as u8).collect();
        File::create(&path).unwrap().write_all(&data).unwrap();

        let map = MmapRo::open(&path).unwrap();
        assert_eq!(map.bytes(), data.as_slice());
        drop(map);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_file_maps_to_empty_slice() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("engram_mmap_empty_{}.bin", std::process::id()));
        File::create(&path).unwrap();
        let map = MmapRo::open(&path).unwrap();
        assert!(map.bytes().is_empty());
        std::fs::remove_file(&path).ok();
    }
}
