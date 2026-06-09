//! Associative vector memory: named indices of dense vectors with
//! nearest-neighbour search.
//!
//! The [`VectorRegistry`] owns every named [`Index`] and lives in the shared
//! server state alongside the key-value store. Vectors travel on the wire as
//! packed little-endian `f32` bulk strings; [`decode`] and [`encode`] are the
//! boundary between those bytes and the `f32` slices the kernels work on.

mod hnsw;
mod index;
mod kernels;
mod metric;
mod rng;
mod store;

pub use index::Index;
pub use metric::Metric;

use std::collections::HashMap;

/// Every named vector index on the server.
#[derive(Default)]
pub struct VectorRegistry {
    indices: HashMap<Vec<u8>, Index>,
}

impl VectorRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new index `name`. Returns `false` if one already exists.
    pub fn create(&mut self, name: Vec<u8>, dim: usize, metric: Metric) -> bool {
        if self.indices.contains_key(&name) {
            return false;
        }
        self.indices.insert(name, Index::new(dim, metric));
        true
    }

    /// Borrows the index named `name`, if it exists.
    pub fn get(&self, name: &[u8]) -> Option<&Index> {
        self.indices.get(name)
    }

    /// Mutably borrows the index named `name`, if it exists.
    pub fn get_mut(&mut self, name: &[u8]) -> Option<&mut Index> {
        self.indices.get_mut(name)
    }

    /// Iterates over `(name, index)` for every index, for snapshotting.
    pub fn iter(&self) -> impl Iterator<Item = (&Vec<u8>, &Index)> {
        self.indices.iter()
    }
}

/// Decodes a packed little-endian `f32` vector. Returns `None` if the byte
/// length is not a positive multiple of four or any component is not finite.
pub fn decode(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut v = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !f.is_finite() {
            return None;
        }
        v.push(f);
    }
    Some(v)
}

/// Encodes a vector as packed little-endian `f32` bytes. The inverse of
/// [`decode`]; used on the snapshot write path and by tests to build vector
/// arguments.
pub fn encode(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let v = vec![1.0, -2.5, 3.25, 0.0];
        assert_eq!(decode(&encode(&v)), Some(v));
    }

    #[test]
    fn decode_rejects_non_multiple_of_four() {
        assert_eq!(decode(&[0, 1, 2]), None);
        assert_eq!(decode(&[0, 1, 2, 3, 4]), None);
    }

    #[test]
    fn decode_rejects_empty() {
        assert_eq!(decode(&[]), None);
    }

    #[test]
    fn decode_rejects_nan_and_inf() {
        assert_eq!(decode(&encode(&[f32::NAN])), None);
        assert_eq!(decode(&encode(&[f32::INFINITY])), None);
    }

    #[test]
    fn registry_create_is_idempotent_guarded() {
        let mut reg = VectorRegistry::new();
        assert!(reg.create(b"m".to_vec(), 4, Metric::Cosine));
        assert!(!reg.create(b"m".to_vec(), 4, Metric::Cosine));
        assert!(reg.get(b"m").is_some());
        assert_eq!(reg.get(b"m").unwrap().dim(), 4);
        assert!(reg.get(b"missing").is_none());
    }
}
