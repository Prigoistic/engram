//! The backing store for vector data.
//!
//! Vectors are kept in one contiguous `f32` buffer, `len * dim` elements long,
//! so that vector `id` occupies the slice `[id * dim, (id + 1) * dim)`. A flat
//! layout keeps each vector cache-friendly for the distance kernels and maps
//! cleanly onto an `mmap`-backed file in a later phase.

/// A growable, fixed-dimension collection of `f32` vectors.
pub struct VectorStore {
    /// The dimension of every vector held here.
    dim: usize,

    /// All vectors concatenated, row-major: `data[id * dim ..][.. dim]`.
    data: Vec<f32>,
}

impl VectorStore {
    /// Creates an empty store for `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            data: Vec::new(),
        }
    }

    /// How many vectors are stored, including any later tombstoned by the index.
    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    /// Appends a vector and returns its new id.
    pub fn push(&mut self, v: &[f32]) -> u32 {
        debug_assert_eq!(v.len(), self.dim);
        let id = self.len() as u32;
        self.data.extend_from_slice(v);
        id
    }

    /// Borrows the vector stored at `id`.
    pub fn get(&self, id: u32) -> &[f32] {
        let start = id as usize * self.dim;
        &self.data[start..start + self.dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_assigns_sequential_ids() {
        let mut s = VectorStore::new(2);
        assert_eq!(s.push(&[1.0, 2.0]), 0);
        assert_eq!(s.push(&[3.0, 4.0]), 1);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn get_returns_stored_slice() {
        let mut s = VectorStore::new(3);
        s.push(&[1.0, 2.0, 3.0]);
        s.push(&[4.0, 5.0, 6.0]);
        assert_eq!(s.get(0), &[1.0, 2.0, 3.0]);
        assert_eq!(s.get(1), &[4.0, 5.0, 6.0]);
    }
}
