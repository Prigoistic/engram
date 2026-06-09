//! A single named vector index.
//!
//! An [`Index`] maps client keys (arbitrary bytes) to dense internal ids, holds
//! the vectors in a [`VectorStore`], and answers nearest-neighbour queries
//! through an [`Hnsw`] graph built over those ids.
//!
//! [`Index::search`] is approximate (it walks the graph); [`Index::search_exact`]
//! is the exhaustive scan kept as the ground truth the graph is measured
//! against in the recall tests.

use std::collections::HashMap;

use super::hnsw::Hnsw;
use super::metric::{self, Metric};
use super::store::VectorStore;

/// HNSW out-degree on the upper layers.
const HNSW_M: usize = 16;
/// HNSW candidate-list size during construction.
const HNSW_EF_CONSTRUCTION: usize = 200;
/// Fixed seed so a given insert order yields a reproducible graph.
const HNSW_SEED: u64 = 0x5eed_1234;

/// A nearest-neighbour result: the client key and its distance to the query.
#[derive(Debug, Clone, PartialEq)]
pub struct Neighbor {
    /// The client-supplied key.
    pub key: Vec<u8>,

    /// The distance to the query under the index's metric; smaller is closer.
    pub distance: f32,
}

/// A named collection of equal-dimension vectors that supports search.
pub struct Index {
    /// The dimension every vector in this index must have.
    dim: usize,

    /// How distances are measured.
    metric: Metric,

    /// The vectors, addressed by internal id.
    store: VectorStore,

    /// The navigable small-world graph over the internal ids.
    hnsw: Hnsw,

    /// Internal id to client key, or `None` once a key has been deleted. A
    /// tombstoned node stays in the graph for connectivity but is filtered out
    /// of results.
    keys: Vec<Option<Vec<u8>>>,

    /// Client key to internal id, for the live keys only.
    key_to_id: HashMap<Vec<u8>, u32>,
}

impl Index {
    /// Creates an empty index over `dim`-dimensional vectors.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            metric,
            store: VectorStore::new(dim),
            hnsw: Hnsw::new(HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_SEED),
            keys: Vec::new(),
            key_to_id: HashMap::new(),
        }
    }

    /// The required vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The distance metric.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// How many live keys the index holds.
    pub fn len(&self) -> usize {
        self.key_to_id.len()
    }

    /// The live `(key, vector)` pairs, for snapshotting. Vectors are returned
    /// as stored, which for a normalising metric means already normalised.
    pub fn entries(&self) -> Vec<(&[u8], &[f32])> {
        self.keys
            .iter()
            .enumerate()
            .filter_map(|(id, key)| {
                key.as_ref()
                    .map(|key| (key.as_slice(), self.store.get(id as u32)))
            })
            .collect()
    }

    /// Inserts or overwrites `key` with `vec`, returning `true` if the key is
    /// new. `vec`'s length must equal [`Index::dim`].
    ///
    /// Overwriting tombstones the previous node and inserts a fresh one, rather
    /// than mutating a vector in place, so the graph's neighbour distances stay
    /// consistent with the data.
    pub fn add(&mut self, key: Vec<u8>, mut vec: Vec<f32>) -> bool {
        debug_assert_eq!(vec.len(), self.dim);
        if self.metric.normalizes() {
            metric::normalize(&mut vec);
        }

        let is_new = match self.key_to_id.remove(&key) {
            Some(old) => {
                self.keys[old as usize] = None;
                false
            }
            None => true,
        };

        let id = self.store.push(&vec);
        debug_assert_eq!(id as usize, self.keys.len());
        self.keys.push(Some(key.clone()));
        self.key_to_id.insert(key, id);

        // Disjoint borrows: the graph (mutable) and the store (shared) are
        // separate fields, so the distance closure can read vectors while the
        // graph mutates its topology.
        let metric = self.metric;
        let store = &self.store;
        let hnsw = &mut self.hnsw;
        hnsw.insert(id, |x, y| metric.distance(store.get(x), store.get(y)));

        is_new
    }

    /// Removes `key`, returning `true` if it was present.
    pub fn remove(&mut self, key: &[u8]) -> bool {
        match self.key_to_id.remove(key) {
            Some(id) => {
                self.keys[id as usize] = None;
                true
            }
            None => false,
        }
    }

    /// Returns the `k` nearest neighbours to `query` via the graph, closest
    /// first. `ef` is the search breadth (clamped up to at least `k`); larger
    /// values trade speed for recall. `query`'s length must equal
    /// [`Index::dim`].
    pub fn search(&self, mut query: Vec<f32>, k: usize, ef: usize) -> Vec<Neighbor> {
        debug_assert_eq!(query.len(), self.dim);
        if k == 0 {
            return Vec::new();
        }
        if self.metric.normalizes() {
            metric::normalize(&mut query);
        }

        let metric = self.metric;
        let store = &self.store;
        let candidates = self
            .hnsw
            .search(|id| metric.distance(&query, store.get(id)), k, ef);

        let mut out = Vec::with_capacity(k);
        for (id, distance) in candidates {
            if let Some(Some(key)) = self.keys.get(id as usize) {
                out.push(Neighbor {
                    key: key.clone(),
                    distance,
                });
                if out.len() == k {
                    break;
                }
            }
        }
        out
    }

    /// The exhaustive nearest-neighbour scan: ground truth for the graph. Kept
    /// for the recall tests; could also back an exact-search command option.
    #[allow(dead_code)]
    pub fn search_exact(&self, mut query: Vec<f32>, k: usize) -> Vec<Neighbor> {
        debug_assert_eq!(query.len(), self.dim);
        if k == 0 {
            return Vec::new();
        }
        if self.metric.normalizes() {
            metric::normalize(&mut query);
        }

        let mut scored: Vec<Neighbor> = self
            .keys
            .iter()
            .enumerate()
            .filter_map(|(id, key)| key.as_ref().map(|key| (id, key)))
            .map(|(id, key)| Neighbor {
                key: key.clone(),
                distance: self.metric.distance(&query, self.store.get(id as u32)),
            })
            .collect();

        scored.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(neighbors: &[Neighbor]) -> Vec<&[u8]> {
        neighbors.iter().map(|n| n.key.as_slice()).collect()
    }

    /// Deterministic vector source for the recall tests.
    struct Lcg(u64);
    impl Lcg {
        fn f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
        fn vec(&mut self, d: usize) -> Vec<f32> {
            (0..d).map(|_| self.f32()).collect()
        }
    }

    #[test]
    fn add_reports_new_then_overwrite() {
        let mut idx = Index::new(2, Metric::L2);
        assert!(idx.add(b"a".to_vec(), vec![1.0, 0.0]));
        assert!(!idx.add(b"a".to_vec(), vec![0.0, 1.0]));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn search_orders_by_distance_l2() {
        let mut idx = Index::new(2, Metric::L2);
        idx.add(b"origin".to_vec(), vec![0.0, 0.0]);
        idx.add(b"near".to_vec(), vec![1.0, 0.0]);
        idx.add(b"far".to_vec(), vec![10.0, 0.0]);

        let got = idx.search(vec![0.0, 0.0], 3, 32);
        assert_eq!(keys(&got), vec![&b"origin"[..], b"near", b"far"]);
    }

    #[test]
    fn search_respects_k() {
        let mut idx = Index::new(1, Metric::L2);
        for i in 0..10u8 {
            idx.add(vec![i], vec![i as f32]);
        }
        let got = idx.search(vec![0.0], 3, 32);
        assert_eq!(got.len(), 3);
    }

    #[test]
    fn k_zero_returns_empty() {
        let mut idx = Index::new(1, Metric::L2);
        idx.add(b"a".to_vec(), vec![1.0]);
        assert!(idx.search(vec![0.0], 0, 32).is_empty());
    }

    #[test]
    fn cosine_ignores_magnitude() {
        let mut idx = Index::new(2, Metric::Cosine);
        idx.add(b"same_dir".to_vec(), vec![10.0, 0.0]);
        idx.add(b"orthogonal".to_vec(), vec![0.0, 5.0]);

        let got = idx.search(vec![1.0, 0.0], 2, 32);
        assert_eq!(got[0].key, b"same_dir");
        assert!(got[0].distance < 1e-5);
    }

    #[test]
    fn overwrite_changes_results() {
        let mut idx = Index::new(2, Metric::L2);
        idx.add(b"x".to_vec(), vec![5.0, 5.0]);
        idx.add(b"x".to_vec(), vec![0.0, 0.0]);
        let got = idx.search(vec![0.0, 0.0], 1, 32);
        assert_eq!(got[0].key, b"x");
        assert!(got[0].distance < 1e-5);
    }

    #[test]
    fn removed_key_is_excluded() {
        let mut idx = Index::new(1, Metric::L2);
        idx.add(b"a".to_vec(), vec![0.0]);
        idx.add(b"b".to_vec(), vec![1.0]);
        assert!(idx.remove(b"a"));
        assert!(!idx.remove(b"a"));

        let got = idx.search(vec![0.0], 5, 32);
        assert_eq!(keys(&got), vec![&b"b"[..]]);
        assert_eq!(idx.len(), 1);
    }

    /// On a small index, a generous `ef` should make the graph match the exact
    /// scan exactly.
    #[test]
    fn small_index_is_exact_with_large_ef() {
        let mut rng = Lcg(1);
        let mut idx = Index::new(8, Metric::L2);
        let mut vecs = Vec::new();
        for i in 0..40u32 {
            let v = rng.vec(8);
            idx.add(i.to_le_bytes().to_vec(), v.clone());
            vecs.push(v);
        }
        for _ in 0..20 {
            let q = rng.vec(8);
            let approx = keys(&idx.search(q.clone(), 5, 200))
                .iter()
                .map(|k| k.to_vec())
                .collect::<Vec<_>>();
            let exact = keys(&idx.search_exact(q, 5))
                .iter()
                .map(|k| k.to_vec())
                .collect::<Vec<_>>();
            assert_eq!(approx, exact);
        }
    }

    /// Recall@10 against the exact scan on a larger random set. The seed is
    /// fixed, so this asserts a stable floor rather than a flaky range.
    #[test]
    fn recall_matches_bruteforce() {
        let mut rng = Lcg(0xC0FFEE);
        let (n, dim, k, ef) = (2000usize, 16usize, 10usize, 128usize);

        let mut idx = Index::new(dim, Metric::L2);
        for i in 0..n as u32 {
            idx.add(i.to_le_bytes().to_vec(), rng.vec(dim));
        }

        let queries = 200;
        let mut hits = 0usize;
        for _ in 0..queries {
            let q = rng.vec(dim);
            let approx: Vec<_> = idx
                .search(q.clone(), k, ef)
                .into_iter()
                .map(|n| n.key)
                .collect();
            let exact: std::collections::HashSet<_> =
                idx.search_exact(q, k).into_iter().map(|n| n.key).collect();
            hits += approx.iter().filter(|key| exact.contains(*key)).count();
        }

        let recall = hits as f64 / (queries * k) as f64;
        println!("HNSW recall@{k} = {recall:.4} (n={n}, dim={dim}, ef={ef})");
        assert!(recall >= 0.90, "recall too low: {recall:.4}");
    }
}
