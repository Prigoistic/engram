//! A hand-rolled Hierarchical Navigable Small World graph.
//!
//! This follows Malkov & Yashunin, "Efficient and robust approximate nearest
//! neighbor search using Hierarchical Navigable Small World graphs" (2018):
//! `search_layer` (Alg. 2), the neighbour-selection heuristic with kept pruned
//! connections (Alg. 4), and the insert routine (Alg. 1).
//!
//! The graph stores only topology — node levels and per-layer adjacency. It
//! never sees a vector: every distance arrives through a caller-supplied
//! closure, so the vector data can live wherever the [`super::index::Index`]
//! keeps it (the heap today, an `mmap` later) without this module changing.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use super::rng::Rng;

/// An `(id, distance)` pair ordered by distance. Used inside the search heaps.
#[derive(Clone, Copy)]
struct Candidate {
    dist: f32,
    id: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Total order over the distance so the heaps behave with any float.
        self.dist.total_cmp(&other.dist)
    }
}

/// One node's per-layer adjacency. `layers[l]` is the neighbour list at layer
/// `l`, for `l` in `0..=node_level`.
struct NodeLinks {
    layers: Vec<Vec<u32>>,
}

/// A Hierarchical Navigable Small World graph over dense node ids `0..n`.
pub struct Hnsw {
    /// Target out-degree on layers above 0.
    m: usize,
    /// Out-degree on layer 0 (denser, conventionally `2 * m`).
    m0: usize,
    /// Candidate-list size used while inserting.
    ef_construction: usize,
    /// Level-generation scale, `1 / ln(m)`.
    ml: f64,
    rng: Rng,
    entry: Option<u32>,
    max_level: usize,
    nodes: Vec<NodeLinks>,
}

impl Hnsw {
    /// Creates an empty graph with degree `m` and construction breadth
    /// `ef_construction`, seeded deterministically by `seed`.
    pub fn new(m: usize, ef_construction: usize, seed: u64) -> Self {
        Self {
            m,
            m0: m * 2,
            ef_construction,
            ml: 1.0 / (m as f64).ln(),
            rng: Rng::new(seed),
            entry: None,
            max_level: 0,
            nodes: Vec::new(),
        }
    }

    /// The neighbours of `id` at `layer`, or an empty slice if `id` does not
    /// reach that layer.
    fn neighbors(&self, id: u32, layer: usize) -> &[u32] {
        match self.nodes[id as usize].layers.get(layer) {
            Some(list) => list,
            None => &[],
        }
    }

    /// Draws a node level from the exponential distribution.
    fn random_level(&mut self) -> usize {
        (-self.rng.unit().ln() * self.ml) as usize
    }

    /// Greedy best-first search confined to one layer (Alg. 2). Returns the up
    /// to `ef` closest nodes found, ascending by distance. `dist` gives the
    /// distance from the query to a node id.
    fn search_layer<F: Fn(u32) -> f32>(
        &self,
        dist: &F,
        entry: &[u32],
        ef: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let mut visited: HashSet<u32> = HashSet::with_capacity(ef * 4);
        // Frontier, explored nearest-first.
        let mut frontier: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        // Results, kept as a max-heap so the farthest is cheap to drop.
        let mut found: BinaryHeap<Candidate> = BinaryHeap::new();

        for &e in entry {
            let c = Candidate {
                dist: dist(e),
                id: e,
            };
            visited.insert(e);
            frontier.push(Reverse(c));
            found.push(c);
        }
        while found.len() > ef {
            found.pop();
        }

        while let Some(Reverse(c)) = frontier.pop() {
            // The current worst result; `found` is seeded, so never empty.
            let worst = found.peek().map(|f| f.dist).unwrap_or(f32::INFINITY);
            if c.dist > worst {
                break;
            }
            // Clone the small neighbour list to release the borrow on `self`.
            for &e in &self.neighbors(c.id, layer).to_vec() {
                if visited.insert(e) {
                    let d = dist(e);
                    let worst = found.peek().map(|f| f.dist).unwrap_or(f32::INFINITY);
                    if d < worst || found.len() < ef {
                        let cand = Candidate { dist: d, id: e };
                        frontier.push(Reverse(cand));
                        found.push(cand);
                        if found.len() > ef {
                            found.pop();
                        }
                    }
                }
            }
        }

        let mut out = found.into_vec();
        out.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        out
    }

    /// The neighbour-selection heuristic (Alg. 4). `candidates` carry their
    /// distance to the base node; `dist_nodes` gives node-to-node distances.
    /// Keeps a candidate only if it is closer to the base than to every
    /// already-chosen neighbour, then backfills from the discarded set so the
    /// result reaches `m` when possible (kept pruned connections).
    fn select_neighbors<F: Fn(u32, u32) -> f32>(
        &self,
        mut candidates: Vec<Candidate>,
        m: usize,
        dist_nodes: &F,
    ) -> Vec<u32> {
        candidates.sort_by(|a, b| a.dist.total_cmp(&b.dist));

        let mut chosen: Vec<Candidate> = Vec::with_capacity(m);
        let mut discarded: Vec<Candidate> = Vec::new();
        for c in candidates {
            if chosen.len() >= m {
                break;
            }
            let diverse = chosen.iter().all(|r| c.dist < dist_nodes(c.id, r.id));
            if diverse {
                chosen.push(c);
            } else {
                discarded.push(c);
            }
        }
        let mut i = 0;
        while chosen.len() < m && i < discarded.len() {
            chosen.push(discarded[i]);
            i += 1;
        }

        chosen.into_iter().map(|c| c.id).collect()
    }

    fn set_neighbors(&mut self, id: u32, layer: usize, neighbors: Vec<u32>) {
        self.nodes[id as usize].layers[layer] = neighbors;
    }

    fn push_neighbor(&mut self, id: u32, layer: usize, neighbor: u32) {
        self.nodes[id as usize].layers[layer].push(neighbor);
    }

    /// Inserts node `id` (which must equal the current node count) into the
    /// graph. `dist_nodes` gives the distance between any two node ids; the new
    /// node's vector must already be reachable through it.
    pub fn insert<F: Fn(u32, u32) -> f32>(&mut self, id: u32, dist_nodes: F) {
        debug_assert_eq!(id as usize, self.nodes.len());
        let level = self.random_level();
        self.nodes.push(NodeLinks {
            layers: vec![Vec::new(); level + 1],
        });

        let entry = match self.entry {
            None => {
                self.entry = Some(id);
                self.max_level = level;
                return;
            }
            Some(e) => e,
        };

        let dist_q = |x: u32| dist_nodes(id, x);
        let cur_max = self.max_level;

        // Descend from the top down to just above the new node's level,
        // greedily following the single nearest neighbour each layer.
        let mut ep = vec![entry];
        let mut layer = cur_max;
        while layer > level {
            let w = self.search_layer(&dist_q, &ep, 1, layer);
            ep = vec![w[0].id];
            layer -= 1;
        }

        // From the new node's level down to 0, find neighbours and link.
        let start = level.min(cur_max);
        for layer in (0..=start).rev() {
            let m_layer = if layer == 0 { self.m0 } else { self.m };
            let w = self.search_layer(&dist_q, &ep, self.ef_construction, layer);
            let selected = self.select_neighbors(w.clone(), m_layer, &dist_nodes);

            for &nbr in &selected {
                self.push_neighbor(id, layer, nbr);
                self.push_neighbor(nbr, layer, id);

                // Re-run the heuristic on the neighbour if it is now over-full.
                let nbr_links = self.neighbors(nbr, layer).to_vec();
                if nbr_links.len() > m_layer {
                    let cands: Vec<Candidate> = nbr_links
                        .iter()
                        .map(|&x| Candidate {
                            id: x,
                            dist: dist_nodes(nbr, x),
                        })
                        .collect();
                    let pruned = self.select_neighbors(cands, m_layer, &dist_nodes);
                    self.set_neighbors(nbr, layer, pruned);
                }
            }

            ep = w.into_iter().map(|c| c.id).collect();
        }

        if level > cur_max {
            self.max_level = level;
            self.entry = Some(id);
        }
    }

    /// Returns up to `max(ef, k)` nearest node ids to a query, ascending by
    /// distance, as `(id, distance)` pairs. `dist` gives the query-to-node
    /// distance. The caller takes the first `k` it still considers live.
    pub fn search<F: Fn(u32) -> f32>(&self, dist: F, k: usize, ef: usize) -> Vec<(u32, f32)> {
        if k == 0 {
            return Vec::new();
        }
        let entry = match self.entry {
            Some(e) => e,
            None => return Vec::new(),
        };

        let mut ep = vec![entry];
        let mut layer = self.max_level;
        while layer > 0 {
            let w = self.search_layer(&dist, &ep, 1, layer);
            ep = vec![w[0].id];
            layer -= 1;
        }

        let ef = ef.max(k);
        self.search_layer(&dist, &ep, ef, 0)
            .into_iter()
            .map(|c| (c.id, c.dist))
            .collect()
    }
}
