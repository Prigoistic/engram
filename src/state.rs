//! The server's shared state.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::config::Config;
use crate::persist::Persist;
use crate::vector::VectorRegistry;

/// The key-value store.
pub type Store = HashMap<Vec<u8>, Vec<u8>>;

/// The named vector indices, shared with the search worker pool. Searches take
/// a read lock on a worker thread; mutations take a write lock on the event
/// loop, so the two never corrupt each other.
pub type SharedVectors = Arc<RwLock<VectorRegistry>>;

/// The shared state commands read and modify.
pub struct State {
    /// The key-value store.
    pub store: Store,

    /// The named vector indices.
    pub vectors: SharedVectors,

    /// The persistence handle, present when a data directory is configured.
    pub persist: Option<Persist>,

    /// The server configuration.
    pub config: Config,
}

impl State {
    /// Creates state with the given configuration.
    ///
    /// When `config.dir` is set, the vector registry is recovered from the data
    /// directory (snapshot + write-ahead log) and the persistence handle is
    /// kept for logging further mutations. A recovery failure is reported and
    /// the server falls back to an empty in-memory registry.
    pub fn new(config: Config) -> Self {
        let (vectors, persist) = match &config.dir {
            Some(dir) => match Persist::open(dir) {
                Ok((persist, vectors)) => (vectors, Some(persist)),
                Err(e) => {
                    eprintln!("persistence init failed for '{dir}': {e}; running in memory");
                    (VectorRegistry::new(), None)
                }
            },
            None => (VectorRegistry::new(), None),
        };

        Self {
            store: Store::new(),
            vectors: Arc::new(RwLock::new(vectors)),
            persist,
            config,
        }
    }
}
