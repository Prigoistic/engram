# engram

A low-level **associative vector memory engine** written in Rust, wire-compatible
with Redis. It speaks the RESP protocol — so `redis-cli`, `redis-benchmark`, and
existing Redis client libraries work unchanged — and adds named vector indices
with approximate nearest-neighbour search on top of the classic key-value store.

It is built from the ground up with no external crates beyond `libc`: the
distance kernels, the [HNSW](https://arxiv.org/abs/1603.09320) graph index, the
persistence layer, and the concurrency are all hand-rolled.

📖 **[Read the project write-up →](https://prigoistic.github.io/engram/)** — architecture, diagrams, and benchmarks.

## What's inside

- **Single-threaded, event-driven core** over raw `kqueue` (macOS) / `epoll` (Linux).
- **SIMD distance kernels** — hand-written NEON (aarch64) and AVX2+FMA (x86_64),
  with a scalar fallback and runtime dispatch (~9× over scalar at 768-dim).
- **HNSW** approximate nearest-neighbour index, written from scratch
  (search-layer, neighbour-selection heuristic, layered graph).
- **Durability** — a CRC-framed, `fsync`'d write-ahead log for crash recovery,
  plus a compacting snapshot that is loaded through `mmap`.
- **Non-blocking search** — `VSEARCH` runs on a worker-thread pool (the registry
  is behind an `RwLock`), so a slow search never stalls other clients. The event
  loop is woken by a self-pipe.

## Running

```sh
cargo run                      # listens on 127.0.0.1:6380, in memory
cargo run -- --dir ./data      # enable persistence (WAL + snapshot)
cargo run -- --port 6390       # override settings with flags
cargo run -- engram.conf       # or a key/value config file
```

Settings are `bind`, `port`, `maxclients`, and `dir`; `bind`/`port`/`maxclients`
are readable and writable at runtime with `CONFIG GET` / `CONFIG SET`.

## Commands

Key-value: `PING`, `ECHO`, `SET`, `GET`, `DEL`, `EXISTS`, `APPEND`, `INCR`, `CONFIG`.

Vector memory:

| Command | Description |
|---|---|
| `VNEW index dim [METRIC cosine\|l2\|dot]` | Create a vector index (default metric `cosine`). |
| `VADD index key vector` | Insert/overwrite `key`; `vector` is packed little-endian `f32` bytes. Replies `1` if new, `0` if overwritten. |
| `VSEARCH index query k [EF n]` | The `k` nearest keys to `query`, closest first, as a flat `key, distance` array. `EF` tunes recall vs. speed. |
| `VDEL index key` | Remove `key`. |
| `VINFO index` | Report `dim`, `metric`, and live `count`. |
| `VSAVE` | Write a snapshot and truncate the WAL (requires `--dir`). |

Vectors travel as raw little-endian `f32` bulk strings, which is what real vector
clients send. Distances are returned per metric (`l2`: Euclidean, `cosine`:
`1 - cos`, `dot`: negative inner product) — smaller is always closer.

## Development

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (stable) via `rustup`, with the `rustfmt` and `clippy` components:
  ```sh
  rustup component add rustfmt clippy
  ```
- [pre-commit](https://pre-commit.com/) for the git hooks.

### Tests

```sh
cargo test                                            # unit + integration
cargo test --release kernels -- --ignored --nocapture # SIMD throughput bench
```

### Commit messages

This project follows [Conventional Commits](https://www.conventionalcommits.org/), enforced by a `commit-msg` hook.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
