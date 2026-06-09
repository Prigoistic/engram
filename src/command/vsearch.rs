use super::{Arity, Command};
use crate::resp::Value;
use crate::state::State;
use crate::vector::{self, VectorRegistry};

/// Default search breadth when the client does not pass `EF`.
const DEFAULT_EF: usize = 64;

/// `VSEARCH index query k [EF n]` returns the `k` nearest keys to `query`.
///
/// `query` is packed little-endian `f32` bytes. The reply is a flat array of
/// `key, distance` pairs, closest first, with distances as strings. `EF` tunes
/// the graph search breadth; larger values trade speed for recall.
///
/// The server runs this command on a worker thread, so its body is the free
/// function [`run`] taking a read-locked registry, which both the handler (used
/// in tests) and the worker pool call.
pub const COMMAND: Command = Command {
    name: "VSEARCH",
    arity: Arity::Min(4),
    handler: vsearch,
};

fn vsearch(args: &[Vec<u8>], state: &mut State) -> Value {
    let vectors = state.vectors.read().unwrap_or_else(|e| e.into_inner());
    run(args, &vectors)
}

/// Executes a search against `vectors`. `args` is the command's arguments after
/// the name. Self-contained (it re-checks arity) because the offload path
/// bypasses the dispatcher's arity check.
pub(crate) fn run(args: &[Vec<u8>], vectors: &VectorRegistry) -> Value {
    let [name, raw, k_raw] = match args {
        [name, raw, k_raw, ..] => [name, raw, k_raw],
        _ => return super::wrong_args("vsearch"),
    };

    let k: usize = match std::str::from_utf8(k_raw).ok().and_then(|s| s.parse().ok()) {
        Some(k) => k,
        None => return super::not_integer(),
    };

    let ef = match &args[3..] {
        [] => DEFAULT_EF,
        [keyword, value] if keyword.eq_ignore_ascii_case(b"EF") => {
            match std::str::from_utf8(value).ok().and_then(|s| s.parse().ok()) {
                Some(ef) => ef,
                None => return super::not_integer(),
            }
        }
        _ => return Value::Error("ERR syntax error".to_string()),
    };

    let index = match vectors.get(name) {
        Some(index) => index,
        None => return Value::Error("ERR no such index".to_string()),
    };

    let query = match vector::decode(raw) {
        Some(query) => query,
        None => return Value::Error("ERR invalid vector encoding".to_string()),
    };

    if query.len() != index.dim() {
        return Value::Error(format!(
            "ERR wrong vector dimension: index expects {}, got {}",
            index.dim(),
            query.len()
        ));
    }

    let neighbors = index.search(query, k, ef);
    let mut out = Vec::with_capacity(neighbors.len() * 2);
    for n in neighbors {
        out.push(Value::Bulk(n.key));
        out.push(Value::Bulk(n.distance.to_string().into_bytes()));
    }
    Value::Array(out)
}

#[cfg(test)]
mod tests {
    use crate::command::{
        dispatch,
        test_utils::{cmd, state},
    };
    use crate::resp::Value;
    use crate::state::State;
    use crate::vector;

    fn add(state: &mut State, index: &str, key: &str, v: &[f32]) {
        let args = vec![
            b"VADD".to_vec(),
            index.as_bytes().to_vec(),
            key.as_bytes().to_vec(),
            vector::encode(v),
        ];
        dispatch(&args, state);
    }

    fn search(state: &mut State, index: &str, q: &[f32], k: usize) -> Vec<Vec<u8>> {
        let args = vec![
            b"VSEARCH".to_vec(),
            index.as_bytes().to_vec(),
            vector::encode(q),
            k.to_string().into_bytes(),
        ];
        match dispatch(&args, state) {
            Value::Array(items) => items
                .into_iter()
                .map(|v| match v {
                    Value::Bulk(b) => b,
                    other => panic!("expected bulk, got {other:?}"),
                })
                .collect(),
            other => panic!("expected array, got {other:?}"),
        }
    }

    fn setup() -> State {
        let mut state = state();
        dispatch(&cmd(&["VNEW", "mem", "2", "METRIC", "l2"]), &mut state);
        add(&mut state, "mem", "origin", &[0.0, 0.0]);
        add(&mut state, "mem", "near", &[1.0, 0.0]);
        add(&mut state, "mem", "far", &[9.0, 0.0]);
        state
    }

    #[test]
    fn returns_keys_closest_first() {
        let mut state = setup();
        let flat = search(&mut state, "mem", &[0.0, 0.0], 3);
        // Flat [key, score, key, score, ...].
        assert_eq!(flat.len(), 6);
        assert_eq!(flat[0], b"origin");
        assert_eq!(flat[2], b"near");
        assert_eq!(flat[4], b"far");
    }

    #[test]
    fn respects_k() {
        let mut state = setup();
        let flat = search(&mut state, "mem", &[0.0, 0.0], 1);
        assert_eq!(flat.len(), 2);
        assert_eq!(flat[0], b"origin");
    }

    #[test]
    fn distance_is_parseable_and_ordered() {
        let mut state = setup();
        let flat = search(&mut state, "mem", &[0.0, 0.0], 3);
        let d0: f32 = std::str::from_utf8(&flat[1]).unwrap().parse().unwrap();
        let d1: f32 = std::str::from_utf8(&flat[3]).unwrap().parse().unwrap();
        let d2: f32 = std::str::from_utf8(&flat[5]).unwrap().parse().unwrap();
        assert!(d0 <= d1 && d1 <= d2);
        assert!((d0 - 0.0).abs() < 1e-5);
        assert!((d1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn accepts_ef_option() {
        let mut state = setup();
        let args = vec![
            b"VSEARCH".to_vec(),
            b"mem".to_vec(),
            vector::encode(&[0.0, 0.0]),
            b"2".to_vec(),
            b"EF".to_vec(),
            b"128".to_vec(),
        ];
        match dispatch(&args, &mut state) {
            Value::Array(items) => assert_eq!(items.len(), 4),
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn rejects_missing_index() {
        let mut state = state();
        let args = vec![
            b"VSEARCH".to_vec(),
            b"nope".to_vec(),
            vector::encode(&[0.0, 0.0]),
            b"1".to_vec(),
        ];
        assert_eq!(
            dispatch(&args, &mut state),
            Value::Error("ERR no such index".to_string())
        );
    }

    #[test]
    fn rejects_wrong_dimension() {
        let mut state = setup();
        let args = vec![
            b"VSEARCH".to_vec(),
            b"mem".to_vec(),
            vector::encode(&[0.0, 0.0, 0.0]),
            b"1".to_vec(),
        ];
        match dispatch(&args, &mut state) {
            Value::Error(e) => assert!(e.contains("wrong vector dimension"), "{e}"),
            other => panic!("expected error, got {other:?}"),
        }
    }
}
