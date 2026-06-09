use super::{Arity, Command};
use crate::resp::Value;
use crate::state::State;
use crate::vector;

/// `VADD index key vector` stores `vector` under `key` in `index`.
///
/// `vector` is packed little-endian `f32` bytes. Replies `1` if `key` is new,
/// `0` if it overwrote an existing key.
pub const COMMAND: Command = Command {
    name: "VADD",
    arity: Arity::Exact(4),
    handler: vadd,
};

fn vadd(args: &[Vec<u8>], state: &mut State) -> Value {
    let [name, key, raw] = [&args[0], &args[1], &args[2]];

    // Apply to the index in a scope so the registry borrow ends before the
    // disjoint persistence borrow below.
    let is_new = {
        let mut vectors = state.vectors.write().unwrap_or_else(|e| e.into_inner());
        let index = match vectors.get_mut(name) {
            Some(index) => index,
            None => return Value::Error("ERR no such index".to_string()),
        };

        let vec = match vector::decode(raw) {
            Some(vec) => vec,
            None => return Value::Error("ERR invalid vector encoding".to_string()),
        };

        if vec.len() != index.dim() {
            return Value::Error(format!(
                "ERR wrong vector dimension: index expects {}, got {}",
                index.dim(),
                vec.len()
            ));
        }

        index.add(key.clone(), vec)
    };

    // Log the raw payload as received, so replay reproduces the same state.
    if let Some(p) = state.persist.as_mut()
        && let Err(e) = p.log_add(name, key, raw)
    {
        eprintln!("wal append (VADD) failed: {e}");
    }

    Value::Integer(if is_new { 1 } else { 0 })
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

    fn add(state: &mut State, index: &str, key: &str, v: &[f32]) -> Value {
        let args = vec![
            b"VADD".to_vec(),
            index.as_bytes().to_vec(),
            key.as_bytes().to_vec(),
            vector::encode(v),
        ];
        dispatch(&args, state)
    }

    #[test]
    fn add_then_overwrite_reports_new_flag() {
        let mut state = state();
        dispatch(&cmd(&["VNEW", "mem", "2", "METRIC", "l2"]), &mut state);
        assert_eq!(add(&mut state, "mem", "a", &[1.0, 0.0]), Value::Integer(1));
        assert_eq!(add(&mut state, "mem", "a", &[0.0, 1.0]), Value::Integer(0));
        assert_eq!(state.vectors.read().unwrap().get(b"mem").unwrap().len(), 1);
    }

    #[test]
    fn rejects_missing_index() {
        let mut state = state();
        assert_eq!(
            add(&mut state, "nope", "a", &[1.0, 0.0]),
            Value::Error("ERR no such index".to_string())
        );
    }

    #[test]
    fn rejects_wrong_dimension() {
        let mut state = state();
        dispatch(&cmd(&["VNEW", "mem", "2"]), &mut state);
        match add(&mut state, "mem", "a", &[1.0, 2.0, 3.0]) {
            Value::Error(e) => assert!(e.contains("wrong vector dimension"), "{e}"),
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_invalid_encoding() {
        let mut state = state();
        dispatch(&cmd(&["VNEW", "mem", "2"]), &mut state);
        let args = vec![
            b"VADD".to_vec(),
            b"mem".to_vec(),
            b"a".to_vec(),
            vec![1, 2, 3], // not a multiple of four
        ];
        assert_eq!(
            dispatch(&args, &mut state),
            Value::Error("ERR invalid vector encoding".to_string())
        );
    }
}
