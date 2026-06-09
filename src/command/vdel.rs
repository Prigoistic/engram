use super::{Arity, Command};
use crate::resp::Value;
use crate::state::State;

/// `VDEL index key` removes `key` from `index`.
///
/// Replies `1` if the key was present, `0` otherwise.
pub const COMMAND: Command = Command {
    name: "VDEL",
    arity: Arity::Exact(3),
    handler: vdel,
};

fn vdel(args: &[Vec<u8>], state: &mut State) -> Value {
    let [name, key] = [&args[0], &args[1]];

    let removed = {
        let mut vectors = state.vectors.write().unwrap_or_else(|e| e.into_inner());
        match vectors.get_mut(name) {
            Some(index) => index.remove(key),
            None => return Value::Error("ERR no such index".to_string()),
        }
    };

    if removed
        && let Some(p) = state.persist.as_mut()
        && let Err(e) = p.log_del(name, key)
    {
        eprintln!("wal append (VDEL) failed: {e}");
    }

    Value::Integer(if removed { 1 } else { 0 })
}

#[cfg(test)]
mod tests {
    use crate::command::{
        dispatch,
        test_utils::{cmd, state},
    };
    use crate::resp::Value;
    use crate::vector;

    #[test]
    fn deletes_present_key() {
        let mut state = state();
        dispatch(&cmd(&["VNEW", "mem", "2"]), &mut state);
        let args = vec![
            b"VADD".to_vec(),
            b"mem".to_vec(),
            b"a".to_vec(),
            vector::encode(&[1.0, 0.0]),
        ];
        dispatch(&args, &mut state);

        assert_eq!(
            dispatch(&cmd(&["VDEL", "mem", "a"]), &mut state),
            Value::Integer(1)
        );
        assert_eq!(
            dispatch(&cmd(&["VDEL", "mem", "a"]), &mut state),
            Value::Integer(0)
        );
    }

    #[test]
    fn rejects_missing_index() {
        let mut state = state();
        assert_eq!(
            dispatch(&cmd(&["VDEL", "nope", "a"]), &mut state),
            Value::Error("ERR no such index".to_string())
        );
    }
}
