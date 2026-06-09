use super::{Arity, Command};
use crate::resp::Value;
use crate::state::State;

/// `VSAVE` writes a compacting snapshot of every index and truncates the
/// write-ahead log. Requires persistence to be enabled (a configured `dir`).
pub const COMMAND: Command = Command {
    name: "VSAVE",
    arity: Arity::Exact(1),
    handler: vsave,
};

fn vsave(_args: &[Vec<u8>], state: &mut State) -> Value {
    // Split the borrow: snapshot reads the registry while the log is rewritten.
    let State {
        vectors, persist, ..
    } = state;

    match persist {
        Some(p) => {
            let registry = vectors.read().unwrap_or_else(|e| e.into_inner());
            match p.save(&registry) {
                Ok(()) => Value::Simple("OK".to_string()),
                Err(e) => Value::Error(format!("ERR save failed: {e}")),
            }
        }
        None => Value::Error("ERR persistence not enabled; start with a data dir".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use crate::command::{
        dispatch,
        test_utils::{cmd, state},
    };
    use crate::resp::Value;

    #[test]
    fn errors_without_persistence() {
        let mut state = state();
        assert_eq!(
            dispatch(&cmd(&["VSAVE"]), &mut state),
            Value::Error("ERR persistence not enabled; start with a data dir".to_string())
        );
    }
}
