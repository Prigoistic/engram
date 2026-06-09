use super::{Arity, Command};
use crate::resp::Value;
use crate::state::State;
use crate::vector::Metric;

/// `VNEW index dim [METRIC cosine|l2|dot]` creates a vector index.
///
/// The metric defaults to `cosine`. It is an error if `index` already exists.
pub const COMMAND: Command = Command {
    name: "VNEW",
    arity: Arity::Min(3),
    handler: vnew,
};

fn vnew(args: &[Vec<u8>], state: &mut State) -> Value {
    let name = &args[0];

    let dim: usize = match std::str::from_utf8(&args[1])
        .ok()
        .and_then(|s| s.parse().ok())
    {
        Some(d) if d > 0 => d,
        _ => return Value::Error("ERR dimension must be a positive integer".to_string()),
    };

    let metric = match &args[2..] {
        [] => Metric::Cosine,
        [keyword, value] if keyword.eq_ignore_ascii_case(b"METRIC") => {
            match std::str::from_utf8(value).ok().and_then(Metric::parse) {
                Some(m) => m,
                None => return Value::Error("ERR unknown metric".to_string()),
            }
        }
        _ => return Value::Error("ERR syntax error".to_string()),
    };

    let created = state
        .vectors
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .create(name.clone(), dim, metric);

    if created {
        if let Some(p) = state.persist.as_mut()
            && let Err(e) = p.log_new(name, dim, metric)
        {
            eprintln!("wal append (VNEW) failed: {e}");
        }
        Value::Simple("OK".to_string())
    } else {
        Value::Error("ERR index already exists".to_string())
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
    fn creates_index_with_default_metric() {
        let mut state = state();
        assert_eq!(
            dispatch(&cmd(&["VNEW", "mem", "8"]), &mut state),
            Value::Simple("OK".to_string())
        );
        assert_eq!(state.vectors.read().unwrap().get(b"mem").unwrap().dim(), 8);
    }

    #[test]
    fn creates_index_with_explicit_metric() {
        let mut state = state();
        assert_eq!(
            dispatch(&cmd(&["VNEW", "mem", "8", "METRIC", "l2"]), &mut state),
            Value::Simple("OK".to_string())
        );
    }

    #[test]
    fn rejects_duplicate() {
        let mut state = state();
        dispatch(&cmd(&["VNEW", "mem", "8"]), &mut state);
        assert_eq!(
            dispatch(&cmd(&["VNEW", "mem", "8"]), &mut state),
            Value::Error("ERR index already exists".to_string())
        );
    }

    #[test]
    fn rejects_bad_dimension() {
        let mut state = state();
        assert_eq!(
            dispatch(&cmd(&["VNEW", "mem", "0"]), &mut state),
            Value::Error("ERR dimension must be a positive integer".to_string())
        );
        assert_eq!(
            dispatch(&cmd(&["VNEW", "mem", "x"]), &mut state),
            Value::Error("ERR dimension must be a positive integer".to_string())
        );
    }

    #[test]
    fn rejects_unknown_metric() {
        let mut state = state();
        assert_eq!(
            dispatch(&cmd(&["VNEW", "mem", "8", "METRIC", "nope"]), &mut state),
            Value::Error("ERR unknown metric".to_string())
        );
    }

    #[test]
    fn rejects_syntax_error() {
        let mut state = state();
        assert_eq!(
            dispatch(&cmd(&["VNEW", "mem", "8", "GARBAGE", "l2"]), &mut state),
            Value::Error("ERR syntax error".to_string())
        );
    }
}
