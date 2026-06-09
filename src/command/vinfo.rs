use super::{Arity, Command};
use crate::resp::Value;
use crate::state::State;

/// `VINFO index` describes an index as a flat array of `field, value` pairs.
pub const COMMAND: Command = Command {
    name: "VINFO",
    arity: Arity::Exact(2),
    handler: vinfo,
};

fn vinfo(args: &[Vec<u8>], state: &mut State) -> Value {
    let name = &args[0];

    let vectors = state.vectors.read().unwrap_or_else(|e| e.into_inner());
    let index = match vectors.get(name) {
        Some(index) => index,
        None => return Value::Error("ERR no such index".to_string()),
    };

    let pair = |k: &str, v: String| [Value::Bulk(k.into()), Value::Bulk(v.into_bytes())];
    let fields = [
        pair("dim", index.dim().to_string()),
        pair("metric", index.metric().as_str().to_string()),
        pair("count", index.len().to_string()),
    ];

    Value::Array(fields.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use crate::command::{
        dispatch,
        test_utils::{cmd, state},
    };
    use crate::resp::Value;

    fn field(reply: &Value, name: &str) -> String {
        let Value::Array(items) = reply else {
            panic!("expected array");
        };
        let mut it = items.iter();
        while let (Some(k), Some(v)) = (it.next(), it.next()) {
            if let (Value::Bulk(k), Value::Bulk(v)) = (k, v)
                && k == name.as_bytes()
            {
                return String::from_utf8(v.clone()).unwrap();
            }
        }
        panic!("field {name} not found");
    }

    #[test]
    fn reports_dim_metric_count() {
        let mut state = state();
        dispatch(&cmd(&["VNEW", "mem", "16", "METRIC", "dot"]), &mut state);
        let reply = dispatch(&cmd(&["VINFO", "mem"]), &mut state);
        assert_eq!(field(&reply, "dim"), "16");
        assert_eq!(field(&reply, "metric"), "dot");
        assert_eq!(field(&reply, "count"), "0");
    }

    #[test]
    fn rejects_missing_index() {
        let mut state = state();
        assert_eq!(
            dispatch(&cmd(&["VINFO", "nope"]), &mut state),
            Value::Error("ERR no such index".to_string())
        );
    }
}
