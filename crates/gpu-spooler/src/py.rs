//! Python behaviour the port has to reproduce exactly: how strings split into lines and words,
//! what `int()` accepts, and how values print in error messages (`repr`, `str`).

use toml::Value;

/// `str.splitlines()`: \n, \r\n, \r and the other line boundaries Python knows; no trailing ''.
pub fn splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let boundary =
            matches!(c, '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}');
        if boundary {
            out.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' {
                if let Some(&(j, '\n')) = chars.peek() {
                    chars.next();
                    end = j + 1;
                }
            }
            start = end;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// `str.split()` with no argument.
pub fn split(s: &str) -> Vec<&str> {
    s.split(is_space).filter(|w| !w.is_empty()).collect()
}

/// `str.strip()` with no argument.
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

fn is_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\x1c'..='\x1f')
}

/// `str.isdigit()`, for the ASCII digits nvidia-smi and ps print.
pub fn isdigit(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// `int(s)` for a decimal string: surrounding whitespace, a sign and `_` between digits allowed.
pub fn int(s: &str) -> Option<i64> {
    let t = strip(s);
    let (neg, digits) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if digits.is_empty() || digits.starts_with('_') || digits.ends_with('_') || digits.contains("__") {
        return None;
    }
    let clean: String = digits.chars().filter(|&c| c != '_').collect();
    if !clean.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: i64 = clean.parse().ok()?;
    Some(if neg { -n } else { n })
}

/// `repr(str)`.
pub fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `repr(list_of_str)`, e.g. `['a', 'b']`.
pub fn repr_list(items: &[String]) -> String {
    let inner: Vec<String> = items.iter().map(|s| repr_str(s)).collect();
    format!("[{}]", inner.join(", "))
}

/// `repr(float)`: the shortest digits that round-trip, positional from 1e-4 up to 1e16 and
/// with an exponent outside that (`1e-05`, `1e+16`), always with a `.` or an exponent.
pub fn repr_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf" } else { "-inf" }.into();
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // Rust's {:e} is the shortest round-trip form too: "1.5e-5", "1.7614000005e9".
    let sci = format!("{f:e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} always has an exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let neg = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if neg { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let point = exp + 1; // digits before the decimal point
        let body = if point <= 0 {
            format!("0.{}{digits}", "0".repeat((-point) as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
        };
        format!("{sign}{body}")
    } else {
        let mant = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits };
        format!("{sign}{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

/// `json.dumps(value)` with Python's defaults: `", "` and `": "` separators, keys in insertion
/// order, non-ASCII as `\uXXXX`, floats as `repr`. Files both versions write (the GPU snapshot,
/// the network cache, taura's env file) then come out the same.
pub fn json_dumps(v: &serde_json::Value) -> String {
    use serde_json::Value as J;
    fn string(s: &str, out: &mut String) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                    let mut buf = [0u16; 2];
                    for unit in c.encode_utf16(&mut buf) {
                        out.push_str(&format!("\\u{unit:04x}"));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
    fn value(v: &J, out: &mut String) {
        match v {
            J::Null => out.push_str("null"),
            J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            J::Number(n) => match (n.as_i64(), n.as_u64(), n.as_f64()) {
                (Some(i), _, _) => out.push_str(&i.to_string()),
                (_, Some(u), _) => out.push_str(&u.to_string()),
                (_, _, Some(f)) if f.is_finite() => out.push_str(&repr_float(f)),
                (_, _, Some(f)) => out.push_str(if f.is_nan() {
                    "NaN"
                } else if f > 0.0 {
                    "Infinity"
                } else {
                    "-Infinity"
                }),
                _ => out.push_str(&n.to_string()),
            },
            J::String(s) => string(s, out),
            J::Array(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    value(x, out);
                }
                out.push(']');
            }
            J::Object(o) => {
                out.push('{');
                for (i, (k, x)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    string(k, out);
                    out.push_str(": ");
                    value(x, out);
                }
                out.push('}');
            }
        }
    }
    let mut out = String::new();
    value(v, &mut out);
    out
}

/// `shutil.which(name)`: the first executable called `name` on $PATH.
pub fn which(name: &str) -> Option<std::path::PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join(name))
        .find(|p| std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0))
}

/// `repr()` of a value read from TOML (tomllib gives Python bools, ints, floats, strs, lists, dicts).
pub fn repr_value(v: &Value) -> String {
    match v {
        Value::String(s) => repr_str(s),
        Value::Integer(i) => i.to_string(),
        Value::Float(f) => repr_float(*f),
        Value::Boolean(b) => if *b { "True" } else { "False" }.into(),
        Value::Datetime(d) => d.to_string(),
        Value::Array(a) => format!("[{}]", a.iter().map(repr_value).collect::<Vec<_>>().join(", ")),
        Value::Table(t) => format!(
            "{{{}}}",
            t.iter().map(|(k, v)| format!("{}: {}", repr_str(k), repr_value(v))).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// `str()` of a TOML value.
pub fn str_value(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => repr_value(other),
    }
}

/// Python truthiness (`bool(value)`).
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::String(s) => !s.is_empty(),
        Value::Integer(i) => *i != 0,
        Value::Float(f) => *f != 0.0,
        Value::Boolean(b) => *b,
        Value::Datetime(_) => true,
        Value::Array(a) => !a.is_empty(),
        Value::Table(t) => !t.is_empty(),
    }
}

/// `s[:n]` by characters.
pub fn head(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// `format(x, "g")`: six significant digits, trailing zeros dropped, exponent form below 1e-4 and
/// from 1e6 (`2.5e+10`, `1e-05`).
pub fn format_g(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    if !x.is_finite() {
        return if x.is_nan() {
            "nan".into()
        } else if x > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    // Round to 6 significant digits first: the exponent is the rounded value's.
    let sci = format!("{x:.5e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} always has an exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let trim = |s: String| {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    };
    if !(-4..6).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mantissa.to_string()), exp.abs())
    } else {
        let decimals = (5 - exp).max(0) as usize;
        trim(format!("{x:.decimals$}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitlines_like_python() {
        assert_eq!(splitlines("a\nb\r\nc\rd\n"), ["a", "b", "c", "d"]);
        assert_eq!(splitlines(""), Vec::<&str>::new());
        assert_eq!(splitlines("\n\n"), ["", ""]);
        assert_eq!(splitlines("x"), ["x"]);
    }

    #[test]
    fn int_like_python() {
        assert_eq!(int(" 12 "), Some(12));
        assert_eq!(int("-5"), Some(-5));
        assert_eq!(int("+5"), Some(5));
        assert_eq!(int("1_000"), Some(1000));
        assert_eq!(int("--5"), None);
        assert_eq!(int("[N/A]"), None);
        assert_eq!(int(""), None);
        assert_eq!(int("_1"), None);
    }

    #[test]
    fn reprs() {
        assert_eq!(repr_str("a"), "'a'");
        assert_eq!(repr_str("it's"), "\"it's\"");
        assert_eq!(repr_str("a'b\"c"), "'a\\'b\"c'");
        assert_eq!(repr_list(&["bar".into(), "foo".into()]), "['bar', 'foo']");
        assert_eq!(repr_float(2.5), "2.5");
        assert_eq!(repr_float(48.0), "48.0");
        assert_eq!(repr_float(0.1), "0.1");
        assert_eq!(repr_float(1761400000.5), "1761400000.5");
        assert_eq!(repr_float(1e-5), "1e-05");
        assert_eq!(repr_float(1.5e-5), "1.5e-05");
        assert_eq!(repr_float(1e16), "1e+16");
        assert_eq!(repr_float(1234.0), "1234.0");
        assert_eq!(repr_float(0.0001), "0.0001");
        assert_eq!(repr_float(-2.5), "-2.5");
        let v: serde_json::Value = serde_json::from_str(r#"{"b": 1, "a": [1.5, "naïve\n", null, true]}"#).unwrap();
        assert_eq!(json_dumps(&v), r#"{"b": 1, "a": [1.5, "na\u00efve\n", null, true]}"#);
        assert_eq!(repr_value(&Value::Boolean(true)), "True");
    }
}
