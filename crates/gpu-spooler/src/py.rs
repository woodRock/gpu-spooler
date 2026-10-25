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

/// `repr(float)` for the values a config holds: always with a decimal point (exponent forms,
/// which Python uses below 1e-4 and from 1e16, are not reproduced).
pub fn repr_float(f: f64) -> String {
    if f.is_finite() && f == f.trunc() {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
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
        assert_eq!(repr_value(&Value::Boolean(true)), "True");
    }
}
