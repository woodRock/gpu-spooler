//! Shell quoting, exactly as Python's `shlex.quote` / `shlex.join` produce it: commands are stored
//! in the queue and shown in `task list`, and remote scripts embed quoted paths.

/// `shlex.quote`.
pub fn quote(s: &str) -> String {
    if s.is_empty() {
        return "''".into();
    }
    let safe = |b: u8| b.is_ascii_alphanumeric() || b"_@%+=:,./-".contains(&b);
    if s.bytes().all(safe) {
        return s.into();
    }
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

/// `shlex.join`.
pub fn join<S: AsRef<str>>(words: &[S]) -> String {
    words.iter().map(|w| quote(w.as_ref())).collect::<Vec<_>>().join(" ")
}

/// Quote a remote path for bash, keeping a leading `~/` expandable (repo.sh_path).
pub fn sh_path(path: &str) -> String {
    if path == "~" {
        return "\"$HOME\"".into();
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return format!("\"$HOME\"/{}", quote(rest));
    }
    quote(path)
}

/// How `task submit` turns its arguments into the stored command: one argument is shell code
/// (`task submit 'a && b'`), several are quoted so they reach the command unchanged.
pub fn command_from_argv<S: AsRef<str>>(argv: &[S]) -> String {
    match argv {
        [one] => one.as_ref().to_string(),
        _ => join(argv),
    }
}

/// `shlex.split` (POSIX mode, no comments). Err for an unclosed quote or a trailing backslash.
pub fn split(s: &str) -> Result<Vec<String>, &'static str> {
    let mut words = vec![];
    let mut word = String::new();
    let mut in_word = false;
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        match c {
            ' ' | '\t' | '\r' | '\n' => {
                if in_word {
                    words.push(std::mem::take(&mut word));
                    in_word = false;
                }
            }
            '\\' => {
                in_word = true;
                word.push(chars.next().ok_or("No escaped character")?);
            }
            '\'' => {
                in_word = true;
                loop {
                    match chars.next().ok_or("No closing quotation")? {
                        '\'' => break,
                        c => word.push(c),
                    }
                }
            }
            '"' => {
                in_word = true;
                loop {
                    match chars.next().ok_or("No closing quotation")? {
                        '"' => break,
                        '\\' => match chars.next().ok_or("No closing quotation")? {
                            c @ ('"' | '\\' | '$' | '`' | '\n') => word.push(c),
                            c => {
                                word.push('\\');
                                word.push(c);
                            }
                        },
                        c => word.push(c),
                    }
                }
            }
            c => {
                in_word = true;
                word.push(c);
            }
        }
    }
    if in_word {
        words.push(word);
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_like_shlex() {
        assert_eq!(split("a 'b c' \"d \\\" e\" f\\ g").unwrap(), ["a", "b c", "d \" e", "f g"]);
        assert_eq!(split("x='' y").unwrap(), ["x=", "y"]);
        assert_eq!(split("\"a\\b\"").unwrap(), ["a\\b"]);
        assert!(split("python3 'unterminated").is_err());
        assert!(split("trailing\\").is_err());
        assert_eq!(split("  ").unwrap(), Vec::<String>::new());
    }
}
