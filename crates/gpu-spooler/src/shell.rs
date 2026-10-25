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
