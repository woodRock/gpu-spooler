//! Formatting shared by the front ends (core.fmt_duration and friends).

use chrono::{DateTime, Local, TimeZone};

/// `3m 41s`, `2h 05m`; `-` for nothing or a negative time.
pub fn fmt_duration(seconds: Option<f64>) -> String {
    let Some(seconds) = seconds.filter(|s| *s >= 0.0) else { return "-".into() };
    let s = seconds as i64;
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m {:02}s", s / 60, s % 60)
    } else {
        format!("{}h {:02}m", s / 3600, (s % 3600) / 60)
    }
}

/// A timestamp as `10-25 13:46`, in local time.
pub fn fmt_ts(ts: Option<f64>) -> String {
    fmt_ts_in(ts, &Local)
}

/// [`fmt_ts`] in a given time zone.
pub fn fmt_ts_in<Tz: TimeZone>(ts: Option<f64>, tz: &Tz) -> String
where
    Tz::Offset: std::fmt::Display,
{
    let Some(ts) = ts else { return "-".into() };
    match DateTime::from_timestamp(ts.floor() as i64, 0) {
        Some(utc) => utc.with_timezone(tz).format("%m-%d %H:%M").to_string(),
        None => "-".into(),
    }
}

/// At most `n` characters, ending in `…` when cut.
pub fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

/// Collapse newlines and runs of whitespace, for table cells.
pub fn one_line(s: &str) -> String {
    crate::py::split(s).join(" ")
}
