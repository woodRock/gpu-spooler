//! The ANSI codes the Python front ends print (always: `task gpus` output is meant for people).

pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";
pub const GREEN: &str = "\x1b[92m";
pub const YELLOW: &str = "\x1b[93m";
pub const RED: &str = "\x1b[91m";
pub const CYAN: &str = "\x1b[96m";
pub const DIM: &str = "\x1b[2m";

pub fn colorize(text: &str, color: &str) -> String {
    format!("{color}{text}{RESET}")
}
pub const ORANGE: &str = "\x1b[33m";

/// The colour `task list` and `task info` give a status.
pub fn status_color(status: &str) -> &'static str {
    match status {
        "queued" => DIM,
        "running" => GREEN,
        "done" => CYAN,
        "failed" => RED,
        "killed" | "cancelled" => YELLOW,
        "lost" => ORANGE,
        _ => "",
    }
}
