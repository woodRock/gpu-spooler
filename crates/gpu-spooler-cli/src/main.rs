//! `task` - the gpu-spooler command line (Rust port, in progress).
//!
//! Ported so far: `task gpus`. Until the rest is, the Python version is the one installed.

mod color;
mod gpu_status;

use clap::{CommandFactory, Parser, Subcommand};
use std::process::ExitCode;

const ABOUT: &str = "\
task - GPU task spooler for SSH-reachable GPU servers (VUW ECS by default)

The Rust port of gpu-spooler, in progress. Ported so far:

  task gpus [-d] [-f] [-H SPEC]           GPU availability

Everything else is still the Python version (install.sh installs it).";

/// Shown at the end of `task COMMAND -h`.
fn examples(name: &str, rows: &[(&str, &str)]) -> String {
    let cmds: Vec<String> = rows.iter().map(|(a, _)| format!("task {name} {a}").trim_end().to_string()).collect();
    let width = cmds.iter().map(|c| c.chars().count()).max().unwrap_or(0);
    let lines: Vec<String> = cmds.iter().zip(rows).map(|(c, (_, what))| format!("  {c:<width$}  # {what}")).collect();
    format!("Examples:\n{}", lines.join("\n"))
}

#[derive(Parser)]
#[command(name = "task", version, about = ABOUT, infer_long_args = true, disable_help_subcommand = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Show GPU availability across servers
    #[command(after_help = examples("gpus", &[
        ("", "one row per server"),
        ("-f", "only servers with a free GPU"),
        ("-d -H cuda14", "one row per GPU on cuda14"),
    ]))]
    Gpus(gpu_status::Args),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let code = match cli.command {
        None => {
            let _ = Cli::command().print_help();
            0
        }
        Some(Cmd::Gpus(args)) => gpu_status::run(&args),
    };
    ExitCode::from(code.clamp(0, 255) as u8)
}
