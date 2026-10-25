//! `task` - the gpu-spooler command line (Rust port, in progress).
//!
//! Ported so far: the commands that only read (the queue, logs, hosts, GPUs) plus `config init`,
//! `ssh`, `connect` and `exec`. Until the rest is, the Python version is the one installed.

mod cluster;
mod color;
mod ctx;
mod gpu_status;
mod tasks;

use clap::{CommandFactory, Parser, Subcommand};
use ctx::{CmdResult, Ctx};
use std::process::ExitCode;

const ABOUT: &str = "\
task - GPU task spooler for SSH-reachable GPU servers (VUW ECS by default)

The Rust port of gpu-spooler, in progress. Ported so far:

Tasks:
  task list [-a]                          list tasks (the 200 most recent; -a: all)
  task info ID                            details of a task
  task logs [-f] ID                       print a task's output (-f: follow it)
  task wait ID... [--timeout S]           wait; the exit code says how they ended

Cluster:
  task gpus [-d] [-f]                     GPU availability
  task hosts                              configured hosts and network mode
  task exec [-H SPEC] -- CMD              run CMD on every (selected) server
  task ssh HOST                           interactive shell on a host
  task connect                            log in to jump hosts once (password/OTP)
  task daemon status                      is the background daemon running?
  task config init|path                   the hosts.toml file

Everything else (submit, kill, sync, fetch, the daemon, tui) is still the Python version.";

/// Shown at the end of `task COMMAND -h`.
fn examples(name: &str, rows: &[(&str, &str)]) -> String {
    let cmds: Vec<String> = rows
        .iter()
        .map(
            |(a, _)| {
                if a.starts_with("id=") {
                    a.to_string()
                } else {
                    format!("task {name} {a}").trim_end().to_string()
                }
            },
        )
        .collect();
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
    /// List tasks
    #[command(visible_alias = "ls", after_help = examples("list", &[("", "the 200 most recent tasks"), ("-a", "every task")]))]
    List(tasks::ListArgs),

    /// Show the details of a task
    #[command(after_help = examples("info", &[("12", "server, GPUs, directories, times, exit code")]))]
    Info(tasks::IdArg),

    /// Print a task's output
    #[command(after_help = examples("logs", &[
        ("12", "everything task 12 printed so far"),
        ("-f 12", "keep following it (Ctrl+C stops following, not the task)"),
    ]))]
    Logs(tasks::LogsArgs),

    /// Wait for tasks to finish. Exit code: 0 all done, the job's exit code if one failed, 130
    /// killed or cancelled, 75 lost, 124 timed out.
    #[command(after_help = examples("wait", &[
        ("12", "block until task 12 ends; exit code says how"),
        ("12 13 14 --timeout 3600", "all three, for at most an hour"),
        ("12 && task fetch results", "fetch only if the task succeeded"),
    ]))]
    Wait(tasks::WaitArgs),

    /// Show GPU availability across servers
    #[command(after_help = examples("gpus", &[
        ("", "one row per server"),
        ("-f", "only servers with a free GPU"),
        ("-d -H cuda14", "one row per GPU on cuda14"),
    ]))]
    Gpus(gpu_status::Args),

    /// Show configured hosts, reachability and network mode
    #[command(after_help = examples("hosts", &[("", "route, reachability and GPU count of every configured host")]))]
    Hosts,

    /// Run a shell command on every (selected) server in parallel
    #[command(after_help = examples("exec", &[
        ("-- nvidia-smi -L", "on every server"),
        ("-H cuda14,cuda15 -- \"df -h /vol/ecrg-solar\"", "on two servers"),
    ]))]
    Exec(cluster::ExecArgs),

    /// Open an interactive shell on a host
    #[command(after_help = examples("ssh", &[("cuda14", "a shell on cuda14, through the jump hosts if needed")]))]
    Ssh(cluster::SshArgs),

    /// Log in to the jump hosts once (password / OTP)
    #[command(after_help = examples("connect", &[("", "type the jump host's password / OTP once; the daemon reuses it")]))]
    Connect,

    /// Start, stop or check the background daemon (only `status` is ported)
    #[command(after_help = examples("daemon", &[("status", "is it running?")]))]
    Daemon(cluster::DaemonArgs),

    /// Write a hosts.toml template, or show where it is
    #[command(after_help = examples("config", &[
        ("init --user woodj4", "write ~/.gpu-spooler/hosts.toml (never overwrites)"),
        ("path", "which config file is in use"),
    ]))]
    Config(cluster::ConfigArgs),
}

fn run(cmd: Cmd) -> CmdResult {
    let ctx = Ctx::from_env();
    match cmd {
        Cmd::List(a) => tasks::list(&ctx, &a),
        Cmd::Info(a) => tasks::info(&ctx, &a),
        Cmd::Logs(a) => tasks::logs(&ctx, &a),
        Cmd::Wait(a) => tasks::wait(&ctx, &a),
        Cmd::Gpus(a) => Ok(gpu_status::run(&a)),
        Cmd::Hosts => cluster::hosts(&ctx),
        Cmd::Exec(a) => cluster::exec(&ctx, &a),
        Cmd::Ssh(a) => cluster::ssh(&ctx, &a),
        Cmd::Connect => cluster::connect(&ctx),
        Cmd::Daemon(a) => cluster::daemon(&ctx, &a),
        Cmd::Config(a) => cluster::config(&ctx, &a),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let code = match cli.command {
        None => {
            let _ = Cli::command().print_help();
            0
        }
        Some(cmd) => match run(cmd) {
            Ok(code) => code,
            Err(fail) if fail.message.is_empty() => fail.code,
            Err(fail) => {
                eprintln!("{}", fail.message);
                fail.code
            }
        },
    };
    ExitCode::from(code.clamp(0, 255) as u8)
}
