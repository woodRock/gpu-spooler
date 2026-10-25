//! `task` - the gpu-spooler command line (Rust port, in progress).
//!
//! Ported so far: the queue, the daemon and the cluster commands. Still Python: sync, fetch, tui.

mod cluster;
mod color;
mod ctx;
mod gpu_status;
mod queue;
mod tasks;

use clap::{CommandFactory, Parser, Subcommand};
use ctx::{CmdResult, Ctx};
use std::process::ExitCode;

const ABOUT: &str = "\
task - GPU task spooler for SSH-reachable GPU servers (VUW ECS by default)

The Rust port of gpu-spooler, in progress. Ported so far:

Tasks:
  task submit [-g N] [OPTIONS] -- CMD     queue CMD (1 GPU unless -g says otherwise)
  task list [-a]                          list tasks (the 200 most recent; -a: all)
  task info ID                            details of a task
  task logs [-f] ID                       print a task's output (-f: follow it)
  task wait ID... [--timeout S]           wait; the exit code says how they ended
  task kill ID... | --all                 cancel queued / kill running tasks
  task resubmit ID                        queue a copy of a task
  task clear [--reset]                    remove finished tasks (--reset: all, IDs from 1)

Cluster:
  task gpus [-d] [-f]                     GPU availability
  task hosts                              configured hosts and network mode
  task exec [-H SPEC] -- CMD              run CMD on every (selected) server
  task ssh HOST                           interactive shell on a host
  task connect                            log in to jump hosts once (password/OTP)
  task daemon start|stop|status           the background daemon
  task config init|path                   the hosts.toml file

Still the Python version: sync, fetch and tui.";

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
    /// Queue a command to run on a GPU server
    #[command(after_help = examples("submit", &[
        ("python3 train.py --epochs 50", "1 GPU on any free server"),
        ("-g 2 -n \"ablation\" -- ./run.sh --lr 3e-4", "2 GPUs, with a label"),
        ("-t \"RTX A5000\" -- python3 eval.py", "only on RTX A5000 GPUs"),
        ("-m 40 -H cuda14,cuda15 -- python3 big.py", "48 GB cards on these hosts"),
        ("-g 0 -H local -- 'make plots && echo done'", "CPU job; one argument = shell code"),
        ("id=$(task submit -q -- python3 train.py)", "keep the ID for wait / fetch"),
    ]))]
    Submit(queue::SubmitArgs),

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

    /// Cancel queued or kill running tasks
    #[command(after_help = examples("kill", &[
        ("12", "cancel it if queued, kill it if running"),
        ("12 13 14", "several at once"),
        ("--all", "every queued and running task"),
    ]))]
    Kill(queue::KillArgs),

    /// Queue a copy of a task (same command and options)
    #[command(after_help = examples("resubmit", &[("12", "queue a copy (same command, GPUs, hosts, directory)")]))]
    Resubmit(queue::ResubmitArgs),

    /// Remove finished tasks, their logs and job directories
    #[command(after_help = examples("clear", &[
        ("", "forget finished tasks and delete their logs"),
        ("--reset", "forget everything and restart IDs at 1"),
    ]))]
    Clear(queue::ClearArgs),

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

    /// Start, stop or check the background daemon
    #[command(after_help = examples("daemon", &[
        ("status", "is it running?"),
        ("stop", "running jobs keep running"),
        ("start", "it also starts by itself on submit"),
    ]))]
    Daemon(cluster::DaemonArgs),

    /// Write a hosts.toml template, or show where it is
    #[command(after_help = examples("config", &[
        ("init --user woodj4", "write ~/.gpu-spooler/hosts.toml (never overwrites)"),
        ("path", "which config file is in use"),
    ]))]
    Config(cluster::ConfigArgs),

    /// The daemon itself (started by `task daemon start` and submit)
    #[command(name = "__daemon", hide = true)]
    RunDaemon,
}

fn run(cmd: Cmd) -> CmdResult {
    let ctx = Ctx::from_env();
    match cmd {
        Cmd::Submit(a) => queue::submit(&ctx, &a),
        Cmd::Kill(a) => queue::kill(&ctx, &a),
        Cmd::Resubmit(a) => queue::resubmit(&ctx, &a),
        Cmd::Clear(a) => queue::clear(&ctx, &a),
        Cmd::RunDaemon => Ok(gpu_spooler::daemon::main(ctx.paths)),
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
