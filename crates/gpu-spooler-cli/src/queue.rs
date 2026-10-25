//! Changing the queue: `task submit`, `task resubmit`, `task kill`, `task clear`.

use crate::color::{colorize, DIM, GREEN, RED, YELLOW};
use crate::ctx::{CmdResult, Ctx, Fail};
use gpu_spooler::core::{self, SubmitOpts, Submitted};

#[derive(clap::Args, Debug)]
pub struct SubmitArgs {
    /// GPUs the task needs (default 1; 0 = CPU-only, on hosts with cpu_slots)
    #[arg(short = 'g', long, value_name = "N", default_value_t = 1, allow_negative_numbers = true)]
    pub gpus: i64,
    /// Only run on these hosts, groups or tags (comma separated)
    #[arg(short = 'H', long, value_name = "SPEC")]
    pub hosts: Option<String>,
    /// Only GPUs nvidia-smi names this model, e.g. "RTX A5000" (exact, ignoring case and 'NVIDIA ';
    /// comma separated for several)
    #[arg(short = 't', long, value_name = "MODEL")]
    pub gpu_type: Option<String>,
    /// Only GPUs with at least this much memory, e.g. 40
    #[arg(short = 'm', long, value_name = "GB", default_value_t = 0)]
    pub min_mem: i64,
    /// A label shown in the task list
    #[arg(short = 'n', long, value_name = "LABEL")]
    pub name: Option<String>,
    /// Working directory (default: here; mapped by path_map or remote_root)
    #[arg(short = 'C', long, value_name = "DIR")]
    pub dir: Option<String>,
    /// Print only the task ID (for scripts)
    #[arg(short, long)]
    pub quiet: bool,
    /// Don't record this task as a taura run (a one-off check, not an experiment)
    #[arg(long)]
    pub no_record: bool,
    /// The command; one quoted argument is run as shell code
    #[arg(value_name = "-- CMD", trailing_var_arg = true, allow_hyphen_values = true)]
    pub command: Vec<String>,
}

#[derive(clap::Args, Debug)]
pub struct ResubmitArgs {
    #[arg(value_name = "ID")]
    pub id: i64,
    /// Print only the new task ID
    #[arg(short, long)]
    pub quiet: bool,
}

#[derive(clap::Args, Debug)]
pub struct KillArgs {
    #[arg(value_name = "ID")]
    pub ids: Vec<i64>,
    /// Every queued and running task
    #[arg(long)]
    pub all: bool,
}

#[derive(clap::Args, Debug)]
pub struct ClearArgs {
    /// Also forget everything else and restart IDs at 1 (no task may be queued or running)
    #[arg(long)]
    pub reset: bool,
}

fn print_submitted(ctx: &Ctx, sub: &Submitted) {
    let label = sub.name.as_ref().filter(|n| !n.is_empty()).map_or(String::new(), |n| format!("  \"{n}\""));
    println!("Queued task {}{label}: {}", sub.task_id, sub.command);
    let need = if sub.num_gpus == 0 {
        "CPU only".to_string()
    } else {
        format!("{} GPU(s) (min {}GB VRAM)", sub.num_gpus, sub.min_mem)
    };
    println!("  Requires: {need}   Working dir: {}", sub.working_dir);
    if sub.host_spec.is_some() {
        println!("  Hosts: {}", sub.candidates.join(", "));
    }
    if let Some(t) = &sub.gpu_type {
        println!("  GPU type: {t}");
    }
    for w in &sub.warnings {
        println!("{}", colorize(&format!("  Warning: {w}"), YELLOW));
    }
    let daemon = if core::daemon_pid(&ctx.paths).is_some() {
        colorize("running", GREEN)
    } else {
        colorize("NOT running — start with: task daemon start", RED)
    };
    println!("  Daemon: {daemon}");
}

fn report_submitted(ctx: &Ctx, sub: &Submitted, quiet: bool) -> CmdResult {
    if !quiet {
        print_submitted(ctx, sub);
        return Ok(0);
    }
    println!("{}", sub.task_id);
    for w in &sub.warnings {
        eprintln!("task: warning: {w}");
    }
    Ok(0)
}

pub fn submit(ctx: &Ctx, args: &SubmitArgs) -> CmdResult {
    if args.command.is_empty() {
        return Err(Fail::new("task submit: no command given, e.g.: task submit -- python3 train.py"));
    }
    let command = gpu_spooler::shell::command_from_argv(&args.command);
    core::check_submit(args.gpus, &command, args.gpu_type.as_deref())?;
    let opts = SubmitOpts {
        command,
        num_gpus: args.gpus,
        min_mem: args.min_mem,
        name: args.name.clone(),
        host_spec: args.hosts.clone(),
        working_dir: args.dir.clone(),
        gpu_type: args.gpu_type.clone(),
        no_record: args.no_record,
        start: true,
        check_repo: true,
    };
    let sub = core::submit(ctx.transport()?, &opts)?;
    report_submitted(ctx, &sub, args.quiet)
}

pub fn resubmit(ctx: &Ctx, args: &ResubmitArgs) -> CmdResult {
    let t = core::get_task(&ctx.db()?, args.id)?;
    core::check_submit(t.num_gpus, &t.command, t.gpu_type.as_deref())?;
    let sub = core::resubmit(ctx.transport()?, args.id, true)?;
    report_submitted(ctx, &sub, args.quiet)
}

pub fn kill(ctx: &Ctx, args: &KillArgs) -> CmdResult {
    if args.all != args.ids.is_empty() {
        return Err(Fail::new("task kill: give task IDs or --all (not both)"));
    }
    let transport = ctx.transport()?;
    if args.all {
        let res = core::kill_all(transport)?;
        for k in &res.unconfirmed {
            println!("{}", colorize(&format!("Task {}: Warning: {}", k.task_id, k.message), YELLOW));
        }
        println!("Cancelled {} queued task(s), killed {} running task(s).", res.cancelled, res.killed);
        return Ok(0);
    }
    let mut seen = vec![];
    for &id in &args.ids {
        if seen.contains(&id) {
            continue;
        }
        seen.push(id);
        let res = core::kill(transport, id)?;
        if res.action == "killed" {
            if res.confirmed {
                println!("{}", res.message);
            } else {
                println!("{}", colorize(&format!("Warning: {}", res.message), YELLOW));
            }
            println!("Task {id} marked as killed.");
        } else {
            println!("{}", res.message);
        }
    }
    Ok(0)
}

pub fn clear(ctx: &Ctx, args: &ClearArgs) -> CmdResult {
    if args.reset {
        let counts = gpu_spooler::db::task_counts(&ctx.db()?)?;
        if counts["running"] > 0 || counts["queued"] > 0 {
            return Err(Fail::new(format!(
                "task clear --reset: {} running and {} queued task(s); kill them first (task kill --all)",
                counts["running"], counts["queued"]
            )));
        }
    }
    let res = core::clear_finished(ctx.transport()?)?;
    let failed = if res.failed_hosts.is_empty() {
        String::new()
    } else {
        colorize(&format!("  Could not clean job directories on: {}", res.failed_hosts.join(", ")), DIM)
    };
    println!("Removed {} finished task(s) and {} log file(s).{failed}", res.tasks, res.log_files);
    if args.reset {
        core::reset_history(&ctx.paths)?;
        println!("Task history cleared. The next task will be ID 1.");
    }
    Ok(0)
}
