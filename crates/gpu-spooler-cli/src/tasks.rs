//! Reading the queue: `task list`, `task info`, `task logs`, `task wait`. They only read the
//! database (and the log files the daemon writes), never the servers.

use crate::color::{colorize, status_color, BOLD, DIM, GREEN, RED, RESET, YELLOW};
use crate::ctx::{CmdResult, Ctx, Fail};
use crate::gpu_status::load_idle;
use gpu_spooler::core::{self, describe_end, get_task, runtime, task_details, wait_exit_code};
use gpu_spooler::db::{self, Task};
use gpu_spooler::fmt::{fmt_duration, fmt_ts, one_line, truncate};
use gpu_spooler::idle::fmt_idle;
use gpu_spooler::py;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[derive(clap::Args, Debug)]
pub struct ListArgs {
    /// All tasks, not only the last 200
    #[arg(short, long)]
    pub all: bool,
}

#[derive(clap::Args, Debug)]
pub struct IdArg {
    #[arg(value_name = "ID")]
    pub id: i64,
}

#[derive(clap::Args, Debug)]
pub struct LogsArgs {
    #[arg(value_name = "ID")]
    pub id: i64,
    /// Keep printing new output
    #[arg(short, long)]
    pub follow: bool,
}

#[derive(clap::Args, Debug)]
pub struct WaitArgs {
    #[arg(value_name = "ID", required = true)]
    pub ids: Vec<i64>,
    /// Give up after S seconds (exit 124; the tasks keep running)
    #[arg(long, value_name = "S")]
    pub timeout: Option<f64>,
}

/// Running tasks whose every GPU gpu-detective has seen held idle: id → 'idle 1h05m'.
fn idle_tasks(ctx: &Ctx, rows: &[Task], out: &mut dyn Write) -> Result<BTreeMap<i64, String>, Fail> {
    let running = |t: &&Task| t.status == "running" && t.gpu_indices.as_deref().is_some_and(|g| !g.is_empty());
    if !rows.iter().any(|t| running(&t)) {
        return Ok(BTreeMap::new());
    }
    let Some(idle) = load_idle(out, ctx.cfg()?)?.filter(|i| !i.is_empty()) else { return Ok(BTreeMap::new()) };
    let mut found = BTreeMap::new();
    for t in rows.iter().filter(running) {
        let Some(server) = &t.server else { continue };
        let gs: Vec<_> = t
            .gpu_indices
            .as_deref()
            .unwrap_or("")
            .split(',')
            .filter(|i| py::isdigit(py::strip(i)))
            .filter_map(py::int)
            .map(|i| idle.get(&(server.clone(), i)))
            .collect();
        if !gs.is_empty() && gs.iter().all(Option::is_some) {
            let mut least = gs[0].unwrap();
            for g in gs.iter().flatten() {
                if g.idle_seconds < least.idle_seconds {
                    least = g;
                }
            }
            found.insert(t.id, fmt_idle(least));
        }
    }
    Ok(found)
}

fn or_dash(v: &Option<String>) -> &str {
    v.as_deref().filter(|s| !s.is_empty()).unwrap_or("-")
}

pub fn list(ctx: &Ctx, args: &ListArgs) -> CmdResult {
    let con = ctx.db()?;
    let rows = db::list_tasks(&con, if args.all { None } else { Some(200) })?;
    let mut out = std::io::stdout();
    if rows.is_empty() {
        writeln!(out, "No tasks.")?;
        return Ok(0);
    }

    let hdr = format!(
        "{:>4}  {:10}  {:>1}  {:<10}  {:<14}  {:>4}  {:11}  {:>8}  LABEL / COMMAND",
        "ID", "STATUS", "G", "HOST", "SERVER", "GPU", "SUBMITTED", "RUNTIME"
    );
    let sep = "─".repeat(hdr.chars().count().max(80));
    writeln!(out, "\n{BOLD}{hdr}{RESET}")?;
    writeln!(out, "{sep}")?;

    let idle = idle_tasks(ctx, &rows, &mut out)?;
    for t in rows.iter().rev() {
        let name = t.name.as_deref().filter(|n| !n.is_empty()).unwrap_or(&t.command);
        let label = truncate(&one_line(name), 42);
        let spec = truncate(or_dash(&t.host_spec), 10);
        let note = idle.get(&t.id).map_or(String::new(), |i| colorize(&format!("  ⚠ {i}"), YELLOW));
        writeln!(
            out,
            "{:>4}  {}  {:>1}  {:<10}  {:<14}  {:>4}  {:11}  {:>8}  {label}{note}",
            t.id,
            colorize(&format!("{:10}", t.status), status_color(&t.status)),
            t.num_gpus,
            spec,
            or_dash(&t.server),
            or_dash(&t.gpu_indices),
            fmt_ts(Some(t.submitted_at)),
            fmt_duration(runtime(t)),
        )?;
    }

    writeln!(out, "{sep}")?;
    if !idle.is_empty() {
        let msg = format!(
            "{} running task(s) have held their GPUs under 5% utilisation for {}m+ (gpu-detective). They still \
             count against max_gpus.",
            idle.len(),
            ctx.cfg()?.idle_minutes
        );
        writeln!(out, "{}", colorize(&msg, YELLOW))?;
    }
    let shown = rows.len() as i64;
    let counts = db::task_counts(&con)?;
    let daemon =
        if core::daemon_pid(&ctx.paths).is_some() { colorize("running", GREEN) } else { colorize("stopped", RED) };
    let total = counts["total"];
    let truncated = if shown < total {
        format!("  {}", colorize(&format!("(showing {shown} of {total})"), DIM))
    } else {
        String::new()
    };
    let cap = ctx.cfg()?.max_gpus;
    let in_use = db::gpus_in_use(&con)?;
    let gpus = cap.map_or(String::new(), |cap| format!("GPUs {in_use}/{cap} (max_gpus)  "));
    writeln!(
        out,
        "\n{total} task(s){truncated}  {} running  {} queued  {gpus}daemon: {daemon}  network: {}\n",
        colorize(&counts["running"].to_string(), GREEN),
        colorize(&counts["queued"].to_string(), YELLOW),
        ctx.transport()?.describe_mode(false),
    )?;
    Ok(0)
}

pub fn info(ctx: &Ctx, args: &IdArg) -> CmdResult {
    let t = get_task(&ctx.db()?, args.id)?;
    let snapshot = core::read_gpu_snapshot(&ctx.paths);
    let cfg = ctx.cfg().ok();
    let mut out = std::io::stdout();
    writeln!(out, "\n{BOLD}Task {}{RESET}", t.id)?;
    for (label, value) in task_details(&t, cfg, snapshot.as_ref(), gpu_spooler::transport::now()) {
        let value = if label == "Status" { colorize(&value, status_color(&value)) } else { value };
        writeln!(out, "  {:<13}{value}", format!("{label}:"))?;
    }
    writeln!(out)?;
    Ok(0)
}

fn log_of(con: &rusqlite::Connection, id: i64) -> Result<Option<(Option<String>, String)>, Fail> {
    let row = con.query_row("SELECT log_file, status FROM tasks WHERE id=?", [id], |r| {
        Ok((r.get::<_, Option<String>>(0)?, r.get::<_, Option<String>>(1)?.unwrap_or_default()))
    });
    match row {
        Ok(r) => Ok(Some(r)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn exists(log: &Option<String>) -> bool {
    log.as_deref().is_some_and(|l| !l.is_empty() && Path::new(l).exists())
}

pub fn logs(ctx: &Ctx, args: &LogsArgs) -> CmdResult {
    let con = ctx.db()?;
    let Some((mut log, status)) = log_of(&con, args.id)? else {
        return Err(Fail::new(format!("task: task {} not found", args.id)));
    };
    if !args.follow {
        if !exists(&log) {
            return Err(Fail::new(format!("task: no output for task {} yet ({status})", args.id)));
        }
        std::io::stdout().write_all(&std::fs::read(log.as_deref().unwrap())?)?;
        return Ok(0);
    }

    // Wait for the log file to appear if the task is queued or starting
    if !exists(&log) {
        if !core::ACTIVE.contains(&status.as_str()) {
            return Err(Fail::new(format!("task: no output for task {} ({status})", args.id)));
        }
        eprintln!("Task {} has not started yet. Waiting...", args.id);
        let mut started = false;
        for _ in 0..60 {
            std::thread::sleep(Duration::from_secs(1));
            log = log_of(&con, args.id)?.and_then(|r| r.0);
            if exists(&log) {
                started = true;
                break;
            }
        }
        if !started {
            return Err(Fail::new("task: timed out waiting for the task to start"));
        }
    }
    eprintln!("Following task {} (Ctrl+C to stop)...\n", args.id);
    let mut child = std::process::Command::new("tail").args(["-n", "+1", "-f"]).arg(log.unwrap()).spawn()?;
    // Ctrl+C stops following, not this command: tail is killed and we exit 0, as Python does.
    let pid = Arc::new(AtomicI32::new(child.id() as i32));
    let _ = ctrlc::set_handler(move || {
        // SAFETY: kill on a child PID we spawned.
        unsafe { libc::kill(pid.load(Ordering::SeqCst), libc::SIGKILL) };
    });
    child.wait()?;
    Ok(0)
}

pub fn wait(ctx: &Ctx, args: &WaitArgs) -> CmdResult {
    let mut ids: Vec<i64> = vec![];
    for id in &args.ids {
        if !ids.contains(id) {
            ids.push(*id);
        }
    }
    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let flag = interrupted.clone();
        let _ = ctrlc::set_handler(move || flag.store(true, Ordering::SeqCst));
    }
    let timeout = args.timeout.map(|s| Duration::from_secs_f64(s.max(0.0)));
    let con = ctx.db()?;
    let (done, pending) = core::wait_for(&con, &ids, timeout, core::WAIT_POLL, &|| interrupted.load(Ordering::SeqCst))?;
    if interrupted.load(Ordering::SeqCst) {
        return Err(Fail { message: "task: stopped waiting (tasks keep running)".into(), code: core::EXIT_KILLED });
    }
    for id in &ids {
        match done.iter().find(|t| t.id == *id) {
            Some(t) => eprintln!("{}", describe_end(t)),
            None => eprintln!(
                "task {id}: still {} (timed out after {}s)",
                get_task(&con, *id)?.status,
                py::format_g(args.timeout.unwrap_or(0.0))
            ),
        }
    }
    if !pending.is_empty() {
        return Ok(core::EXIT_TIMEOUT);
    }
    let mut sorted = ids.clone();
    sorted.sort();
    Ok(sorted
        .iter()
        .map(|id| wait_exit_code(done.iter().find(|t| t.id == *id).expect("every id is done")))
        .find(|c| *c != 0)
        .unwrap_or(0))
}
