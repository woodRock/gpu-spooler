//! Repos on the servers: `task sync [--check]` pushes this git repo's working tree to its copy on
//! the servers; `task fetch` copies outputs back.

use crate::color::{colorize, BOLD, DIM, GREEN, RED, RESET, YELLOW};
use crate::ctx::{CmdResult, Ctx, Fail};
use gpu_spooler::core;
use gpu_spooler::db::Task;
use gpu_spooler::fetch::{self, FetchError, FetchResult};
use gpu_spooler::fmt::truncate;
use gpu_spooler::provenance;

#[derive(clap::Args, Debug)]
pub struct SyncArgs {
    /// Only report whether the copies are up to date (exit 1 if not)
    #[arg(long)]
    pub check: bool,
    /// Only these hosts, groups or tags (comma separated)
    #[arg(short = 'H', long, value_name = "SPEC")]
    pub hosts: Option<String>,
    /// The repo (default: the one containing the current directory)
    #[arg(short = 'C', long, value_name = "DIR")]
    pub dir: Option<String>,
}

#[derive(clap::Args, Debug)]
pub struct FetchArgs {
    #[arg(value_name = "PATH", required = true)]
    pub paths: Vec<String>,
    /// Wait for these tasks first; fetch only if all are done
    #[arg(long, value_name = "ID", num_args = 1..)]
    pub after: Vec<i64>,
    /// Also replace locally changed files
    #[arg(long)]
    pub force: bool,
    /// Show what would change, copy nothing
    #[arg(long)]
    pub dry_run: bool,
    /// Fetch from this host group (needed if several are selected)
    #[arg(short = 'H', long, value_name = "SPEC")]
    pub hosts: Option<String>,
    /// Resolve PATHs from this directory instead of the current one
    #[arg(short = 'C', long, value_name = "DIR")]
    pub dir: Option<String>,
}

pub fn sync(ctx: &Ctx, args: &SyncArgs) -> CmdResult {
    let transport = ctx.transport()?;
    let info = core::repo_for(transport, args.dir.as_deref(), args.hosts.as_deref())?;
    if args.check {
        println!("{BOLD}{}{RESET}  local: {}  at {}", info.repo.name(), info.repo.root, info.repo.describe());
        let mut stale = 0;
        for (target, problem) in core::repo_status(transport, &info.repo, &info.copies) {
            stale += problem.is_some() as usize;
            let status = match &problem {
                None => colorize("up to date", GREEN),
                Some(p) => colorize(p, YELLOW),
            };
            println!("  {:<10} {}  {status}", target.label(), target.path);
            println!("  {:<10} {DIM}used by: {}{RESET}", "", truncate(&target.members.join(", "), 70));
        }
        return Ok(if stale > 0 { 1 } else { 0 });
    }
    println!(
        "Syncing {BOLD}{}{RESET} ({}) to {} location(s)  {DIM}network: {}{RESET}",
        info.repo.name(),
        info.repo.describe(),
        info.copies.len(),
        transport.describe_mode(true)
    );
    let mut failed = 0;
    for res in core::sync_repo(transport, &info) {
        let status = if res.ok { colorize("OK  ", GREEN) } else { colorize("FAIL", RED) };
        failed += !res.ok as usize;
        println!("  {:<10} {status}  {}", res.target.label(), res.message);
    }
    let paths: Vec<String> = info.copies.iter().map(|t| t.path.clone()).collect();
    let busy = core::running_in(&ctx.db()?, &paths)?;
    if busy > 0 {
        println!(
            "{}",
            colorize(&format!("  Note: {busy} running task(s) use this checkout and now see the new files."), YELLOW)
        );
    }
    Ok(if failed > 0 { 1 } else { 0 })
}

fn fetch_fail(e: FetchError) -> Fail {
    Fail::new(format!("task: {e}"))
}

pub fn fetch(ctx: &Ctx, args: &FetchArgs) -> CmdResult {
    let mut done: Vec<Task> = vec![];
    if !args.after.is_empty() {
        let mut ids: Vec<i64> = vec![];
        for id in &args.after {
            if !ids.contains(id) {
                ids.push(*id);
            }
        }
        done = core::wait_for(&ctx.db()?, &ids, None, core::WAIT_POLL, &|| false)?.0;
        let bad: Vec<String> = done.iter().filter(|t| t.status != "done").map(core::describe_end).collect();
        if !bad.is_empty() {
            return Err(Fail::new(format!("task: not fetching, a task did not finish cleanly: {}", bad.join("; "))));
        }
    }
    let transport = ctx.transport()?;
    let cfg = &transport.cfg;
    let cwd = core::working_dir(&ctx.paths, args.dir.as_deref());
    if core::on_ecs_filesystem(&cwd) && !args.paths.iter().any(|s| fetch::is_explicit(s)) {
        return Err(Fail::new("task: this directory is on the servers' filesystem already; nothing to fetch"));
    }
    let hosts = cfg
        .select(args.hosts.as_deref())
        .map_err(|e| Fail::new(format!("task: -H {}: {e}", args.hosts.as_deref().unwrap_or("None"))))?;
    let (_, items) = fetch::resolve(cfg, &ctx.paths, &hosts, &args.paths, &cwd).map_err(fetch_fail)?;
    let remotes: Vec<String> = items.iter().map(|i| i.remote.clone()).collect();
    let busy = core::running_in(&ctx.db()?, &remotes)?;
    if busy > 0 && args.after.is_empty() {
        println!(
            "{}",
            colorize(
                &format!(
                    "Warning: {busy} running task(s) work under these paths; files may be incomplete (use --after ID \
                     to wait for them)."
                ),
                YELLOW
            )
        );
    }
    let res = fetch::fetch(transport, &hosts, &args.paths, &cwd, args.force, args.dry_run).map_err(fetch_fail)?;

    let verb = if args.dry_run { "Would fetch" } else { "Fetched" };
    let mut kept_any = false;
    for plan in &res.plans {
        let it = plan.item();
        let place = format!("{}:{} → {}", res.host.name, it.remote, gpu_spooler::config::relpath(&it.local, &cwd));
        if plan.kind == "missing" {
            println!("  {}  {place}", colorize("MISSING", RED));
            continue;
        }
        let mut parts = vec![
            format!("{} new", plan.new.len()),
            format!("{} updated", plan.update.len()),
            format!("{} unchanged", plan.same.len()),
        ];
        if !plan.kept.is_empty() {
            parts.push(colorize(&format!("{} kept (local changes)", plan.kept.len()), YELLOW));
        }
        if !plan.conflict.is_empty() {
            parts.push(colorize(&format!("{} conflict (local directory)", plan.conflict.len()), RED));
        }
        let skipped = res.skipped.get(&it.spec).cloned().unwrap_or_default();
        if !skipped.is_empty() {
            parts.push(colorize(&format!("{} skipped (unsafe link)", skipped.len()), YELLOW));
        }
        println!("  {verb} {:>4}  {place}  ({})", plan.to_fetch().len(), parts.join(", "));
        for rel in plan.kept.iter().chain(&plan.conflict).take(20) {
            let shown = if rel.is_empty() { it.spec.clone() } else { fetch::join(&it.spec, rel) };
            println!("           {DIM}kept: {shown}{RESET}");
        }
        for m in skipped.iter().take(20) {
            println!("           {DIM}skipped: {m}{RESET}");
        }
        kept_any |= !plan.kept.is_empty() || !plan.conflict.is_empty();
    }
    if kept_any {
        println!(
            "{}",
            colorize(
                "Local files that changed since they were fetched, or that git cannot restore, were kept; --force \
                 replaces them.",
                YELLOW
            )
        );
    }
    if !args.after.is_empty() && !args.dry_run {
        record_outputs(ctx, &done, &res);
    }
    Ok(if res.plans.iter().any(|p| p.kind == "missing") { 1 } else { 0 })
}

/// With --after, what this fetch brought back new or changed is those tasks' output: record it
/// with taura (if the repo keeps a taura ledger). With several tasks, each file goes to the task
/// that wrote it (fetch::attribute).
fn record_outputs(ctx: &Ctx, done: &[Task], res: &FetchResult) {
    let Ok(cfg) = ctx.cfg() else { return };
    let outputs: Vec<String> = res
        .plans
        .iter()
        .flat_map(|plan| plan.to_fetch().into_iter().map(|rel| plan.dest(&rel)).collect::<Vec<_>>())
        .filter(|p| std::path::Path::new(p).is_file())
        .collect();
    let (by_task, shared) = fetch::attribute(done, &outputs, &ctx.paths.home);
    if !shared.is_empty() {
        let ids: Vec<String> = done.iter().map(|t| t.id.to_string()).collect();
        println!(
            "{}",
            colorize(
                &format!(
                    "Warning: {} file(s) could not be told apart between tasks {} (no command names them, and the \
                     tasks ran at the same time); recorded on each. Fetch each task's outputs on its own to keep \
                     them apart.",
                    shared.len(),
                    ids.join(", ")
                ),
                YELLOW
            )
        );
        for p in shared.iter().take(10) {
            println!("           {DIM}{}{RESET}", gpu_spooler::config::relpath(p, "."));
        }
    }
    for t in done {
        let mine = &by_task[&t.id];
        let log = |m: &str| println!("{}", colorize(m, YELLOW));
        if provenance::notify("fetch", t, &cfg.taura, mine, &log) {
            println!("  Recorded {} output(s) of task {} as taura run {}.", mine.len(), t.id, provenance::run_id(t));
        }
    }
}
