//! Optional run provenance: record each task as a taura run (github.com/woodRock/taura).
//!
//! When `taura` is installed and the task's local repo has a taura ledger (a `ledger/` directory),
//! the daemon calls `taura record` when a task starts and finishes, and `task fetch --after ID`
//! calls it with the files it brought back. The record lands in the local repo's ledger/runs/,
//! with the commit the server actually ran (captured at launch), the host, GPUs and exit code.
//! An SGE task starts when a grid machine starts it, so that is when its "start" is recorded.
//! taura is never required: without it, or with `[provenance] taura = "off"`, nothing happens. A
//! failing `taura record` is logged and never affects the task.

use crate::config::realpath;
use crate::db::Task;
use crate::fmt::fmt_duration;
use crate::jobs::ENV_MARK;
use crate::py;
use crate::shell;
use chrono::{DateTime, Local, TimeZone};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;
use wait_timeout::ChildExt;

pub const RECORD_TIMEOUT: Duration = Duration::from_secs(60);

/// Stable per task attempt: `<submit date>-spooler-<task id>[-<attempt>]`.
pub fn run_id(t: &Task) -> String {
    run_id_in(t, &Local)
}

pub fn run_id_in<Tz: TimeZone>(t: &Task, tz: &Tz) -> String
where
    Tz::Offset: std::fmt::Display,
{
    let day = DateTime::from_timestamp(t.submitted_at.floor() as i64, 0)
        .map_or(String::new(), |d| d.with_timezone(tz).format("%Y-%m-%d").to_string());
    let attempt = t.retry_count + 1;
    format!("{day}-spooler-{}{}", t.id, if attempt > 1 { format!("-{attempt}") } else { String::new() })
}

fn git_toplevel(start: &str) -> Option<String> {
    let out = Command::new("git").args(["-C", start, "rev-parse", "--show-toplevel"]).output().ok()?;
    out.status.success().then(|| py::strip(&String::from_utf8_lossy(&out.stdout)).to_string())
}

/// The local git repo the task came from, if it keeps a taura ledger.
pub fn ledger_repo(t: &Task) -> Option<String> {
    let start = t.repo_root.clone().filter(|r| !r.is_empty()).unwrap_or_else(|| t.working_dir.clone());
    if start.is_empty() || !Path::new(&start).is_dir() {
        return None;
    }
    let top = git_toplevel(&start).unwrap_or(start);
    Path::new(&top).join("ledger").is_dir().then_some(top)
}

/// `taura`, on PATH or where it gets installed: a daemon the desktop app started has launchd's
/// PATH, which has none of these.
pub fn taura_exe() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    py::which("taura").or_else(|| {
        [home.join(".local/bin"), "/opt/homebrew/bin".into(), "/usr/local/bin".into()]
            .into_iter()
            .map(|d| d.join("taura"))
            .find(|exe| exe.is_file())
    })
}

/// Will this task be recorded with taura? (taura installed, repo keeps a ledger, not turned off in
/// hosts.toml or for this task with task submit --no-record)
pub fn enabled(t: &Task, setting: &str) -> bool {
    setting != "off" && t.no_record == 0 && taura_exe().is_some() && ledger_repo(t).is_some()
}

/// How to run the probe like the job's command: its VAR=value prefix and its python (python3 if
/// the command doesn't start with one, e.g. bash -c or torchrun).
pub fn probe_command(command: &str) -> String {
    let mut words = shell::split(command).unwrap_or_default();
    let is_assignment = |w: &str| {
        let Some((name, _)) = w.split_once('=') else { return false };
        let mut chars = name.chars();
        chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    };
    let mut env = vec![];
    while words.first().is_some_and(|w| is_assignment(w)) {
        env.push(words.remove(0));
    }
    let is_python = |w: &str| {
        let base = w.rsplit('/').next().unwrap_or(w);
        let Some(version) = base.strip_prefix("python") else { return false };
        version.is_empty() || version.split('.').all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
    };
    let py = match words.first() {
        Some(w) if is_python(w) => w.clone(),
        _ => "python3".to_string(),
    };
    env.into_iter().chain([py]).map(|w| shell::quote(&w)).collect::<Vec<_>>().join(" ")
}

/// The environment a job reported (its last TAURA_ENV line), if any.
pub fn env_from_log(path: Option<&str>) -> Option<serde_json::Value> {
    let bytes = std::fs::read(path?).ok()?;
    let text = String::from_utf8_lossy(&bytes);
    let mut last = None;
    for line in text.split_inclusive('\n') {
        if let Some(rest) = line.strip_prefix(ENV_MARK) {
            last = Some(rest);
        }
    }
    serde_json::from_str(last?).ok()
}

fn relpath_in(path: &str, repo: &str) -> String {
    crate::config::relpath(&realpath(path), &realpath(repo))
}

/// How many tracked files were changed on the server, and which, so taura can tell an output an
/// earlier job rewrote from an edit to the code.
fn dirty_args(t: &Task) -> Vec<String> {
    let mut args = vec!["--dirty-count".into(), t.git_dirty.unwrap_or(0).to_string()];
    if let Some(paths) = &t.git_dirty_paths {
        args.push("--dirty-paths".into());
        args.extend(paths.split('\n').filter(|p| !p.is_empty()).map(String::from));
    }
    args
}

/// Where it ran: the grid machine, if SGE chose one.
fn host_of(t: &Task) -> Option<&String> {
    t.machine.as_ref().or(t.server.as_ref()).filter(|s| !s.is_empty())
}

/// What the record's times and status don't say.
fn note(t: &Task) -> Option<String> {
    let mut notes = vec![];
    if t.status == "lost" {
        notes.push(if t.started_at.is_some() {
            "Lost: the job disappeared without an exit code; finished is when gpu-spooler noticed.".to_string()
        } else {
            "Lost: the job disappeared from SGE before it started, so it never ran.".to_string()
        });
    }
    if let Some(secs) = t.sge_suspended.filter(|s| *s != 0.0) {
        let took = t.started_at.zip(t.finished_at).map(|(s, f)| f - s - secs);
        notes.push(format!(
            "SGE suspended it for {} between started and finished, so it ran for {}.",
            fmt_duration(Some(secs)),
            fmt_duration(took)
        ));
    }
    (!notes.is_empty()).then(|| notes.join(" "))
}

/// The `taura record` arguments for this event, or None if this task isn't tracked.
pub fn record_args(event: &str, t: &Task, outputs: &[String]) -> Option<Vec<String>> {
    let repo = ledger_repo(t)?;
    let mut args: Vec<String> =
        ["-C", &repo, "record", &run_id(t), "--event", event, "--runner", "gpu-spooler"].map(String::from).to_vec();
    args.extend(["--spooler-job".into(), t.id.to_string()]);
    let commit = t.git_commit.as_ref().filter(|c| !c.is_empty());
    match event {
        "start" => {
            args.extend(["--command".into(), t.command.clone()]);
            let wdir = realpath(&t.working_dir);
            let root = realpath(&repo);
            if wdir != root && wdir.starts_with(&format!("{root}/")) {
                args.extend(["--cwd".into(), crate::config::relpath(&wdir, &root)]);
            }
            if let Some(s) = host_of(t) {
                args.extend(["--host".into(), s.clone()]);
            }
            if let Some(g) = &t.gpu_indices {
                args.extend(["--gpus".into(), g.clone()]);
            }
            // A task that never began (lost in the SGE queue) ran for no time at all.
            if let Some(s) = t.started_at.or(t.finished_at).filter(|s| *s != 0.0) {
                args.extend(["--started".into(), py::repr_float(s)]);
            }
            match commit {
                Some(c) => {
                    args.extend(["--commit".into(), c.clone()]);
                    args.extend(dirty_args(t));
                }
                // the server didn't report its checkout: say so, rather than let taura assume this machine's
                None => args.extend(["--commit".into(), "unknown".into()]),
            }
        }
        "finish" => {
            // SGE may have restarted the job elsewhere since "start".
            if let Some(s) = host_of(t) {
                args.extend(["--host".into(), s.clone()]);
            }
            if let Some(s) = t.started_at.filter(|s| *s != 0.0) {
                args.extend(["--started".into(), py::repr_float(s)]);
            }
            if let Some(n) = note(t) {
                args.extend(["--note".into(), n]);
            }
            if let Some(c) = t.exit_code {
                args.extend(["--exit-code".into(), c.to_string()]);
            }
            if let Some(f) = t.finished_at.filter(|f| *f != 0.0) {
                args.extend(["--finished".into(), py::repr_float(f)]);
            }
            if let Some(c) = commit {
                args.extend(["--commit".into(), c.clone()]);
                args.extend(dirty_args(t));
            }
        }
        "fetch" => {
            if outputs.is_empty() {
                return None;
            }
            args.push("--outputs".into());
            args.extend(outputs.iter().map(|p| relpath_in(p, &repo)));
        }
        _ => {}
    }
    Some(args)
}

/// Has this attempt a record yet?
fn recorded(t: &Task) -> bool {
    ledger_repo(t).is_some_and(|repo| Path::new(&repo).join("ledger/runs").join(format!("{}.toml", run_id(t))).exists())
}

/// Record a task event with taura, if enabled and available. True if a record was written.
pub fn notify(event: &str, t: &Task, setting: &str, outputs: &[String], log: &dyn Fn(&str)) -> bool {
    if setting == "off" || t.no_record != 0 {
        return false;
    }
    let Some(exe) = taura_exe() else {
        if ledger_repo(t).is_some() {
            log(&format!("Task {}: not recorded: the repo has a taura ledger, but taura was not found.", t.id));
        }
        return false;
    };
    let Some(mut args) = record_args(event, t, outputs) else { return false };
    // Without a "start" (a job lost in the queue, a daemon that couldn't find taura at the time)
    // taura would write a record with no command.
    if event == "finish" && !recorded(t) {
        notify("start", t, setting, &[], log);
    }
    let env = if event == "finish" { env_from_log(t.log_file.as_deref()) } else { None };
    let mut env_file = None;
    if let Some(env) = env {
        let Ok(mut f) = tempfile::Builder::new().prefix("taura-env-").suffix(".json").tempfile() else { return false };
        if std::io::Write::write_all(&mut f, py::json_dumps(&env).as_bytes()).is_err() {
            return false;
        }
        args.extend(["--env-json".into(), f.path().display().to_string()]);
        env_file = Some(f); // removed when dropped, after taura has run
    }
    let fail = |e: &dyn std::fmt::Display| {
        log(&format!("Task {}: taura record ({event}) failed: {e}", t.id));
        false
    };
    // Output goes to files, not pipes, so a chatty taura can't fill a pipe and stall.
    let (Ok(mut out), Ok(mut err)) = (tempfile::tempfile(), tempfile::tempfile()) else { return false };
    let child = Command::new(&exe)
        .args(&args)
        .stdin(std::process::Stdio::null())
        .stdout(out.try_clone().map_or(std::process::Stdio::null(), Into::into))
        .stderr(err.try_clone().map_or(std::process::Stdio::null(), Into::into))
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => return fail(&e),
    };
    let status = match child.wait_timeout(RECORD_TIMEOUT) {
        Ok(Some(s)) => s,
        Ok(None) => {
            let _ = child.kill();
            let _ = child.wait();
            return fail(&format!("Command '{}' timed out after {} seconds", exe.display(), RECORD_TIMEOUT.as_secs()));
        }
        Err(e) => return fail(&e),
    };
    drop(env_file);
    if !status.success() {
        let read = |f: &mut std::fs::File| {
            use std::io::{Read, Seek};
            let mut s = String::new();
            let _ = f.rewind().and_then(|_| f.read_to_string(&mut s));
            s
        };
        let stderr = read(&mut err);
        let text = if stderr.is_empty() { read(&mut out) } else { stderr };
        let last: Vec<String> =
            py::splitlines(py::strip(&text)).last().map(|l| vec![l.to_string()]).unwrap_or_default();
        let shown = if last.is_empty() { String::new() } else { py::repr_list(&last) };
        log(&format!("Task {}: taura record ({event}) exited {}: {shown}", t.id, status.code().unwrap_or(-1)));
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn git_repo(dir: &Path, ledger: bool) -> String {
        std::fs::create_dir_all(dir).unwrap();
        Command::new("git").args(["init", "-q"]).arg(dir).status().unwrap();
        if ledger {
            std::fs::create_dir(dir.join("ledger")).unwrap();
        }
        dir.canonicalize().unwrap().display().to_string()
    }

    fn row(repo: &str) -> Task {
        Task {
            id: 12,
            command: "python3 train.py --seed 1".into(),
            submitted_at: 1760839200.0,
            started_at: Some(1760839300.0),
            working_dir: repo.into(),
            repo_root: Some(repo.into()),
            server: Some("cuda01".into()),
            gpu_indices: Some("0,1".into()),
            git_commit: Some("a".repeat(40)),
            git_dirty: Some(0),
            status: "running".into(),
            ..Default::default()
        }
    }

    #[test]
    fn record_args_like_python() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = git_repo(&tmp.path().join("proj"), true);
        std::fs::create_dir(Path::new(&repo).join("sub")).unwrap();
        let t = Task { working_dir: format!("{repo}/sub"), ..row(&repo) };
        let start = record_args("start", &t, &[]).unwrap();
        assert_eq!(start[..2], ["-C".to_string(), repo.clone()]);
        let pos = |args: &[String], flag: &str| args.iter().position(|a| a == flag).unwrap();
        let i = pos(&start, "--cwd");
        assert_eq!(start[i + 1], "sub");
        assert_eq!(start[pos(&start, "--started") + 1], "1760839300.0");
        let dirty =
            Task { git_dirty: Some(2), git_dirty_paths: Some("results/a.json\nfigs/b.png".into()), ..t.clone() };
        let d = record_args("start", &dirty, &[]).unwrap();
        let i = pos(&d, "--dirty-count");
        assert_eq!(
            d[i..i + 5],
            ["--dirty-count", "2", "--dirty-paths", "results/a.json", "figs/b.png"].map(String::from)
        );
        let out = format!("{repo}/results/o.txt");
        std::fs::create_dir(format!("{repo}/results")).unwrap();
        std::fs::write(&out, "x").unwrap();
        let fetch = record_args("fetch", &t, &[out]).unwrap();
        assert_eq!(fetch[pos(&fetch, "--outputs") + 1..], ["results/o.txt".to_string()]);
        assert!(record_args("fetch", &t, &[]).is_none());
        let none = record_args("start", &Task { git_commit: None, ..t.clone() }, &[]).unwrap();
        assert_eq!(none[pos(&none, "--commit") + 1], "unknown");
        let plain = git_repo(&tmp.path().join("plain"), false);
        assert!(record_args("start", &row(&plain), &[]).is_none());
    }

    #[test]
    fn a_grid_task_says_which_machine_ran_it() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = git_repo(&tmp.path().join("proj"), true);
        let t = Task {
            server: Some("grid".into()),
            gpu_indices: Some(String::new()),
            machine: Some("big-thumb".into()),
            finished_at: Some(1760839500.0),
            exit_code: Some(0),
            sge_suspended: Some(60.0),
            ..row(&repo)
        };
        let value = |args: &[String], flag: &str| args.iter().position(|a| a == flag).map(|i| args[i + 1].clone());
        let start = record_args("start", &t, &[]).unwrap();
        assert_eq!(value(&start, "--host").unwrap(), "big-thumb");
        assert_eq!(value(&start, "--gpus").unwrap(), "");
        let finish = record_args("finish", &t, &[]).unwrap();
        assert_eq!(value(&finish, "--host").unwrap(), "big-thumb");
        assert_eq!(value(&finish, "--started").unwrap(), "1760839300.0");
        assert_eq!(
            value(&finish, "--note").unwrap(),
            "SGE suspended it for 1m 00s between started and finished, so it ran for 2m 20s."
        );

        let queue = Task { status: "lost".into(), started_at: None, machine: None, sge_suspended: None, ..t.clone() };
        let start = record_args("start", &queue, &[]).unwrap();
        assert_eq!(value(&start, "--host").unwrap(), "grid");
        assert_eq!(value(&start, "--started").unwrap(), "1760839500.0");
        assert!(value(&record_args("finish", &queue, &[]).unwrap(), "--note").unwrap().contains("never ran"));
        let gpu = record_args("finish", &row(&repo), &[]).unwrap();
        assert_eq!(value(&gpu, "--note"), None);
    }

    #[test]
    fn env_from_the_last_marker_line() {
        let tmp = tempfile::tempdir().unwrap();
        let log = tmp.path().join("1.log");
        std::fs::write(&log, "x\nTAURA_ENV {\"python\": \"3.11\"}\nTAURA_ENV {\"python\": \"3.12\"}\nmore\n").unwrap();
        assert_eq!(env_from_log(log.to_str()).unwrap()["python"], "3.12");
        assert!(env_from_log(Some("/nonexistent")).is_none());
        std::fs::write(&log, "TAURA_ENV not json\n").unwrap();
        assert!(env_from_log(log.to_str()).is_none());
    }
}
