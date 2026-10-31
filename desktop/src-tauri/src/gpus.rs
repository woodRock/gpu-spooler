//! The GPU heatmap and hosts views: the daemon's snapshot (gpu_snapshot.json) joined with the
//! config and our running tasks, as `task tui`'s GPU tab does. The daemon only keeps the snapshot
//! fresh while someone asks for it (gpu_snapshot.request), so the window asks while it shows GPUs.

use crate::state::AppState;
use gpu_spooler::config::{Config, Host, Kind, Role};
use gpu_spooler::db::Task;
use gpu_spooler::gpu::{GpuProcess, ServerResult};
use gpu_spooler::idle::Idle;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, specta::Type)]
pub struct GpuCell {
    pub index: i64,
    pub name: String,
    pub mem_used_mb: i64,
    pub mem_total_mb: i64,
    /// -1 if unknown
    pub utilization_pct: i64,
    /// free | ours | used
    pub state: String,
    /// Ours: the task on it.
    pub task_id: Option<i64>,
    pub processes: Vec<GpuProcess>,
    /// Held but idle for this long, from gpu-detective ([detective] in hosts.toml).
    pub idle_seconds: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, specta::Type)]
pub struct HostView {
    pub name: String,
    pub group: Option<String>,
    /// worker | jump
    pub role: String,
    /// local | direct | the jump chain's name
    pub route: String,
    pub enabled: bool,
    /// In hosts.toml; false: only in the snapshot.
    pub configured: bool,
    pub cpu_slots: i64,
    pub remote_root: Option<String>,
    /// As configured (display only; the snapshot's GPU names win).
    pub gpu_type: Option<String>,
    /// None: not in the snapshot.
    pub reachable: Option<bool>,
    pub error: String,
    pub gpus: Vec<GpuCell>,
}

#[derive(Clone, Debug, PartialEq, Serialize, specta::Type, tauri_specta::Event)]
pub struct GpuState {
    /// When the daemon took the snapshot; None: there is none yet.
    pub ts: Option<f64>,
    /// The network mode it was taken in (direct | jump).
    pub mode: Option<String>,
    /// Configured hosts in config order, then hosts only the snapshot knows, by name.
    pub hosts: Vec<HostView>,
}

fn route(h: &Host) -> String {
    match (h.kind, &h.jump) {
        (Kind::Local, _) => "local".into(),
        (_, Some(chain)) => chain.clone(),
        (_, None) => "direct".into(),
    }
}

fn cells(r: &ServerResult, ours: &HashMap<(String, i64), i64>, idle: &Idle) -> Vec<GpuCell> {
    r.gpus
        .iter()
        .map(|g| {
            let task_id = ours.get(&(r.name.clone(), g.index)).copied();
            let state = match task_id {
                Some(_) => "ours",
                None if g.is_free() => "free",
                None => "used",
            };
            GpuCell {
                index: g.index,
                name: g.name.clone(),
                mem_used_mb: g.mem_used_mb,
                mem_total_mb: g.mem_total_mb,
                utilization_pct: g.utilization_pct,
                state: state.into(),
                task_id,
                processes: g.processes.clone(),
                idle_seconds: idle.get(&(r.name.clone(), g.index)).map(|i| i.idle_seconds),
            }
        })
        .collect()
}

/// Join the snapshot with the config and our running tasks.
pub fn build(snapshot: Option<&Value>, cfg: Option<&Config>, running: &[Task], idle: &Idle) -> GpuState {
    let mut ours: HashMap<(String, i64), i64> = HashMap::new();
    for t in running {
        for idx in t.gpu_indices.as_deref().unwrap_or("").split(',') {
            if let Ok(i) = idx.trim().parse() {
                ours.insert((t.server.clone().unwrap_or_default(), i), t.id);
            }
        }
    }
    let mut results: HashMap<String, ServerResult> = snapshot
        .and_then(|s| s.get("hosts"))
        .and_then(Value::as_object)
        .map(|h| h.iter().filter_map(|(n, r)| Some((n.clone(), serde_json::from_value(r.clone()).ok()?))).collect())
        .unwrap_or_default();

    let mut hosts: Vec<HostView> = vec![];
    for h in cfg.map(|c| c.hosts.values().collect::<Vec<_>>()).unwrap_or_default() {
        let r = results.remove(&h.name);
        hosts.push(HostView {
            name: h.name.clone(),
            group: h.group.clone(),
            role: if h.role == Role::Jump { "jump" } else { "worker" }.into(),
            route: route(h),
            enabled: h.enabled,
            configured: true,
            cpu_slots: h.cpu_slots,
            remote_root: h.remote_root.clone(),
            gpu_type: h.gpu_type.clone(),
            reachable: r.as_ref().map(|r| r.reachable),
            error: r.as_ref().map_or(String::new(), |r| r.error.clone()),
            gpus: r.as_ref().map_or(vec![], |r| cells(r, &ours, idle)),
        });
    }
    let mut extra: Vec<ServerResult> = results.into_values().collect();
    extra.sort_by(|a, b| a.name.cmp(&b.name));
    for r in extra {
        hosts.push(HostView {
            name: r.name.clone(),
            group: None,
            role: "worker".into(),
            route: "-".into(),
            enabled: true,
            configured: false,
            cpu_slots: 0,
            remote_root: None,
            gpu_type: None,
            reachable: Some(r.reachable),
            error: r.error.clone(),
            gpus: cells(&r, &ours, idle),
        });
    }
    GpuState {
        ts: snapshot.and_then(|s| s.get("ts")).and_then(Value::as_f64),
        mode: snapshot.and_then(|s| s.get("mode")).and_then(Value::as_str).map(String::from),
        hosts,
    }
}

/// GPUs held but idle, when [detective] is configured; otherwise none.
pub fn idle(cfg: Option<&Config>) -> Idle {
    let Some(cfg) = cfg else { return Idle::new() };
    let Some(history) = cfg.detective_history.as_ref() else { return Idle::new() };
    gpu_spooler::idle::idle_gpus(
        std::path::Path::new(history),
        cfg.idle_minutes as f64 * 60.0,
        gpu_spooler::transport::now(),
    )
}

/// Ask the daemon to keep the snapshot fresh for the next couple of minutes.
#[tauri::command]
#[specta::specta]
pub fn request_gpu_snapshot(state: tauri::State<'_, AppState>) -> Result<(), String> {
    gpu_spooler::core::request_gpu_snapshot(&state.paths).map_err(|e| e.to_string())
}

/// Open Terminal on `task connect`, to log in to the jump hosts (a password or OTP prompt needs a
/// terminal). Its login shell finds `task` on the user's PATH.
#[tauri::command]
#[specta::specta]
pub fn open_task_connect() -> Result<(), String> {
    let status = std::process::Command::new("osascript")
        .args(["-e", r#"tell application "Terminal" to do script "task connect""#])
        .args(["-e", r#"tell application "Terminal" to activate"#])
        .status()
        .map_err(|e| e.to_string())?;
    status.success().then_some(()).ok_or_else(|| format!("osascript exited with {status}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg(toml: &str) -> Config {
        let home = tempfile::tempdir().unwrap();
        let paths = gpu_spooler::paths::Paths::for_home(home.path());
        std::fs::create_dir_all(&paths.state).unwrap();
        std::fs::write(&paths.config, toml).unwrap();
        gpu_spooler::config::load_config(&paths).unwrap()
    }

    fn gpu(index: i64, used_mb: i64, users: &[&str]) -> Value {
        let procs: Vec<Value> =
            users.iter().map(|u| json!({"pid": 1, "user": u, "mem_used_mb": used_mb, "command": "python3"})).collect();
        json!({"index": index, "name": "NVIDIA A40", "mem_used_mb": used_mb, "mem_free_mb": 1000,
               "mem_total_mb": 46068, "utilization_pct": 50, "processes": procs})
    }

    #[test]
    fn joins_config_snapshot_and_our_tasks() {
        let cfg = cfg(r#"
[jumps.ecs]
chain = ["me@entry.example"]
[groups.ecs]
jump = "ecs"
hosts = ["b", "a"]
[hosts.local]
kind = "local"
cpu_slots = 2
"#);
        let snapshot = json!({"ts": 100.0, "mode": "jump", "hosts": {
            "a": {"name": "a", "reachable": true, "error": "", "gpus": [gpu(0, 0, &[]), gpu(1, 9000, &["me"]), gpu(2, 9000, &["them"])]},
            "b": {"name": "b", "reachable": false, "error": "ssh exited 255", "gpus": []},
            "zz": {"name": "zz", "reachable": true, "error": "", "gpus": []}
        }});
        let running = [Task {
            id: 12,
            status: "running".into(),
            server: Some("a".into()),
            gpu_indices: Some("1".into()),
            ..Default::default()
        }];
        let s = build(Some(&snapshot), Some(&cfg), &running, &Idle::new());

        assert_eq!((s.ts, s.mode.as_deref()), (Some(100.0), Some("jump")));
        let names: Vec<&str> = s.hosts.iter().map(|h| h.name.as_str()).collect();
        // The chain's hop is a configured host too, as a jump host.
        assert_eq!(names, ["b", "a", "local", "ts-entry", "zz"]);
        assert_eq!((s.hosts[3].role.as_str(), s.hosts[3].reachable), ("jump", None));
        let a = &s.hosts[1];
        assert_eq!((a.route.as_str(), a.group.as_deref(), a.reachable), ("ecs", Some("ecs"), Some(true)));
        let states: Vec<(&str, Option<i64>)> = a.gpus.iter().map(|g| (g.state.as_str(), g.task_id)).collect();
        assert_eq!(states, [("free", None), ("ours", Some(12)), ("used", None)]);
        assert_eq!((s.hosts[0].reachable, s.hosts[0].error.as_str()), (Some(false), "ssh exited 255"));
        assert_eq!((s.hosts[2].route.as_str(), s.hosts[2].cpu_slots, s.hosts[2].reachable), ("local", 2, None));
        assert!(!s.hosts[4].configured);
    }

    #[test]
    fn no_snapshot_still_lists_the_configured_hosts() {
        let s = build(None, Some(&cfg("[hosts.local]\nkind = \"local\"\n")), &[], &Idle::new());
        assert_eq!((s.ts, s.hosts.len(), s.hosts[0].reachable), (None, 1, None));
    }
}
