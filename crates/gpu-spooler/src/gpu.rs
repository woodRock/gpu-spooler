//! GPU querying shared by the daemon and `task gpus`.
//!
//! One script runs nvidia-smi on a host and prints sections delimited by sentinel lines;
//! [`parse_output`] turns that into [`GpuInfo`]s.

use crate::config::Host;
use crate::py;
use crate::transport::{auth_hint, Transport};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// A GPU using at least this much VRAM counts as busy.
pub const BUSY_MEM_MB: i64 = 256;
pub const QUERY_TIMEOUT: Duration = Duration::from_secs(25);

/// The script sent to each host (`bash -s`), byte for byte as the Python version sends it.
pub const REMOTE_CMD: &str = include_str!("remote/gpu_query.sh");

/// A GPU name as compared and shown: nvidia-smi's name without a leading 'NVIDIA '.
pub fn gpu_model(name: &str) -> String {
    let name = py::split(name).join(" ");
    if name.to_lowercase().starts_with("nvidia ") {
        name[7..].to_string()
    } else {
        name
    }
}

/// Whether a GPU called `name` (by nvidia-smi) satisfies --gpu-type `spec`: one of the
/// comma-separated models, compared exactly, ignoring case and a leading 'NVIDIA '. Exact, not
/// substring: 'RTX 6000' must not also pick the 'RTX 6000 Ada Generation'.
pub fn gpu_type_matches(spec: Option<&str>, name: &str) -> bool {
    let Some(spec) = spec.filter(|s| !s.is_empty()) else { return true };
    let model = gpu_model(name).to_lowercase();
    spec.split(',').filter(|t| !py::strip(t).is_empty()).any(|t| gpu_model(t).to_lowercase() == model)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuProcess {
    pub pid: i64,
    pub user: String,
    pub mem_used_mb: i64,
    /// truncated command/script name
    pub command: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: i64,
    pub name: String,
    pub mem_used_mb: i64,
    pub mem_free_mb: i64,
    pub mem_total_mb: i64,
    /// -1 if unknown
    pub utilization_pct: i64,
    #[serde(default)]
    pub processes: Vec<GpuProcess>,
}

impl GpuInfo {
    /// Free only if no process is listed AND VRAM usage is very low (zombie contexts can hold VRAM
    /// without showing up in pmon).
    pub fn is_free(&self) -> bool {
        self.processes.is_empty() && self.mem_used_mb < BUSY_MEM_MB
    }

    /// Decimal GB, as GPUs are marketed (an A40 reads ~48.3).
    pub fn mem_total_gb(&self) -> f64 {
        self.mem_total_mb as f64 * 1.048576 / 1000.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerResult {
    pub name: String,
    pub reachable: bool,
    #[serde(default)]
    pub error: String,
    #[serde(default)]
    pub gpus: Vec<GpuInfo>,
}

impl ServerResult {
    pub fn up(name: &str, gpus: Vec<GpuInfo>) -> ServerResult {
        ServerResult { name: name.into(), reachable: true, error: String::new(), gpus }
    }

    pub fn down(name: &str, error: impl Into<String>) -> ServerResult {
        ServerResult { name: name.into(), reachable: false, error: error.into(), gpus: vec![] }
    }

    pub fn total_gpus(&self) -> usize {
        self.gpus.len()
    }

    pub fn free_gpus(&self) -> usize {
        self.gpus.iter().filter(|g| g.is_free()).count()
    }

    pub fn busy_gpus(&self) -> usize {
        self.total_gpus() - self.free_gpus()
    }

    pub fn total_vram_mb(&self) -> i64 {
        self.gpus.iter().map(|g| g.mem_total_mb).sum()
    }

    pub fn free_vram_mb(&self) -> i64 {
        self.gpus.iter().map(|g| g.mem_free_mb).sum()
    }

    /// Distinct GPU models on this host, in index order.
    pub fn models(&self) -> Vec<String> {
        let mut out: Vec<String> = vec![];
        for g in &self.gpus {
            let m = gpu_model(&g.name);
            if !out.contains(&m) {
                out.push(m);
            }
        }
        out
    }
}

/// Parse the structured output from [`REMOTE_CMD`].
pub fn parse_output(raw: &str) -> Vec<GpuInfo> {
    const SENTINELS: [&str; 4] = ["GPU_INFO", "PMON_INFO", "APPS_INFO", "PROC_INFO"];
    let mut sections: [Vec<&str>; 4] = Default::default();
    let mut current: Option<usize> = None;
    for line in py::splitlines(raw) {
        let stripped = line.trim_end_matches('\r'); // tolerate \r\n from some SSH configs
        if let Some(i) = SENTINELS.iter().position(|s| *s == stripped) {
            current = Some(i);
        } else if stripped == "END" {
            break;
        } else if let Some(i) = current {
            if !py::strip(stripped).is_empty() {
                sections[i].push(stripped);
            }
        }
    }
    let [gpu_info, pmon_info, apps_info, proc_info] = sections;

    let mut gpus: IndexMap<i64, GpuInfo> = IndexMap::new();
    for line in gpu_info {
        let parts: Vec<&str> = line.split(',').map(py::strip).collect();
        if parts.len() < 6 {
            continue;
        }
        let parse = || -> Option<GpuInfo> {
            let index = py::int(parts[0])?;
            let util = if py::isdigit(parts[5].trim_start_matches('-')) { py::int(parts[5])? } else { -1 };
            Some(GpuInfo {
                index,
                name: parts[1].to_string(),
                mem_used_mb: py::int(parts[2])?,
                mem_free_mb: py::int(parts[3])?,
                mem_total_mb: py::int(parts[4])?,
                utilization_pct: util,
                processes: vec![],
            })
        };
        if let Some(g) = parse() {
            gpus.insert(g.index, g);
        }
    }

    // pid -> (gpu_idx, fb_mb, cmd_name). pmon's column set varies between driver versions (jpg/ofa
    // were added), so take fb and command from the end.
    let mut pid_to_gpu: IndexMap<i64, (i64, i64, String)> = IndexMap::new();
    for line in pmon_info {
        if py::strip(line).starts_with('#') {
            continue;
        }
        let parts = py::split(line);
        if parts.len() < 2 || !py::isdigit(parts[0]) || !py::isdigit(parts[1]) {
            continue;
        }
        let n = parts.len();
        let fb_mb = if n >= 4 && py::isdigit(parts[n - 2]) { py::int(parts[n - 2]).unwrap_or(0) } else { 0 };
        let cmd = if n >= 3 { parts[n - 1] } else { "unknown" };
        let (Some(gpu), Some(pid)) = (py::int(parts[0]), py::int(parts[1])) else { continue };
        pid_to_gpu.insert(pid, (gpu, fb_mb, cmd.to_string()));
    }

    for line in apps_info {
        let parts: Vec<&str> = line.split(',').map(py::strip).collect();
        if parts.len() >= 2 && py::isdigit(parts[0]) && py::isdigit(parts[1]) {
            if let (Some(gpu), Some(pid)) = (py::int(parts[0]), py::int(parts[1])) {
                pid_to_gpu.entry(pid).or_insert((gpu, 0, "unknown".into()));
            }
        }
    }

    let mut pid_user: std::collections::HashMap<i64, String> = std::collections::HashMap::new();
    for line in proc_info {
        let parts = py::split(line);
        if parts.len() >= 2 && py::isdigit(parts[0]) {
            if let Some(pid) = py::int(parts[0]) {
                pid_user.insert(pid, parts[1].to_string());
            }
        }
    }

    for (pid, (gpu_idx, fb_mb, cmd)) in pid_to_gpu {
        if let Some(g) = gpus.get_mut(&gpu_idx) {
            g.processes.push(GpuProcess {
                pid,
                user: pid_user.get(&pid).cloned().unwrap_or_else(|| "?".into()),
                mem_used_mb: fb_mb,
                command: py::head(&cmd, 60).to_string(),
            });
        }
    }

    let mut out: Vec<GpuInfo> = gpus.into_values().collect();
    out.sort_by_key(|g| g.index);
    out
}

/// Query one host. With `debug`, the raw output is printed.
pub fn query_host(transport: &Transport, host: &Host, debug: bool) -> ServerResult {
    let r = transport.run(host, REMOTE_CMD, QUERY_TIMEOUT);
    if debug {
        println!("\n--- RAW OUTPUT FROM {} ---", host.name);
        println!("{}", if r.stdout.is_empty() { "(empty stdout)" } else { &r.stdout });
        if !r.stderr.is_empty() {
            println!("STDERR: {}", py::strip(&r.stderr));
        }
        println!("--- END {} (exit={}) ---\n", host.name, r.rc);
    }
    if py::strip(&r.stdout) == "NO_NVIDIA_SMI" {
        // A local machine (or a configured CPU host) without GPUs is still usable.
        if host.is_local() || host.cpu_slots > 0 {
            return ServerResult::up(&host.name, vec![]);
        }
        return ServerResult::down(&host.name, "nvidia-smi not found");
    }
    if r.rc != 0 && r.stdout.is_empty() {
        let err = py::strip(&r.stderr);
        let msg = match py::splitlines(err).last() {
            Some(last) if !err.is_empty() => last.to_string(),
            _ => format!("ssh exited {}", r.rc),
        };
        return ServerResult::down(
            &host.name,
            match auth_hint(&r.stderr) {
                Some(hint) => format!("{msg} ({hint})"),
                None => msg,
            },
        );
    }
    if let Some(errline) = py::splitlines(&r.stdout).into_iter().find(|l| l.starts_with("ERROR:")) {
        return ServerResult::down(&host.name, errline);
    }
    let gpus = parse_output(&r.stdout);
    if gpus.is_empty() && !r.stdout.contains("GPU_INFO") {
        return ServerResult::down(&host.name, "no nvidia-smi output");
    }
    ServerResult::up(&host.name, gpus)
}

pub fn query_hosts(
    transport: &Transport,
    hosts: &[&Host],
    workers: usize,
    debug: bool,
) -> IndexMap<String, ServerResult> {
    transport.fan_out(hosts, |h| query_host(transport, h, debug), workers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_needs_no_processes_and_little_vram() {
        let g = GpuInfo {
            index: 0,
            name: "NVIDIA A40".into(),
            mem_used_mb: 255,
            mem_free_mb: 0,
            mem_total_mb: 46068,
            utilization_pct: 0,
            processes: vec![],
        };
        assert!(g.is_free());
        assert!(!GpuInfo { mem_used_mb: 256, ..g.clone() }.is_free());
        assert!((g.mem_total_gb() - 48.3).abs() < 0.05);
    }

    #[test]
    fn snapshot_json_reads_python_field_names() {
        let json = r#"{"name": "h", "reachable": true, "error": "", "ts": 1.5, "gpus": [{"index": 0,
            "name": "NVIDIA L4", "mem_used_mb": 1, "mem_free_mb": 2, "mem_total_mb": 3,
            "utilization_pct": -1, "processes": [{"pid": 7, "user": "u", "mem_used_mb": 0, "command": "x"}]}]}"#;
        let r: ServerResult = serde_json::from_str(json).unwrap();
        assert_eq!(r.models(), ["L4"]);
        assert_eq!(r.gpus[0].processes[0].pid, 7);
    }
}
