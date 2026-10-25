//! Host configuration.
//!
//! Hosts are read from ~/.gpu-spooler/hosts.toml (override with the GPU_SPOOLER_CONFIG environment
//! variable). When the file is absent a built-in default reproducing the original hardcoded ECS
//! server list is used. Parsing follows config.py rule for rule, error messages included: group
//! defaults first (the first group a host is listed in wins), then `[hosts.X]`; raw `user@host:port`
//! jump specs become synthetic `ts-*` hosts.

use crate::paths::{expand_user, Paths};
use crate::py;
use indexmap::IndexMap;
use std::fmt;
use std::path::{Path, PathBuf};
use toml::{Table, Value};

// ── Built-in server catalogue (the original hardcoded list) ───────────────────

/// gpu_type is the name nvidia-smi reports, without "NVIDIA " (checked 2026-10-17 on every host
/// but cuda00–04, which were unreachable; `task gpus` warns when a host reports something else, and
/// live values win wherever they are known).
///
/// name, gpu_type, num_gpus, gpu_mem_gb, cpu_cores, sys_mem_gb
pub const DEFAULT_SERVER_INFO: &[(&str, &str, i64, i64, i64, i64)] = &[
    ("cuda-small0", "RTX A4000", 4, 16, 16, 192),
    ("cuda-small1", "RTX 4000 Ada Generation", 3, 20, 16, 144),
    ("cuda00", "RTX 6000", 2, 24, 32, 96),
    ("cuda01", "RTX A5000", 2, 24, 16, 64),
    ("cuda02", "RTX A5000", 2, 24, 16, 64),
    ("cuda03", "RTX A5000", 2, 24, 16, 64),
    ("cuda04", "RTX A5000", 2, 24, 16, 64),
    ("cuda05", "RTX PRO 5000 Blackwell", 2, 48, 24, 144),
    ("cuda06", "RTX PRO 5000 Blackwell", 2, 48, 16, 144),
    ("cuda07", "RTX A5000", 3, 24, 24, 144),
    ("cuda08", "RTX A5000", 3, 24, 24, 144),
    ("cuda09", "RTX PRO 4000 Blackwell", 3, 24, 32, 96),
    ("cuda10", "RTX A5000", 3, 24, 24, 128),
    ("cuda11", "RTX A5000", 3, 24, 24, 112),
    ("cuda12", "RTX A5000", 3, 24, 32, 96),
    ("cuda13", "RTX A5000", 3, 24, 32, 96),
    ("cuda14", "A40", 4, 48, 64, 256),
    ("cuda15", "A40", 4, 48, 64, 256),
    ("cuda16", "RTX A6000", 3, 48, 32, 192),
    ("cuda17", "Quadro RTX 6000", 3, 24, 24, 144),
    ("cuda18", "RTX A6000", 3, 48, 32, 192),
    ("cuda19", "L4", 4, 24, 64, 256),
    ("cuda20", "L40S", 4, 48, 64, 384),
    ("cuda21", "RTX 6000 Ada Generation", 2, 48, 48, 256),
    ("cuda22", "RTX 6000 Ada Generation", 2, 48, 48, 256),
    ("gryphon", "RTX A6000", 1, 48, 64, 192),
    ("red-tomatoes", "RTX A6000", 2, 48, 32, 96),
    ("piccolo", "RTX A5000", 2, 24, 20, 128),
    ("the-villa", "RTX A6000", 1, 48, 64, 192),
    ("bordeaux", "Quadro RTX 6000", 1, 24, 20, 64),
];

const INFO_FIELDS: [&str; 5] = ["gpu_type", "num_gpus", "gpu_mem_gb", "cpu_cores", "sys_mem_gb"];

const HOST_KEYS: [&str; 17] = [
    "kind",
    "hostname",
    "user",
    "port",
    "jump",
    "role",
    "tags",
    "path_map",
    "cpu_slots",
    "enabled",
    "domain",
    "remote_root",
    "gpu_type",
    "num_gpus",
    "gpu_mem_gb",
    "cpu_cores",
    "sys_mem_gb",
];

/// The names of the built-in servers, in order.
pub fn default_servers() -> impl Iterator<Item = &'static str> {
    DEFAULT_SERVER_INFO.iter().map(|s| s.0)
}

// ── Model ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ConfigError(pub String);

fn err<T>(msg: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError(msg.into()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Ssh,
    Local,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Worker,
    Jump,
}

/// `auto | direct | jump`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetMode {
    Auto,
    Direct,
    Jump,
}

impl NetMode {
    pub fn as_str(self) -> &'static str {
        match self {
            NetMode::Auto => "auto",
            NetMode::Direct => "direct",
            NetMode::Jump => "jump",
        }
    }
}

/// A number from hosts.toml that is shown as written: `48` stays `48`, `23.5` stays `23.5`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Num {
    Int(i64),
    Float(f64),
}

impl Num {
    pub fn as_f64(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(f) => f,
        }
    }
}

impl fmt::Display for Num {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Num::Int(i) => write!(f, "{i}"),
            Num::Float(x) => f.write_str(&py::repr_float(*x)),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Host {
    pub name: String,
    pub kind: Kind,
    /// None → ssh resolves the alias itself
    pub hostname: Option<String>,
    pub user: Option<String>,
    pub port: Option<u16>,
    /// name of a [jumps.X] chain
    pub jump: Option<String>,
    pub role: Role,
    pub tags: Vec<String>,
    /// first [groups.X] the host is listed in
    pub group: Option<String>,
    /// (local, remote) prefixes, local already `~`-expanded
    pub path_map: Vec<(String, String)>,
    pub cpu_slots: i64,
    /// git repos live at <remote_root>/<repo name>
    pub remote_root: Option<String>,
    pub enabled: bool,
    pub gpu_type: Option<String>,
    pub num_gpus: Option<i64>,
    pub gpu_mem_gb: Option<Num>,
    pub cpu_cores: Option<i64>,
    pub sys_mem_gb: Option<Num>,
}

impl Host {
    fn new(name: &str) -> Host {
        Host {
            name: name.into(),
            kind: Kind::Ssh,
            hostname: None,
            user: None,
            port: None,
            jump: None,
            role: Role::Worker,
            tags: vec![],
            group: None,
            path_map: vec![],
            cpu_slots: 0,
            remote_root: None,
            enabled: true,
            gpu_type: None,
            num_gpus: None,
            gpu_mem_gb: None,
            cpu_cores: None,
            sys_mem_gb: None,
        }
    }

    /// Name used as the `Host` alias in the generated ssh config.
    pub fn alias(&self) -> &str {
        &self.name
    }

    pub fn is_local(&self) -> bool {
        self.kind == Kind::Local
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub hosts: IndexMap<String, Host>,
    /// chain name → ordered hop host names
    pub jumps: IndexMap<String, Vec<String>>,
    pub mode: NetMode,
    pub probe: Option<String>,
    pub identity_file: Option<String>,
    pub control_persist: String,
    pub jump_control_persist: String,
    pub extra_options: Vec<String>,
    pub source: Option<PathBuf>,
    /// modification time of `source` (seconds since the epoch), to notice edits
    pub mtime: f64,
    /// true when no hosts.toml exists
    pub builtin: bool,
    /// [limits] max_gpus: GPUs our tasks may hold at once
    pub max_gpus: Option<i64>,
    /// [provenance] taura: "auto" | "off"
    pub taura: String,
    /// [detective]: gpu-detective's history.db, unexpanded; None = off
    pub detective_history: Option<String>,
    /// [detective] idle_minutes: flag GPUs idle this long
    pub idle_minutes: i64,
    home: PathBuf,
}

impl Config {
    fn empty(home: &Path) -> Config {
        Config {
            hosts: IndexMap::new(),
            jumps: IndexMap::new(),
            mode: NetMode::Auto,
            probe: None,
            identity_file: None,
            control_persist: "10m".into(),
            jump_control_persist: "12h".into(),
            extra_options: vec![],
            source: None,
            mtime: 0.0,
            builtin: false,
            max_gpus: None,
            taura: "auto".into(),
            detective_history: None,
            idle_minutes: 30,
            home: home.to_path_buf(),
        }
    }

    pub fn source_label(&self) -> String {
        if self.builtin {
            "built-in default".into()
        } else {
            self.source.as_ref().map_or("None".into(), |p| p.display().to_string())
        }
    }

    /// gpu-detective's history database, when [detective] is enabled.
    pub fn detective_path(&self) -> Option<PathBuf> {
        self.detective_history.as_ref().map(|h| PathBuf::from(expand_user(h, &self.home)))
    }

    pub fn workers(&self) -> Vec<&Host> {
        self.hosts.values().filter(|h| h.enabled && h.role == Role::Worker).collect()
    }

    pub fn get(&self, name: &str) -> Result<&Host, ConfigError> {
        self.hosts.get(name).ok_or_else(|| ConfigError(format!("unknown host '{name}'")))
    }

    /// Resolve a comma-separated list of host names, group names or tags to enabled worker hosts
    /// (in config order). Empty/None selects all workers.
    pub fn select(&self, spec: Option<&str>) -> Result<Vec<&Host>, ConfigError> {
        let workers = self.workers();
        let Some(spec) = spec.filter(|s| !s.is_empty()) else { return Ok(workers) };
        let mut wanted = std::collections::HashSet::new();
        for item in spec.split(',').map(py::strip) {
            if item.is_empty() {
                continue;
            }
            let matched: Vec<&str> = workers
                .iter()
                .filter(|h| h.name == item || h.tags.iter().any(|t| t == item))
                .map(|h| h.name.as_str())
                .collect();
            if matched.is_empty() {
                return err(format!("'{item}' matches no enabled host, group or tag"));
            }
            wanted.extend(matched);
        }
        Ok(workers.into_iter().filter(|h| wanted.contains(h.name.as_str())).collect())
    }

    /// Ordered jump hops for a host (empty if it has no jump).
    pub fn chain(&self, host: &Host) -> Vec<&Host> {
        match &host.jump {
            None => vec![],
            Some(j) => self.jumps[j].iter().map(|n| &self.hosts[n]).collect(),
        }
    }

    fn path_match(&self, host: &Host, local_path: &str) -> Option<(String, String)> {
        let mut best: Option<(String, String)> = None;
        for (local, remote) in &host.path_map {
            // Also match the resolved form, e.g. /tmp vs /private/tmp on macOS.
            let stripped = local.trim_end_matches('/');
            let stripped = if stripped.is_empty() { "/".to_string() } else { stripped.to_string() };
            let resolved = realpath(local);
            let mut forms = vec![stripped];
            if !forms.contains(&resolved) {
                forms.push(resolved);
            }
            for l in forms {
                if (l == "/" || local_path == l || local_path.starts_with(&format!("{l}/")))
                    && best.as_ref().is_none_or(|b| l.len() > b.0.len())
                {
                    best = Some((l, remote.clone()));
                }
            }
        }
        best
    }

    /// Longest-prefix match of local_path against the host's path_map.
    pub fn map_path(&self, host: &Host, local_path: &str) -> String {
        let Some((l, remote)) = self.path_match(host, local_path) else { return local_path.to_string() };
        let rest = if l != "/" { &local_path[l.len()..] } else { local_path }.trim_matches('/');
        let remote = remote.trim_end_matches('/');
        if !rest.is_empty() {
            format!("{remote}/{rest}")
        } else if remote.is_empty() {
            "/".into()
        } else {
            remote.into()
        }
    }

    pub fn is_mapped(&self, host: &Host, local_path: &str) -> bool {
        self.path_match(host, local_path).is_some()
    }

    /// Where a job submitted from local_dir runs on host:
    ///   1. a matching path_map entry wins;
    ///   2. inside a git repo with remote_root set: <remote_root>/<repo name>/<path within the repo>;
    ///   3. otherwise local_dir unchanged (shared filesystem).
    pub fn remote_dir(&self, host: &Host, local_dir: &str, repo_root: Option<&str>) -> String {
        if host.is_local() {
            return local_dir.into();
        }
        let (Some(repo_root), Some(remote_root)) = (repo_root.filter(|r| !r.is_empty()), &host.remote_root) else {
            return self.map_path(host, local_dir);
        };
        if self.is_mapped(host, local_dir) {
            return self.map_path(host, local_dir);
        }
        let rel = relpath(&realpath(local_dir), &realpath(repo_root));
        let name = repo_root.trim_end_matches('/').rsplit('/').next().unwrap_or("");
        let base = format!("{remote_root}/{name}");
        if rel == "." {
            base
        } else {
            format!("{base}/{rel}")
        }
    }
}

// ── Paths, as os.path does them ───────────────────────────────────────────────

/// `os.path.normpath` (POSIX).
pub fn normpath(path: &str) -> String {
    if path.is_empty() {
        return ".".into();
    }
    let initial = if path.starts_with("//") && !path.starts_with("///") {
        2
    } else if path.starts_with('/') {
        1
    } else {
        0
    };
    let mut comps: Vec<&str> = vec![];
    for c in path.split('/') {
        if c.is_empty() || c == "." {
            continue;
        }
        if c != ".." || (initial == 0 && comps.is_empty()) || comps.last() == Some(&"..") {
            comps.push(c);
        } else if !comps.is_empty() {
            comps.pop();
        }
    }
    let joined = format!("{}{}", "/".repeat(initial), comps.join("/"));
    if joined.is_empty() {
        ".".into()
    } else {
        joined
    }
}

/// `os.path.abspath`: absolute (from the process's cwd) and normalised.
pub fn abspath(path: &str) -> String {
    if path.starts_with('/') {
        normpath(path)
    } else {
        let cwd = std::env::current_dir().map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|_| "/".into());
        normpath(&format!("{cwd}/{path}"))
    }
}

/// `os.path.realpath`: symlinks in the part of the path that exists resolved, the rest kept.
pub fn realpath(path: &str) -> String {
    let abs = abspath(path);
    let comps: Vec<&str> = abs.split('/').filter(|c| !c.is_empty()).collect();
    for i in (0..=comps.len()).rev() {
        let prefix = format!("/{}", comps[..i].join("/"));
        if let Ok(canon) = std::fs::canonicalize(&prefix) {
            let mut out = canon.to_string_lossy().into_owned();
            for c in &comps[i..] {
                if !out.ends_with('/') {
                    out.push('/');
                }
                out.push_str(c);
            }
            return out;
        }
    }
    abs
}

/// `os.path.relpath(path, start)`.
pub fn relpath(path: &str, start: &str) -> String {
    let p = abspath(path);
    let s = abspath(start);
    let pc: Vec<&str> = p.split('/').filter(|c| !c.is_empty()).collect();
    let sc: Vec<&str> = s.split('/').filter(|c| !c.is_empty()).collect();
    let common = pc.iter().zip(&sc).take_while(|(a, b)| a == b).count();
    let mut parts: Vec<&str> = vec![".."; sc.len() - common];
    parts.extend(&pc[common..]);
    if parts.is_empty() {
        ".".into()
    } else {
        parts.join("/")
    }
}

// ── Loading ───────────────────────────────────────────────────────────────────

pub fn builtin_config(home: &Path) -> Config {
    let mut cfg = Config::empty(home);
    for &(name, gpu_type, num_gpus, mem, cores, sysmem) in DEFAULT_SERVER_INFO {
        let mut h = Host::new(name);
        h.tags = vec!["ecs".into()];
        h.group = Some("ecs".into());
        h.gpu_type = Some(gpu_type.into());
        h.num_gpus = Some(num_gpus);
        h.gpu_mem_gb = Some(Num::Int(mem));
        h.cpu_cores = Some(cores);
        h.sys_mem_gb = Some(Num::Int(sysmem));
        cfg.hosts.insert(name.into(), h);
    }
    cfg.mode = NetMode::Direct;
    cfg.builtin = true;
    cfg
}

/// The config in use: `paths.config`, or the built-in default when that file doesn't exist.
pub fn load_config(paths: &Paths) -> Result<Config, ConfigError> {
    load_config_from(&paths.config, paths)
}

pub fn load_config_from(path: &Path, paths: &Paths) -> Result<Config, ConfigError> {
    if !path.exists() {
        return Ok(builtin_config(&paths.home));
    }
    let text = std::fs::read_to_string(path).map_err(|e| ConfigError(format!("{}: {e}", path.display())))?;
    let data: Table = text
        .parse()
        .map_err(|e: toml::de::Error| ConfigError(format!("{}: {}", path.display(), e.message().trim_end())))?;
    let mut cfg = parse_config(&data, paths)?;
    cfg.source = Some(path.to_path_buf());
    cfg.mtime = mtime(path);
    Ok(cfg)
}

/// Modification time in seconds since the epoch (0 if unknown).
pub fn mtime(path: &Path) -> f64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0.0, |d| d.as_secs_f64())
}

/// `'user@host[:port]'` → (user, host, port).
fn parse_spec(spec: &str) -> Result<(Option<String>, String, Option<u16>), ConfigError> {
    let (user, rest) = match spec.rsplit_once('@') {
        Some((u, r)) => (Some(u.to_string()), r),
        None => (None, spec),
    };
    let bad_port = || ConfigError(format!("bad port in jump host '{spec}'"));
    let (host, port) = if let Some(inner) = rest.strip_prefix('[') {
        // [ipv6]:port
        let (host, after) = inner.split_once(']').unwrap_or((inner, ""));
        let port = match after.strip_prefix(':') {
            Some(p) => Some(py::int(p).and_then(|p| u16::try_from(p).ok()).ok_or_else(bad_port)?),
            None => None,
        };
        (host.to_string(), port)
    } else if rest.matches(':').count() == 1 {
        let (h, p) = rest.split_once(':').unwrap();
        (h.to_string(), Some(py::int(p).and_then(|p| u16::try_from(p).ok()).ok_or_else(bad_port)?))
    } else {
        (rest.to_string(), None)
    };
    Ok((user, host, port))
}

fn spec_alias(hostname: &str, taken: &dyn Fn(&str) -> bool) -> String {
    let base = if hostname.parse::<std::net::IpAddr>().is_ok() {
        format!("ts-{}", hostname.replace(['.', ':'], "-"))
    } else {
        format!("ts-{}", hostname.split('.').next().unwrap_or(""))
    };
    let (mut alias, mut n) = (base.clone(), 2);
    while taken(&alias) {
        alias = format!("{base}-{n}");
        n += 1;
    }
    alias
}

/// One host's merged settings: group defaults, then its [hosts.X] table.
#[derive(Default)]
struct Settings {
    values: Table,
    tags: Vec<String>,
    group: Option<String>,
}

fn table<'a>(data: &'a Table, key: &str) -> Result<Option<&'a Table>, ConfigError> {
    match data.get(key) {
        None => Ok(None),
        Some(Value::Table(t)) => Ok(Some(t)),
        Some(_) => err(format!("{key} must be a table")),
    }
}

fn tables<'a>(data: &'a Table, key: &str) -> Result<Vec<(&'a String, &'a Table)>, ConfigError> {
    let mut out = vec![];
    for (name, v) in table(data, key)?.into_iter().flatten() {
        match v {
            Value::Table(t) => out.push((name, t)),
            _ => return err(format!("{key}.{name} must be a table")),
        }
    }
    Ok(out)
}

fn strings(v: Option<&Value>, what: &str) -> Result<Vec<String>, ConfigError> {
    match v {
        None => Ok(vec![]),
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| match x {
                Value::String(s) => Ok(s.clone()),
                _ => err(format!("{what} must be a list of strings")),
            })
            .collect(),
        Some(_) => err(format!("{what} must be a list of strings")),
    }
}

fn unknown_keys<'a>(keys: impl Iterator<Item = &'a String>, allowed: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = keys.filter(|k| !allowed.contains(&k.as_str())).cloned().collect();
    out.sort();
    out
}

/// A string setting; `""` counts as unset, as it is falsy in Python.
fn opt_string(s: &Table, key: &str, where_: &str) -> Result<Option<String>, ConfigError> {
    match s.get(key) {
        None => Ok(None),
        Some(Value::String(x)) if x.is_empty() => Ok(None),
        Some(Value::String(x)) => Ok(Some(x.clone())),
        Some(_) => err(format!("{where_}: {key} must be a string")),
    }
}

fn opt_int(s: &Table, key: &str, where_: &str) -> Result<Option<i64>, ConfigError> {
    match s.get(key) {
        None => Ok(None),
        Some(Value::Integer(i)) => Ok(Some(*i)),
        Some(Value::Float(f)) => Ok(Some(f.trunc() as i64)),
        Some(Value::Boolean(b)) => Ok(Some(*b as i64)),
        Some(Value::String(x)) if py::int(x).is_some() => Ok(py::int(x)),
        Some(_) => err(format!("{where_}: {key} must be a number")),
    }
}

fn opt_num(s: &Table, key: &str, where_: &str) -> Result<Option<Num>, ConfigError> {
    match s.get(key) {
        None => Ok(None),
        Some(Value::Integer(i)) => Ok(Some(Num::Int(*i))),
        Some(Value::Float(f)) => Ok(Some(Num::Float(*f))),
        Some(_) => err(format!("{where_}: {key} must be a number")),
    }
}

fn path_map(v: Option<&Value>, where_: &str, home: &Path) -> Result<Vec<(String, String)>, ConfigError> {
    let entries = match v {
        None => return Ok(vec![]),
        Some(Value::Array(a)) => a,
        Some(_) => return err(format!("{where_}: path_map entries need 'local' and 'remote'")),
    };
    let mut out = vec![];
    for entry in entries {
        let (Some(Value::String(local)), Some(Value::String(remote))) =
            (entry.as_table().and_then(|t| t.get("local")), entry.as_table().and_then(|t| t.get("remote")))
        else {
            return err(format!("{where_}: path_map entries need 'local' and 'remote'"));
        };
        out.push((expand_user(local, home), remote.clone()));
    }
    Ok(out)
}

fn positive_int(v: &Value, what: &str) -> Result<i64, ConfigError> {
    match v {
        Value::Integer(i) if *i >= 1 => Ok(*i),
        other => err(format!("{what} must be a positive integer (got {})", py::repr_value(other))),
    }
}

pub fn parse_config(data: &Table, paths: &Paths) -> Result<Config, ConfigError> {
    let empty = Table::new();
    let net = table(data, "network")?.unwrap_or(&empty);
    let ssh = table(data, "ssh")?.unwrap_or(&empty);
    let groups = tables(data, "groups")?;
    let host_tables = tables(data, "hosts")?;
    let limits = table(data, "limits")?.unwrap_or(&empty);
    let jump_tables = tables(data, "jumps")?;
    let detective = table(data, "detective")?;
    let taura = match table(data, "provenance")?.and_then(|p| p.get("taura")) {
        None => "auto".to_string(),
        Some(Value::String(t)) if t == "auto" || t == "off" => t.clone(),
        Some(t) => return err(format!("provenance.taura must be auto or off (got '{}')", py::str_value(t))),
    };

    let mode = match net.get("mode") {
        None => NetMode::Auto,
        Some(Value::String(m)) if m == "auto" => NetMode::Auto,
        Some(Value::String(m)) if m == "direct" => NetMode::Direct,
        Some(Value::String(m)) if m == "jump" => NetMode::Jump,
        Some(m) => return err(format!("network.mode must be auto, direct or jump (got '{}')", py::str_value(m))),
    };

    // 1. Collect settings per host name: group defaults first, then [hosts.X].
    let mut settings: IndexMap<String, Settings> = IndexMap::new();
    for (gname, g) in &groups {
        let unknown = unknown_keys(g.keys().filter(|k| *k != "hosts"), &HOST_KEYS);
        if !unknown.is_empty() {
            return err(format!("groups.{gname}: unknown key(s) {}", py::repr_list(&unknown)));
        }
        let group_tags = strings(g.get("tags"), &format!("groups.{gname}.tags"))?;
        for hname in strings(g.get("hosts"), &format!("groups.{gname}.hosts"))? {
            let s = settings.entry(hname).or_default();
            if s.group.is_none() {
                s.group = Some(gname.to_string()); // first group listed wins
            }
            for (k, v) in g.iter().filter(|(k, _)| *k != "hosts" && *k != "tags") {
                s.values.entry(k.clone()).or_insert_with(|| v.clone());
            }
            for t in std::iter::once(gname.to_string()).chain(group_tags.iter().cloned()) {
                if !s.tags.contains(&t) {
                    s.tags.push(t);
                }
            }
        }
    }

    for (hname, h) in &host_tables {
        let unknown = unknown_keys(h.keys(), &HOST_KEYS);
        if !unknown.is_empty() {
            return err(format!("hosts.{hname}: unknown key(s) {}", py::repr_list(&unknown)));
        }
        let s = settings.entry(hname.to_string()).or_default();
        for (k, v) in h.iter() {
            if k == "tags" {
                for t in strings(Some(v), &format!("hosts.{hname}.tags"))? {
                    if !s.tags.contains(&t) {
                        s.tags.push(t);
                    }
                }
            } else {
                s.values.insert(k.clone(), v.clone());
            }
        }
    }

    // 2. Jump chains: names refer to hosts; raw specs become synthetic hosts.
    let mut jumps: IndexMap<String, Vec<String>> = IndexMap::new();
    let mut synthetic: IndexMap<String, Host> = IndexMap::new();
    let mut spec_aliases: IndexMap<String, String> = IndexMap::new();
    for (jname, j) in &jump_tables {
        let chain = strings(j.get("chain"), &format!("jumps.{jname}.chain"))?;
        if chain.is_empty() {
            return err(format!("jumps.{jname}: chain is empty"));
        }
        let mut hops = vec![];
        for elem in chain {
            if let Some(s) = settings.get_mut(&elem) {
                if s.group.is_none() {
                    s.values.entry("role").or_insert_with(|| Value::String("jump".into()));
                }
                hops.push(elem);
            } else if let Some(alias) = spec_aliases.get(&elem) {
                hops.push(alias.clone());
            } else {
                let (user, hostname, port) = parse_spec(&elem)?;
                let alias = spec_alias(&hostname, &|a| settings.contains_key(a) || synthetic.contains_key(a));
                let mut h = Host::new(&alias);
                h.hostname = Some(hostname);
                h.user = user;
                h.port = port;
                h.role = Role::Jump;
                synthetic.insert(alias.clone(), h);
                spec_aliases.insert(elem, alias.clone());
                hops.push(alias);
            }
        }
        jumps.insert(jname.to_string(), hops);
    }

    // 3. Build Host objects.
    let mut hosts: IndexMap<String, Host> = IndexMap::new();
    for (hname, s) in &settings {
        let where_ = format!("hosts.{hname}");
        let v = &s.values;
        let mut h = Host::new(hname);
        let domain = opt_string(v, "domain", &where_)?;
        h.hostname = opt_string(v, "hostname", &where_)?.or_else(|| domain.map(|d| format!("{hname}.{d}")));
        h.kind = match v.get("kind") {
            None => Kind::Ssh,
            Some(Value::String(k)) if k == "ssh" => Kind::Ssh,
            Some(Value::String(k)) if k == "local" => Kind::Local,
            Some(_) => return err(format!("{where_}: kind must be ssh or local")),
        };
        let jump = opt_string(v, "jump", &where_)?;
        if let Some(j) = &jump {
            if !jumps.contains_key(j) {
                return err(format!("{where_}: unknown jump chain '{j}'"));
            }
        }
        h.role = match v.get("role") {
            None => Role::Worker,
            Some(Value::String(r)) if r == "worker" => Role::Worker,
            Some(Value::String(r)) if r == "jump" => Role::Jump,
            Some(_) => return err(format!("{where_}: role must be worker or jump")),
        };
        h.user = opt_string(v, "user", &where_)?;
        h.port = match opt_int(v, "port", &where_)? {
            None | Some(0) => None,
            Some(p) => Some(u16::try_from(p).map_err(|_| ConfigError(format!("{where_}: port {p} is out of range")))?),
        };
        h.jump = if h.kind == Kind::Local { None } else { jump };
        h.tags = s.tags.clone();
        h.group = s.group.clone();
        h.path_map = path_map(v.get("path_map"), &where_, &paths.home)?;
        h.remote_root = opt_string(v, "remote_root", &where_)?
            .map(|r| r.trim_end_matches('/').to_string())
            .filter(|r| !r.is_empty());
        h.cpu_slots = opt_int(v, "cpu_slots", &where_)?.unwrap_or(0);
        h.enabled = v.get("enabled").is_none_or(py::truthy);
        h.gpu_type = v.get("gpu_type").map(py::str_value);
        h.num_gpus = opt_int(v, "num_gpus", &where_)?;
        h.gpu_mem_gb = opt_num(v, "gpu_mem_gb", &where_)?;
        h.cpu_cores = opt_int(v, "cpu_cores", &where_)?;
        h.sys_mem_gb = opt_num(v, "sys_mem_gb", &where_)?;
        debug_assert!(INFO_FIELDS.iter().all(|f| HOST_KEYS.contains(f)));
        hosts.insert(hname.clone(), h);
    }
    hosts.extend(synthetic);

    // A hop must not itself route through the chain it belongs to.
    for (jname, hops) in &jumps {
        for hop in hops {
            if hosts[hop].jump.is_some() {
                return err(format!("jumps.{jname}: hop '{hop}' must not set its own jump"));
            }
        }
    }

    let unknown = unknown_keys(limits.keys(), &["max_gpus"]);
    if !unknown.is_empty() {
        return err(format!("limits: unknown key(s) {}", py::repr_list(&unknown)));
    }
    let max_gpus = match limits.get("max_gpus") {
        None => None,
        Some(v) => Some(positive_int(v, "limits.max_gpus")?),
    };

    let (mut history, mut idle_minutes) = (None, 30);
    if let Some(d) = detective {
        let unknown = unknown_keys(d.keys(), &["enabled", "history", "idle_minutes"]);
        if !unknown.is_empty() {
            return err(format!("detective: unknown key(s) {}", py::repr_list(&unknown)));
        }
        if let Some(v) = d.get("idle_minutes") {
            idle_minutes = positive_int(v, "detective.idle_minutes")?;
        }
        if d.get("enabled").is_none_or(py::truthy) {
            let configured = d.get("history").filter(|h| py::truthy(h)).map(py::str_value);
            history = Some(
                configured
                    .or_else(|| paths.detective_db.clone())
                    .unwrap_or_else(|| "~/.gpu-detective/history.db".into()),
            );
        }
    }

    let mut cfg = Config::empty(&paths.home);
    cfg.hosts = hosts;
    cfg.jumps = jumps;
    cfg.mode = mode;
    cfg.probe = net.get("probe").map(py::str_value);
    cfg.max_gpus = max_gpus;
    cfg.taura = taura;
    cfg.detective_history = history;
    cfg.idle_minutes = idle_minutes;
    cfg.identity_file = opt_string(ssh, "identity_file", "ssh")?;
    if let Some(p) = ssh.get("control_persist") {
        cfg.control_persist = py::str_value(p);
    }
    if let Some(p) = ssh.get("jump_control_persist") {
        cfg.jump_control_persist = py::str_value(p);
    }
    cfg.extra_options = strings(ssh.get("extra_options"), "ssh.extra_options")?;
    Ok(cfg)
}

// ── Template for `task config init` ───────────────────────────────────────────

const TEMPLATE: &str = include_str!("hosts_template.toml");

pub fn config_template(user: &str) -> String {
    let info_blocks: Vec<String> = DEFAULT_SERVER_INFO
        .iter()
        .map(|(name, gpu_type, num_gpus, mem, cores, sysmem)| {
            format!(
                "[hosts.{name}]\ngpu_type = \"{gpu_type}\"\nnum_gpus = {num_gpus}\ngpu_mem_gb = {mem}\n\
                 cpu_cores = {cores}\nsys_mem_gb = {sysmem}\n"
            )
        })
        .collect();
    let host_list: Vec<String> = default_servers().map(|n| format!("\"{n}\"")).collect();
    TEMPLATE
        .replace("{info_blocks}", &info_blocks.join("\n"))
        .replace("{host_list}", &host_list.join(",\n    "))
        .replace("{user}", user)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<Config, ConfigError> {
        parse_config(&text.parse().unwrap(), &Paths::for_home("/home/me"))
    }

    #[test]
    fn missing_file_gives_builtin() {
        let paths = Paths::for_home("/nonexistent-home");
        let cfg = load_config(&paths).unwrap();
        assert!(cfg.builtin && cfg.source.is_none());
        assert_eq!(cfg.mode, NetMode::Direct);
        let names: Vec<&str> = cfg.workers().iter().map(|h| h.name.as_str()).collect();
        assert_eq!(names, default_servers().collect::<Vec<_>>());
        assert_eq!(names.len(), 30);
        let h = cfg.get("cuda14").unwrap();
        assert_eq!(
            (h.hostname.as_deref(), h.user.as_deref(), h.jump.as_deref(), h.kind),
            (None, None, None, Kind::Ssh)
        );
        assert_eq!((h.gpu_type.as_deref(), h.num_gpus, h.gpu_mem_gb), (Some("A40"), Some(4), Some(Num::Int(48))));
        assert_eq!(cfg.source_label(), "built-in default");
    }

    #[test]
    fn loads_a_file_and_reports_toml_errors_with_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("h.toml");
        std::fs::write(&p, "[hosts.box]\nhostname = \"box.example\"\n").unwrap();
        let paths = Paths::for_home(dir.path());
        let cfg = load_config_from(&p, &paths).unwrap();
        assert_eq!(cfg.source.as_deref(), Some(p.as_path()));
        assert!(cfg.mtime > 0.0);
        assert_eq!(cfg.workers().len(), 1);
        assert_eq!(cfg.mode, NetMode::Auto);
        std::fs::write(&p, "[network\n").unwrap();
        let e = load_config_from(&p, &paths).unwrap_err();
        assert!(e.0.starts_with(&format!("{}: ", p.display())), "{e}");
    }

    #[test]
    fn template_parses_and_covers_all_servers() {
        let cfg = parse(&config_template("someone")).unwrap();
        assert_eq!(cfg.mode, NetMode::Auto);
        let h = cfg.get("cuda14").unwrap();
        assert_eq!(h.hostname.as_deref(), Some("cuda14.ecs.vuw.ac.nz"));
        assert_eq!(h.user.as_deref(), Some("someone"));
        let hops = cfg.chain(h);
        let names: Vec<&str> = hops.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(names, ["ts-entry", "ts-barretts"]);
        assert!(hops.iter().all(|x| x.role == Role::Jump && x.user.as_deref() == Some("someone")));
    }

    #[test]
    fn select_errors() {
        let cfg = parse("[hosts.a]\n[hosts.b]\nenabled = false\n").unwrap();
        assert_eq!(cfg.select(Some("b")).unwrap_err().0, "'b' matches no enabled host, group or tag");
        assert_eq!(cfg.select(None).unwrap().len(), 1);
    }

    #[test]
    fn detective_settings() {
        let off = parse("").unwrap();
        assert!(off.detective_path().is_none());
        let on = parse("[detective]\nidle_minutes = 45\n").unwrap();
        assert_eq!(on.detective_path().unwrap(), PathBuf::from("/home/me/.gpu-detective/history.db"));
        assert_eq!(on.idle_minutes, 45);
        assert!(parse("[detective]\nenabled = false\n").unwrap().detective_path().is_none());
        let mut paths = Paths::for_home("/home/me");
        paths.detective_db = Some("/elsewhere/h.db".into());
        let cfg = parse_config(&"[detective]\n".parse().unwrap(), &paths).unwrap();
        assert_eq!(cfg.detective_path().unwrap(), PathBuf::from("/elsewhere/h.db"));
    }

    #[test]
    fn os_path_helpers() {
        assert_eq!(normpath("/a/./b/../c/"), "/a/c");
        assert_eq!(normpath("//x"), "//x");
        assert_eq!(normpath("///x/.."), "/");
        assert_eq!(normpath("a/../.."), "..");
        assert_eq!(normpath(""), ".");
        assert_eq!(relpath("/a/b/c", "/a"), "b/c");
        assert_eq!(relpath("/a", "/a/b"), "..");
        assert_eq!(relpath("/a", "/a"), ".");
        assert_eq!(realpath("/nonexistent-root/x/../y"), "/nonexistent-root/y");
    }
}
