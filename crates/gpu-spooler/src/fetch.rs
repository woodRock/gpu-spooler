//! Copying outputs back from the servers: `task fetch PATH...`.
//!
//! Paths resolve the way jobs' working directories do (Config::remote_dir): from a laptop checkout
//! of a git repo, a relative PATH is the same path in the repo's copy under remote_root, or wherever
//! a path_map entry points. REMOTE:LOCAL names both sides explicitly, for anything outside the repo
//! copy. Directories are merged into the local directory of the same name.
//!
//! Safety rules:
//!   - a relative PATH must stay inside the repo (REMOTE:LOCAL for anything else);
//!   - symlinks are copied as links, never followed, and a link pointing outside the destination
//!     is skipped (the rules of Python's tarfile 'data' filter, see [`filter`]);
//!   - a local file is replaced only if it is unchanged since the last fetch wrote it, or tracked by
//!     git and unmodified (git still has it). Anything else is kept and reported, unless force;
//!   - fetch never deletes local files that are missing on the server.

use crate::config::{abspath, normpath, realpath, relpath, Config, Host};
use crate::db::Task;
use crate::paths::Paths;
use crate::py;
use crate::repo::find_repo;
use crate::transport::Transport;
use base64::Engine;
use indexmap::IndexMap;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read, Seek};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

pub const LIST_TIMEOUT: Duration = Duration::from_secs(300);
const STEP: &str = "TS_FETCH";
/// Runs on the server: describe each requested path (files with their SHA-256, links).
const LIST_PY: &str = include_str!("remote/fetch_list.py");
/// Runs on the server: a gzipped tar of the given members of base, links as links.
const TAR_PY: &str = include_str!("remote/fetch_tar.py");
/// Seconds around a task's run: clock skew and the daemon's 5 s polls.
pub const WRITE_SLACK: f64 = 120.0;

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct FetchError(pub String);

fn err<T>(msg: impl Into<String>) -> Result<T, FetchError> {
    Err(FetchError(msg.into()))
}

/// A remote Python program for `bash -s`, its input passed as base64-encoded JSON.
pub fn py_script(program: &str, payload: &Value) -> String {
    let arg = base64::engine::general_purpose::STANDARD.encode(py::json_dumps(payload));
    format!(
        "command -v python3 >/dev/null 2>&1 || {{ echo \"python3 not found\" >&2; exit 3; }}\npython3 - {arg} <<'TS_PY'\n{program}\nTS_PY\n"
    )
}

pub fn list_script(remotes: &[String]) -> String {
    py_script(LIST_PY, &json!(remotes))
}

pub fn tar_script(base: &str, members: &[String]) -> String {
    py_script(TAR_PY, &json!({"base": base, "members": members}))
}

// ── Resolving paths ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    /// as given
    pub spec: String,
    /// absolute local path
    pub local: String,
    /// path on the server
    pub remote: String,
}

#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub item: Option<Item>,
    /// file | dir | link | missing
    pub kind: String,
    /// relative to the item (or "" for a file)
    pub new: Vec<String>,
    pub update: Vec<String>,
    pub same: Vec<String>,
    /// local changes, not replaced
    pub kept: Vec<String>,
    /// a directory where a file would go
    pub conflict: Vec<String>,
    pub links: BTreeMap<String, String>,
}

impl Plan {
    pub fn item(&self) -> &Item {
        self.item.as_ref().expect("a plan has its item")
    }

    pub fn to_fetch(&self) -> Vec<String> {
        self.new.iter().chain(&self.update).cloned().collect()
    }

    /// Where a member of this plan lands locally.
    pub fn dest(&self, rel: &str) -> String {
        if self.kind == "dir" {
            join(&self.item().local, rel)
        } else {
            self.item().local.clone()
        }
    }
}

/// `os.path.join` for two parts.
pub fn join(a: &str, b: &str) -> String {
    if b.starts_with('/') {
        b.to_string()
    } else if a.is_empty() || a.ends_with('/') {
        format!("{a}{b}")
    } else {
        format!("{a}/{b}")
    }
}

/// `os.path.dirname`.
fn dirname(p: &str) -> &str {
    let Some(i) = p.rfind('/') else { return "" };
    let head = &p[..=i];
    if head.bytes().all(|b| b == b'/') {
        head
    } else {
        head.trim_end_matches('/')
    }
}

fn basename(p: &str) -> &str {
    p.rsplit('/').next().unwrap_or(p)
}

fn lexists(p: &str) -> bool {
    std::fs::symlink_metadata(p).is_ok()
}

fn is_link(p: &str) -> bool {
    std::fs::symlink_metadata(p).is_ok_and(|m| m.file_type().is_symlink())
}

/// `path` is `root` or inside it, after resolving symlinks.
pub fn inside(path: &str, root: &str) -> bool {
    let (path, root) = (realpath(path), realpath(root));
    path == root || path.starts_with(&format!("{}/", root.trim_end_matches('/')))
}

/// REMOTE:LOCAL, where REMOTE is absolute or ~-relative.
pub fn is_explicit(spec: &str) -> bool {
    spec.split_once(':').is_some_and(|(head, _)| head.starts_with('/') || head.starts_with('~'))
}

/// The hosts to fetch from (one storage group, tried in order) and each spec's local and remote
/// path. Errors when a path cannot be mapped.
pub fn resolve<'c>(
    cfg: &'c Config,
    paths: &Paths,
    hosts: &[&'c Host],
    specs: &[String],
    cwd: &str,
) -> Result<(Vec<&'c Host>, Vec<Item>), FetchError> {
    let hosts: Vec<&Host> = hosts.iter().copied().filter(|h| !h.is_local()).collect();
    if hosts.is_empty() {
        return err("no remote host selected");
    }
    let mut groups: IndexMap<(String, Option<String>), Vec<&Host>> = IndexMap::new();
    for h in hosts {
        groups.entry((h.group.clone().unwrap_or_else(|| h.name.clone()), h.remote_root.clone())).or_default().push(h);
    }
    if groups.len() > 1 {
        let mut names: Vec<&str> = groups.keys().map(|(g, _)| g.as_str()).collect();
        names.sort();
        return err(format!("the selected hosts span several storage groups ({}); pick one with -H", names.join(", ")));
    }
    let group = groups.into_values().next().expect("one group");
    let host = group[0];
    let local_repo = find_repo(cwd);

    let mut items = vec![];
    for spec in specs {
        if is_explicit(spec) {
            let (remote, local) = spec.split_once(':').expect("explicit specs have a colon");
            let local = abspath(&join(cwd, &paths.expand_user(if local.is_empty() { "." } else { local })));
            let remote = remote.trim_end_matches('/');
            items.push(Item {
                spec: spec.clone(),
                local,
                remote: if remote.is_empty() { "/".into() } else { remote.into() },
            });
            continue;
        }
        let local = abspath(&join(cwd, &paths.expand_user(spec)));
        if cfg.is_mapped(host, &local) {
            items.push(Item { spec: spec.clone(), remote: cfg.map_path(host, &local), local });
            continue;
        }
        let (Some(repo), Some(_)) = (&local_repo, host.remote_root.as_ref().filter(|r| !r.is_empty())) else {
            return err(format!(
                "{spec}: not inside a git repo with a remote copy and no path_map entry covers it; use REMOTE:LOCAL"
            ));
        };
        if !inside(&local, &repo.root) {
            return err(format!("{spec} is outside the repo {}; use REMOTE:LOCAL to fetch it", repo.root));
        }
        items.push(Item { spec: spec.clone(), remote: cfg.remote_dir(host, &local, Some(&repo.root)), local });
    }
    Ok((group, items))
}

// ── Comparing ────────────────────────────────────────────────────────────────

fn sha(path: &str) -> String {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    if let Ok(mut f) = std::fs::File::open(path) {
        let _ = std::io::copy(&mut f, &mut h);
    }
    format!("{:x}", h.finalize())
}

/// [size, mtime_ns] as `lstat` sees them: how fetch left a file.
pub fn stamp(path: &str) -> Option<[i64; 2]> {
    let m = std::fs::symlink_metadata(path).ok()?;
    Some([m.size() as i64, m.mtime() * 1_000_000_000 + m.mtime_nsec()])
}

/// ~/.gpu-spooler/fetched.json: local path → [size, mtime_ns] as fetch left it.
pub type Manifest = IndexMap<String, Value>;

pub fn load_manifest(paths: &Paths) -> Manifest {
    std::fs::read_to_string(paths.state.join("fetched.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| match v {
            Value::Object(m) => Some(m.into_iter().collect()),
            _ => None,
        })
        .unwrap_or_default()
}

pub fn save_manifest(paths: &Paths, m: &Manifest) -> std::io::Result<()> {
    std::fs::create_dir_all(&paths.state)?;
    let path = paths.state.join("fetched.json");
    let tmp = paths.state.join("fetched.json.tmp");
    let obj: serde_json::Map<String, Value> = m.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    std::fs::write(&tmp, py::json_dumps(&Value::Object(obj)))?;
    std::fs::rename(&tmp, path)
}

/// Those of `paths` that git tracks and that are unmodified (git can restore them).
pub fn git_clean_tracked(paths: &[String]) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut roots: HashMap<String, Option<String>> = HashMap::new(); // dir → repo root
    let mut by_root: IndexMap<String, Vec<&String>> = IndexMap::new();
    for p in paths {
        let dir = if Path::new(p).is_dir() { p.clone() } else { dirname(p).to_string() };
        let root = roots.entry(dir.clone()).or_insert_with(|| find_repo(&dir).map(|r| r.root)).clone();
        if let Some(root) = root {
            by_root.entry(root).or_default().push(p);
        }
    }
    for (root, ps) in by_root {
        let real_root = realpath(&root);
        let rels: Vec<String> = ps.iter().map(|p| relpath(&realpath(p), &real_root)).collect();
        let git = |args: &[&str]| -> HashSet<String> {
            let out = Command::new("git").arg("-C").arg(&root).args(args).arg("--").args(&rels).output();
            match out {
                Ok(o) if o.status.success() => {
                    String::from_utf8_lossy(&o.stdout).split('\0').filter(|x| !x.is_empty()).map(String::from).collect()
                }
                _ => HashSet::new(),
            }
        };
        let tracked = git(&["ls-files", "-z"]);
        let modified = git(&["diff", "--name-only", "-z", "HEAD"]);
        for (p, rel) in ps.iter().zip(&rels) {
            if tracked.contains(rel) && !modified.contains(rel) {
                out.insert((*p).clone());
                out.insert(join(&real_root, rel));
            }
        }
    }
    out
}

/// What to do with each file of one item, given the server's description of it.
pub fn plan_item(mut item: Item, desc: &Value, manifest: &Manifest, force: bool) -> Plan {
    let kind = desc["kind"].as_str().unwrap_or("missing").to_string();
    if kind == "missing" {
        return Plan { item: Some(item), kind, ..Default::default() };
    }
    if kind != "dir" && Path::new(&item.local).is_dir() && !is_link(&item.local) {
        item.local = join(&item.local, basename(&item.remote)); // like scp
    }
    // rel → (file | link, sha | target)
    let mut entries: BTreeMap<String, (&str, String)> = BTreeMap::new();
    if kind == "dir" {
        for (rel, s) in desc["files"].as_object().into_iter().flatten() {
            entries.insert(rel.clone(), ("file", s.as_str().unwrap_or("").into()));
        }
        for (rel, t) in desc["links"].as_object().into_iter().flatten() {
            entries.insert(rel.clone(), ("link", t.as_str().unwrap_or("").into()));
        }
    } else {
        let reference = desc["sha"].as_str().filter(|s| !s.is_empty()).or(desc["target"].as_str()).unwrap_or("");
        entries.insert(String::new(), (if kind == "link" { "link" } else { "file" }, reference.into()));
    }
    let mut plan = Plan { kind: kind.clone(), ..Default::default() };
    let dest_of = |rel: &str| if kind == "dir" { join(&item.local, rel) } else { item.local.clone() };
    let existing: Vec<String> = entries.keys().map(|r| dest_of(r)).filter(|d| lexists(d)).collect();
    let clean = git_clean_tracked(&existing);
    for (rel, (entry_kind, reference)) in &entries {
        let dest = dest_of(rel);
        if *entry_kind == "link" {
            plan.links.insert(rel.clone(), reference.clone());
        }
        if !lexists(&dest) {
            plan.new.push(rel.clone());
            continue;
        }
        if Path::new(&dest).is_dir() && !is_link(&dest) {
            plan.conflict.push(rel.clone());
            continue;
        }
        if *entry_kind == "file" && !is_link(&dest) && sha(&dest) == *reference {
            plan.same.push(rel.clone());
            continue;
        }
        if *entry_kind == "link" && std::fs::read_link(&dest).is_ok_and(|t| t.to_string_lossy() == reference.as_str()) {
            plan.same.push(rel.clone());
            continue;
        }
        let recorded = manifest.get(&dest).and_then(|v| serde_json::from_value::<[i64; 2]>(v.clone()).ok());
        let replaceable = force
            || (recorded.is_some() && recorded == stamp(&dest))
            || clean.contains(&dest)
            || clean.contains(&realpath(&dest));
        if replaceable {
            plan.update.push(rel.clone());
        } else {
            plan.kept.push(rel.clone());
        }
    }
    plan.item = Some(item);
    plan
}

// ── Fetching ─────────────────────────────────────────────────────────────────

/// Ask the first reachable host of the group to describe every item.
fn describe<'c>(
    transport: &Transport,
    hosts: &[&'c Host],
    items: &[Item],
) -> Result<(&'c Host, Vec<Value>), FetchError> {
    let remotes: Vec<String> = items.iter().map(|i| i.remote.clone()).collect();
    let mut last = String::new();
    for &host in hosts {
        let r = transport.run(host, &list_script(&remotes), LIST_TIMEOUT);
        let line = py::splitlines(&r.stdout).into_iter().find(|l| l.starts_with(&format!("{STEP} ")));
        if r.rc == 0 {
            if let Some(line) = line {
                let descs: Vec<Value> = serde_json::from_str(&line[STEP.len() + 1..])
                    .map_err(|e| FetchError(format!("could not list the remote paths: {e}")))?;
                return Ok((host, descs));
            }
        }
        let e = py::strip(&r.stderr);
        last = py::splitlines(e).last().filter(|_| !e.is_empty()).map_or(format!("exit {}", r.rc), |l| l.to_string());
        if r.rc != 255 {
            break; // reachable but failed: another host of the group won't help
        }
    }
    err(format!("could not list the remote paths: {last}"))
}

/// Why the tarfile 'data' filter refuses a member (its exceptions' messages).
#[derive(Debug, PartialEq, Eq)]
pub struct Refused(pub String);

/// What extracting a member under `dest` (already resolved) would do, by Python's tarfile 'data'
/// filter: the member's path, its mode for regular files, and for links their target. Absolute
/// paths, paths or links leading outside `dest`, absolute links and special files are refused.
pub fn filter(
    name: &str,
    kind: tar::EntryType,
    mode: u32,
    link: Option<&str>,
    dest: &str,
) -> Result<(String, Option<u32>), Refused> {
    let repr = py::repr_str(name);
    let name = name.trim_start_matches('/');
    let target = realpath(&join(dest, name));
    let within = |p: &str| p == dest || p.starts_with(&format!("{}/", dest.trim_end_matches('/')));
    if !within(&target) {
        return Err(Refused(format!(
            "{repr} would be extracted to {}, which is outside the destination",
            py::repr_str(&target)
        )));
    }
    let regular = kind.is_file() || kind.is_hard_link() || kind == tar::EntryType::Continuous || kind.is_gnu_sparse();
    let mode = mode & 0o755;
    let mode = if regular {
        let m = if mode & 0o100 == 0 { mode & !0o111 } else { mode };
        Some(m | 0o600)
    } else if kind.is_dir() || kind.is_symlink() {
        None
    } else {
        return Err(Refused(format!("{repr} is a special file")));
    };
    if kind.is_hard_link() || kind.is_symlink() {
        let link = link.unwrap_or("");
        if link.starts_with('/') {
            return Err(Refused(format!("{repr} is a link to an absolute path")));
        }
        let t = if kind.is_symlink() { join(&join(dest, dirname(name)), link) } else { join(dest, link) };
        let t = realpath(&t);
        if !within(&t) {
            return Err(Refused(format!(
                "{repr} would link to {}, which is outside the destination",
                py::repr_str(&t)
            )));
        }
    }
    Ok((name.to_string(), mode))
}

/// Extract one tar stream into `dest` (an empty temporary directory) as `tarfile.extract(...,
/// filter="data")` does. Returns the members the filter refused, as "name (why)".
pub fn extract(stream: impl Read, dest: &str) -> std::io::Result<Vec<String>> {
    let dest = realpath(dest);
    let mut skipped = vec![];
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(stream));
    for entry in archive.entries()? {
        let mut entry = entry?;
        let name = String::from_utf8_lossy(&entry.path_bytes()).into_owned();
        let kind = entry.header().entry_type();
        let mode = entry.header().mode().unwrap_or(0o644);
        let link = entry.link_name_bytes().map(|l| String::from_utf8_lossy(&l).into_owned());
        // Python writes pax headers, with the exact (fractional) mtime.
        let mut mtime = entry.header().mtime().ok().map(|m| m as f64);
        if let Ok(Some(pax)) = entry.pax_extensions() {
            for ext in pax.flatten() {
                if ext.key() == Ok("mtime") {
                    if let Some(m) = ext.value().ok().and_then(|v| v.parse::<f64>().ok()) {
                        mtime = Some(m);
                    }
                }
            }
        }
        let (rel, mode) = match filter(&name, kind, mode, link.as_deref(), &dest) {
            Ok(x) => x,
            Err(Refused(why)) => {
                skipped.push(format!("{name} ({why})"));
                continue;
            }
        };
        let target = join(&dest, &rel);
        if kind.is_dir() {
            std::fs::create_dir_all(&target)?;
            continue;
        }
        if let Some(parent) = Path::new(&target).parent() {
            std::fs::create_dir_all(parent)?;
        }
        if lexists(&target) && !Path::new(&target).is_dir() {
            std::fs::remove_file(&target)?;
        }
        if kind.is_symlink() {
            std::os::unix::fs::symlink(link.unwrap_or_default(), &target)?;
            continue;
        }
        if kind.is_hard_link() {
            std::fs::hard_link(join(&dest, &link.unwrap_or_default()), &target)?;
        } else {
            let mut f = std::fs::File::create(&target)?;
            std::io::copy(&mut entry, &mut f)?;
            drop(f);
        }
        if let Some(mode) = mode {
            std::fs::set_permissions(&target, std::fs::Permissions::from_mode(mode))?;
        }
        if let Some(mtime) = mtime.filter(|m| *m >= 0.0) {
            let f = std::fs::File::options().write(true).open(&target)?;
            f.set_modified(std::time::UNIX_EPOCH + Duration::from_secs_f64(mtime))?;
        }
    }
    Ok(skipped)
}

/// Copy plan.to_fetch into place; returns the members skipped by the filter.
fn transfer(transport: &Transport, host: &Host, plan: &Plan) -> Result<Vec<String>, FetchError> {
    let item = plan.item();
    let (base_remote, members, dest_root) = if plan.kind == "dir" {
        (item.remote.clone(), plan.to_fetch(), item.local.clone())
    } else {
        let d = dirname(&item.remote);
        (
            if d.is_empty() { "/".into() } else { d.to_string() },
            vec![basename(&item.remote).to_string()],
            dirname(&item.local).to_string(),
        )
    };
    let io = |e: std::io::Error| FetchError(format!("{}: {e}", item.spec));
    std::fs::create_dir_all(&dest_root).map_err(io)?;
    let tmp = tempfile::Builder::new().prefix(".task-fetch-").tempdir_in(&dest_root).map_err(io)?;
    let tmp_path = tmp.path().display().to_string();

    let mut stderr_file = tempfile::tempfile().map_err(io)?;
    let mut child = transport
        .command(host)
        .map_err(io)?
        .stdout(Stdio::piped())
        .stderr(stderr_file.try_clone().map_err(io)?)
        .spawn()
        .map_err(io)?;
    {
        use std::io::Write;
        let mut stdin = child.stdin.take().expect("piped stdin");
        stdin.write_all(tar_script(&base_remote, &members).as_bytes()).map_err(io)?;
    }
    let stdout = child.stdout.take().expect("piped stdout");
    let read_err = |f: &mut std::fs::File| {
        let mut s = String::new();
        let _ = f.rewind().and_then(|_| f.read_to_string(&mut s));
        py::strip(&s).to_string()
    };
    let skipped = match extract(stdout, &tmp_path) {
        Ok(s) => s,
        Err(e) => {
            let _ = child.kill();
            let _ = child.wait();
            let detail = read_err(&mut stderr_file);
            return err(format!(
                "{}: transfer failed: {}",
                item.spec,
                if detail.is_empty() { e.to_string() } else { detail }
            ));
        }
    };
    let status = child.wait().map_err(io)?;
    if !status.success() {
        return err(format!("{}: transfer failed: {}", item.spec, read_err(&mut stderr_file)));
    }
    for rel in &members {
        let src = join(&tmp_path, rel);
        if !lexists(&src) {
            continue; // skipped by the filter
        }
        let dest = if plan.kind == "dir" { join(&dest_root, rel) } else { item.local.clone() };
        std::fs::create_dir_all(dirname(&dest)).map_err(io)?;
        if is_link(&dest) || Path::new(&dest).is_file() {
            std::fs::remove_file(&dest).map_err(io)?;
        }
        std::fs::rename(&src, &dest).map_err(io)?;
    }
    Ok(skipped)
}

pub struct FetchResult<'c> {
    pub plans: Vec<Plan>,
    pub host: &'c Host,
    /// spec → members the filter refused
    pub skipped: HashMap<String, Vec<String>>,
}

pub fn fetch<'c>(
    transport: &'c Transport,
    hosts: &[&'c Host],
    specs: &[String],
    cwd: &str,
    force: bool,
    dry_run: bool,
) -> Result<FetchResult<'c>, FetchError> {
    let (group, items) = resolve(&transport.cfg, &transport.paths, hosts, specs, cwd)?;
    let (host, descs) = describe(transport, &group, &items)?;
    let mut manifest = load_manifest(&transport.paths);
    let plans: Vec<Plan> = items.into_iter().zip(&descs).map(|(i, d)| plan_item(i, d, &manifest, force)).collect();
    let mut res = FetchResult { plans, host, skipped: HashMap::new() };
    if dry_run {
        return Ok(res);
    }
    for plan in &res.plans {
        if plan.to_fetch().is_empty() {
            continue;
        }
        res.skipped.insert(plan.item().spec.clone(), transfer(transport, host, plan)?);
        for rel in plan.to_fetch() {
            let dest = plan.dest(&rel);
            if let Some(s) = stamp(&dest) {
                manifest.insert(dest, json!(s));
            }
        }
    }
    save_manifest(&transport.paths, &manifest).map_err(|e| FetchError(format!("fetched.json: {e}")))?;
    Ok(res)
}

// ── Which task wrote what (task fetch --after A B ...) ─────────────────────────

/// Local paths the task's command names, as an argument or a --flag=value, resolved from its
/// working directory. The working directory and anything above it are left out: they name every
/// file.
pub fn named_paths(t: &Task, home: &Path) -> Vec<String> {
    let words = crate::shell::split(&t.command)
        .unwrap_or_else(|_| py::split(&t.command).into_iter().map(String::from).collect());
    let wdir = &t.working_dir;
    let mut out = vec![];
    for w in words {
        let value = w.split_once('=').map_or(w.as_str(), |(_, v)| v);
        if value.is_empty() || value.starts_with('-') {
            continue;
        }
        let p = normpath(&join(wdir, &crate::paths::expand_user(value, home)));
        if !inside(wdir, &p) {
            out.push(p);
        }
    }
    out
}

/// How deep the command names path (its own name or a directory above it); -1 if it doesn't.
fn depth_named(t: &Task, path: &str, home: &Path) -> i64 {
    named_paths(t, home).iter().filter(|p| inside(path, p)).map(|p| p.split('/').count() as i64).max().unwrap_or(-1)
}

fn ran_when_written(t: &Task, path: &str) -> bool {
    let (Some(started), Some(finished)) = (t.started_at.filter(|s| *s != 0.0), t.finished_at.filter(|f| *f != 0.0))
    else {
        return false;
    };
    let Ok(m) = std::fs::symlink_metadata(path) else { return false };
    let mtime = m.mtime() as f64 + m.mtime_nsec() as f64 / 1e9; // the server's: the extraction keeps it
    started - WRITE_SLACK <= mtime && mtime <= finished + WRITE_SLACK
}

/// Credit each fetched file to the tasks that wrote it: the tasks whose command names it most
/// specifically (`--out results/a` names results/a/x.csv), then, if that leaves more than one,
/// those that were running when it was written. Returns ({task id: paths}, the paths still
/// credited to more than one task). With one task, it gets everything.
pub fn attribute(rows: &[Task], paths: &[String], home: &Path) -> (IndexMap<i64, Vec<String>>, Vec<String>) {
    let mut by_task: IndexMap<i64, Vec<String>> = rows.iter().map(|r| (r.id, vec![])).collect();
    let mut shared = vec![];
    for path in paths {
        let depth: Vec<i64> = rows.iter().map(|r| depth_named(r, path, home)).collect();
        let deepest = depth.iter().copied().max().unwrap_or(-1);
        let mut cands: Vec<&Task> = rows.iter().zip(&depth).filter(|(_, d)| **d == deepest).map(|(r, _)| r).collect();
        if cands.len() > 1 {
            let ran: Vec<&Task> = cands.iter().copied().filter(|r| ran_when_written(r, path)).collect();
            if !ran.is_empty() {
                cands = ran;
            }
        }
        if cands.len() > 1 {
            shared.push(path.clone());
        }
        for r in cands {
            by_task.get_mut(&r.id).expect("every row has an entry").push(path.clone());
        }
    }
    (by_task, shared)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;

    fn git(root: &Path, args: &[&str]) {
        let ok = Command::new("git")
            .args(["-c", "user.email=t@example.com", "-c", "user.name=t", "-C"])
            .arg(root)
            .args(args)
            .stdout(Stdio::null())
            .status()
            .unwrap()
            .success();
        assert!(ok, "git {args:?}");
    }

    /// A committed repo with these files, named projN so its remote copy is <remote_root>/projN.
    fn repo(tmp: &Path, files: &[(&str, &str)]) -> String {
        let root = tmp.join("proj");
        for (rel, text) in files {
            std::fs::create_dir_all(root.join(rel).parent().unwrap()).unwrap();
            std::fs::write(root.join(rel), text).unwrap();
        }
        std::fs::create_dir_all(&root).unwrap();
        git(&root, &["init", "-q"]);
        git(&root, &["add", "."]);
        git(&root, &["commit", "-qm", "init"]);
        realpath(&root.display().to_string())
    }

    fn cfg(two_groups: bool, path_map: Option<&str>) -> (Config, Paths) {
        let paths = Paths::for_home("/home/me");
        let mut toml = String::from("[groups.g]\nhosts = [\"a\", \"b\"]\nremote_root = \"/vol/me\"\n");
        if let Some(local) = path_map {
            toml += &format!("path_map = [ {{ local = \"{local}\", remote = \"/vol/mapped\" }} ]\n");
        }
        if two_groups {
            toml += "[groups.h]\nhosts = [\"c\"]\nremote_root = \"/vol/other\"\n";
        }
        (crate::config::parse_config(&toml.parse().unwrap(), &paths).unwrap(), paths)
    }

    fn sha_of(text: &str) -> String {
        format!("{:x}", sha2::Sha256::digest(text.as_bytes()))
    }

    fn specs(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn resolve_maps_repo_paths_like_jobs() {
        let tmp = tempfile::tempdir().unwrap();
        let root = repo(tmp.path(), &[("results/x.txt", "x")]);
        let (cfg, paths) = cfg(false, None);
        let hosts = cfg.select(Some("g")).unwrap();
        let (group, items) = resolve(&cfg, &paths, &hosts, &specs(&["results", "results/x.txt"]), &root).unwrap();
        assert_eq!(group.iter().map(|h| h.name.as_str()).collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(items[0].remote, "/vol/me/proj/results");
        assert_eq!(items[1].remote, "/vol/me/proj/results/x.txt");
        assert_eq!(items[0].local, format!("{root}/results"));
        let (_, items) = resolve(&cfg, &paths, &hosts, &specs(&["x.txt"]), &format!("{root}/results")).unwrap();
        assert_eq!(items[0].remote, "/vol/me/proj/results/x.txt"); // relative to the cwd
    }

    #[test]
    fn resolve_refuses_paths_outside_the_repo_and_ambiguous_groups() {
        let tmp = tempfile::tempdir().unwrap();
        let root = repo(tmp.path(), &[("a", "a")]);
        let (cfg, paths) = cfg(false, None);
        let hosts = cfg.select(Some("g")).unwrap();
        let e = resolve(&cfg, &paths, &hosts, &specs(&["../elsewhere"]), &root).unwrap_err();
        assert!(e.0.contains("outside the repo"), "{e}");
        let e = resolve(&cfg, &paths, &hosts, &specs(&["x"]), "/").unwrap_err();
        assert!(e.0.contains("REMOTE:LOCAL"), "{e}");
        let (cfg2, paths) = self::cfg(true, None);
        let all = cfg2.select(None).unwrap();
        let e = resolve(&cfg2, &paths, &all, &specs(&["a"]), &root).unwrap_err();
        assert_eq!(e.0, "the selected hosts span several storage groups (g, h); pick one with -H");
        let h = cfg2.select(Some("h")).unwrap();
        assert_eq!(resolve(&cfg2, &paths, &h, &specs(&["a"]), &root).unwrap().1[0].remote, "/vol/other/proj/a");
    }

    #[test]
    fn resolve_explicit_and_path_map() {
        let tmp = tempfile::tempdir().unwrap();
        let root = repo(tmp.path(), &[("a", "a")]);
        let (cfg, paths) = cfg(false, None);
        let hosts = cfg.select(Some("g")).unwrap();
        let (_, items) = resolve(&cfg, &paths, &hosts, &specs(&["/vol/run/results:out", "~/r:"]), &root).unwrap();
        assert_eq!((items[0].remote.as_str(), items[0].local.clone()), ("/vol/run/results", format!("{root}/out")));
        assert_eq!((items[1].remote.as_str(), items[1].local.as_str()), ("~/r", root.as_str()));
        let (cfg, paths) = self::cfg(false, Some(&root));
        let hosts = cfg.select(Some("g")).unwrap();
        let (_, items) = resolve(&cfg, &paths, &hosts, &specs(&["sub/f"]), &root).unwrap();
        assert_eq!(items[0].remote, "/vol/mapped/sub/f"); // path_map wins
    }

    #[test]
    fn plan_replaces_only_what_can_be_recovered() {
        let tmp = tempfile::tempdir().unwrap();
        let root = repo(tmp.path(), &[("res/tracked_clean", "old"), ("res/tracked_dirty", "old")]);
        let res = format!("{root}/res");
        std::fs::write(format!("{res}/tracked_dirty"), "my edit").unwrap();
        std::fs::write(format!("{res}/untracked"), "mine").unwrap();
        std::fs::write(format!("{res}/same"), "same").unwrap();
        std::fs::write(format!("{res}/fetched_before"), "v1").unwrap();
        std::fs::create_dir(format!("{res}/is_a_dir")).unwrap();
        let before = format!("{res}/fetched_before");
        let manifest: Manifest = [(before.clone(), json!(stamp(&before).unwrap()))].into_iter().collect();
        let mut files = serde_json::Map::new();
        for n in ["tracked_clean", "tracked_dirty", "untracked", "fetched_before", "is_a_dir", "new"] {
            files.insert(n.into(), json!(sha_of("remote")));
        }
        files.insert("same".into(), json!(sha_of("same")));
        let desc = json!({"kind": "dir", "links": {"link": "same"}, "files": files});
        let item = Item { spec: "res".into(), local: res.clone(), remote: "/remote/res".into() };
        let plan = plan_item(item.clone(), &desc, &manifest, false);
        assert_eq!(plan.new, ["link", "new"]);
        assert_eq!(plan.update, ["fetched_before", "tracked_clean"]); // the manifest / git has it
        assert_eq!(plan.kept, ["tracked_dirty", "untracked"]);
        assert_eq!(
            (plan.same.clone(), plan.conflict.clone()),
            (vec!["same".to_string()], vec!["is_a_dir".to_string()])
        );
        let forced = plan_item(item, &desc, &manifest, true);
        assert_eq!(forced.update, ["fetched_before", "tracked_clean", "tracked_dirty", "untracked"]);
        assert_eq!(forced.conflict, ["is_a_dir"]);
    }

    #[test]
    fn plan_file_into_an_existing_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().display().to_string();
        let item = Item { spec: "/r/x.txt:.".into(), local: dir.clone(), remote: "/r/x.txt".into() };
        let plan = plan_item(item, &json!({"kind": "file", "sha": sha_of("x")}), &Manifest::new(), false);
        assert_eq!(plan.new, [""]);
        assert_eq!(plan.item().local, format!("{dir}/x.txt"));
    }

    fn task(id: i64, command: &str, wdir: &str, started: Option<f64>, finished: Option<f64>) -> Task {
        Task {
            id,
            command: command.into(),
            working_dir: wdir.into(),
            started_at: started,
            finished_at: finished,
            ..Default::default()
        }
    }

    fn written(root: &str, rel: &str, mtime: f64) -> String {
        let p = format!("{root}/{rel}");
        std::fs::create_dir_all(dirname(&p)).unwrap();
        std::fs::write(&p, rel).unwrap();
        let f = std::fs::File::options().write(true).open(&p).unwrap();
        f.set_modified(std::time::UNIX_EPOCH + Duration::from_secs_f64(mtime)).unwrap();
        p
    }

    fn by(pairs: &[(i64, &[&String])]) -> IndexMap<i64, Vec<String>> {
        pairs.iter().map(|(id, ps)| (*id, ps.iter().map(|p| p.to_string()).collect())).collect()
    }

    #[test]
    fn attribute_by_what_commands_name_then_when_they_ran() {
        let tmp = tempfile::tempdir().unwrap();
        let w = realpath(&tmp.path().display().to_string());
        let home = Path::new("/home/me");
        let (s, f) = (Some(1000.0), Some(2000.0));
        // tasks 85 and 86 of the red-herring rerun: same time, same script, different --out
        let a = task(85, "PYTHONPATH=src python3 -m red_herring.experiment --out results/ungrouped", &w, s, f);
        let b = task(
            86,
            "PYTHONPATH=src python3 -m red_herring.experiment --groups tissue --out=results/grouped",
            &w,
            s,
            f,
        );
        let (ua, ga) =
            (written(&w, "results/ungrouped/summary.csv", 1500.0), written(&w, "results/grouped/s.csv", 1500.0));
        assert_eq!(attribute(&[a, b], &[ua.clone(), ga.clone()], home), (by(&[(85, &[&ua]), (86, &[&ga])]), vec![]));

        // the most specific name wins
        let a = task(1, "python3 sweep.py --out results", &w, s, f);
        let b = task(2, "python3 eval.py --out results/eval", &w, s, f);
        let (top, ev) = (written(&w, "results/sweep.csv", 1500.0), written(&w, "results/eval/e.csv", 1500.0));
        assert_eq!(attribute(&[a, b], &[top.clone(), ev.clone()], home), (by(&[(1, &[&top]), (2, &[&ev])]), vec![]));

        // then when each ran
        let a = task(1, "python3 train.py", &w, Some(1000.0), Some(2000.0));
        let b = task(2, "python3 train.py", &w, Some(5000.0), Some(6000.0));
        let (early, late) = (written(&w, "results/early.txt", 2050.0), written(&w, "results/late.txt", 5500.0));
        assert_eq!(
            attribute(&[a, b], &[early.clone(), late.clone()], home),
            (by(&[(1, &[&early]), (2, &[&late])]), vec![])
        );

        // nothing tells them apart: both, and it's reported
        let (a, b) = (task(1, "python3 train.py", &w, s, f), task(2, "python3 train.py", &w, s, f));
        let p = written(&w, "results/out.txt", 1500.0);
        assert_eq!(
            attribute(&[a, b], std::slice::from_ref(&p), home),
            (by(&[(1, &[&p]), (2, &[&p])]), vec![p.clone()])
        );
    }

    #[test]
    fn one_task_gets_everything_and_dot_names_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let w = realpath(&tmp.path().display().to_string());
        let home = Path::new("/home/me");
        let a = task(7, "cd . && python3 train.py --data /", &w, None, None);
        let p = written(&w, "results/out.txt", 9e9);
        assert_eq!(attribute(std::slice::from_ref(&a), std::slice::from_ref(&p), home), (by(&[(7, &[&p])]), vec![]));
        let named = named_paths(&a, home);
        assert!(named.contains(&format!("{w}/train.py")), "{named:?}");
        assert!(!named.contains(&w) && !named.contains(&"/".to_string()), "{named:?}");
        assert!(!named_paths(&task(8, "python3 'unbalanced", &w, None, None), home).is_empty());
        // not shell-parsable
    }

    #[test]
    fn extract_keeps_links_and_refuses_unsafe_members() {
        let tmp = tempfile::tempdir().unwrap();
        let mut buf = Vec::new();
        {
            let gz = flate2::write::GzEncoder::new(&mut buf, flate2::Compression::default());
            let mut b = tar::Builder::new(gz);
            let mut file = |name: &str, data: &[u8], mode: u32| {
                let mut h = tar::Header::new_gnu();
                h.set_size(data.len() as u64);
                h.set_mode(mode);
                h.set_mtime(1_700_000_000);
                h.set_entry_type(tar::EntryType::Regular);
                b.append_data(&mut h, name, data).unwrap();
            };
            file("a.txt", b"hello\n", 0o4775);
            file("sub/b c.txt", b"x", 0o640);
            let mut link = |name: &str, target: &str, kind: tar::EntryType| {
                let mut h = tar::Header::new_gnu();
                h.set_entry_type(kind);
                h.set_size(0);
                h.set_mode(0o777);
                h.set_mtime(1_700_000_000); // a hard link shares the file's inode, mtime included
                b.append_link(&mut h, name, target).unwrap();
            };
            link("inside", "a.txt", tar::EntryType::Symlink);
            link("outside", "../../etc", tar::EntryType::Symlink);
            link("abs", "/etc/passwd", tar::EntryType::Symlink);
            link("hard", "a.txt", tar::EntryType::Link);
            b.into_inner().unwrap().finish().unwrap();
        }
        let dest = tmp.path().join("t");
        std::fs::create_dir(&dest).unwrap();
        let d = dest.display().to_string();
        let skipped = extract(&buf[..], &d).unwrap();
        assert_eq!(skipped.len(), 2, "{skipped:?}");
        assert!(skipped[0].starts_with("outside ('outside' would link to "), "{skipped:?}");
        assert_eq!(skipped[1], "abs ('abs' is a link to an absolute path)");
        assert_eq!(std::fs::read_to_string(dest.join("a.txt")).unwrap(), "hello\n");
        let meta = std::fs::metadata(dest.join("a.txt")).unwrap();
        assert_eq!(meta.permissions().mode() & 0o7777, 0o755); // setuid and group write dropped
        assert_eq!(meta.mtime(), 1_700_000_000);
        assert_eq!(std::fs::read_link(dest.join("inside")).unwrap(), Path::new("a.txt")); // a link, not a copy
        assert_eq!(std::fs::metadata(dest.join("hard")).unwrap().ino(), meta.ino());
        assert_eq!(std::fs::read_to_string(dest.join("sub/b c.txt")).unwrap(), "x");
        assert!(!lexists(&format!("{d}/outside")) && !lexists(&format!("{d}/abs")));
    }

    #[test]
    fn os_path_helpers() {
        assert_eq!(dirname("/a/b"), "/a");
        assert_eq!(dirname("/a"), "/");
        assert_eq!(dirname("a"), "");
        assert_eq!(dirname("a//b"), "a");
        assert_eq!(dirname("//a"), "//");
        assert_eq!(join("a", "b"), "a/b");
        assert_eq!(join("a/", "b"), "a/b");
        assert_eq!(join("a", "/b"), "/b");
        assert!(is_explicit("/vol/x:out") && is_explicit("~/r:") && !is_explicit("results") && !is_explicit("a:b"));
    }
}
