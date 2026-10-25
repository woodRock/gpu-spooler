//! Keeping a remote copy of a local git repo, for submitting from a machine that doesn't share a
//! filesystem with the servers (e.g. a laptop).
//!
//! The remote copy lives at <remote_root>/<repo name> on each host group (ECS servers share NFS,
//! so one copy serves the whole group). `task sync` pushes a snapshot of the local working tree -
//! HEAD plus uncommitted changes to tracked files, via `git stash create`, which doesn't touch the
//! local repo - over the spooler's own ssh route, then checks it out (detached) on the remote
//! side. The servers never need credentials for the git origin.

use crate::config::{Config, Host};
use crate::jobs::FIND_GIT;
use crate::py;
use crate::shell::sh_path;
use crate::transport::Transport;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

pub const SYNC_REF: &str = "refs/gpu-spooler/sync";
pub const RECEIVE_PACK: &str = "bash -lc 'exec git-receive-pack \"$@\"' git-receive-pack";
pub const CHECK_TIMEOUT: Duration = Duration::from_secs(20);
pub const PUSH_TIMEOUT: Duration = Duration::from_secs(600);

fn git(root: &str, args: &[&str]) -> Option<(bool, String)> {
    let out = Command::new("git").arg("-C").arg(root).args(args).output().ok()?;
    Some((out.status.success(), String::from_utf8_lossy(&out.stdout).into_owned()))
}

// ── Local repo ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct LocalRepo {
    /// top-level directory
    pub root: String,
    /// None for a repo without commits
    pub head: Option<String>,
    pub branch: Option<String>,
    /// uncommitted changes to tracked files
    pub dirty: bool,
    snapshot: OnceLock<String>,
}

impl LocalRepo {
    pub fn name(&self) -> &str {
        self.root.trim_end_matches('/').rsplit('/').next().unwrap_or("")
    }

    /// Commit capturing the working tree (HEAD itself when clean).
    pub fn snapshot(&self) -> &str {
        self.snapshot.get_or_init(|| {
            let mut commit = String::new();
            if self.dirty {
                commit = git(&self.root, &["stash", "create", "gpu-spooler sync"])
                    .map(|(_, out)| py::strip(&out).to_string())
                    .unwrap_or_default();
            }
            if commit.is_empty() {
                commit = self.head.clone().unwrap_or_default();
            }
            commit
        })
    }

    pub fn tree(&self) -> String {
        git(&self.root, &["rev-parse", &format!("{}^{{tree}}", self.snapshot())])
            .map(|(_, out)| py::strip(&out).to_string())
            .unwrap_or_default()
    }

    pub fn describe(&self) -> String {
        let head: String = self.head.as_deref().unwrap_or("").chars().take(7).collect();
        let branch = self.branch.as_ref().map_or(String::new(), |b| format!(" ({b})"));
        format!("{head}{branch}{}", if self.dirty { " + uncommitted changes" } else { "" })
    }
}

/// The git repo containing path, or None (also when git is missing).
pub fn find_repo(path: &str) -> Option<LocalRepo> {
    let (ok, out) = git(path, &["rev-parse", "--show-toplevel"])?;
    let root = py::strip(&out).to_string();
    if !ok || root.is_empty() {
        return None;
    }
    let value = |args: &[&str]| git(&root, args).map(|(_, o)| py::strip(&o).to_string()).filter(|s| !s.is_empty());
    let head = value(&["rev-parse", "-q", "--verify", "HEAD"]);
    let branch = value(&["symbolic-ref", "-q", "--short", "HEAD"]);
    let dirty = value(&["status", "--porcelain", "--untracked-files=no"]).is_some();
    Some(LocalRepo { root, head, branch, dirty, snapshot: OnceLock::new() })
}

// ── Where the copies live ─────────────────────────────────────────────────────

/// One remote copy: a representative host for a group sharing remote_root.
#[derive(Clone, Debug)]
pub struct Target<'c> {
    pub host: &'c Host,
    pub members: Vec<String>,
    pub path: String,
}

impl Target<'_> {
    pub fn label(&self) -> &str {
        self.host.group.as_deref().filter(|g| !g.is_empty()).unwrap_or(&self.host.name)
    }
}

/// Remote copies needed for running local_dir's repo on these hosts. Hosts in the same group with
/// the same remote_root are assumed to share storage. Hosts where a path_map entry applies are
/// left out (that path is yours).
pub fn targets<'c>(cfg: &'c Config, hosts: &[&'c Host], repo: &LocalRepo, local_dir: &str) -> Vec<Target<'c>> {
    let mut out: Vec<((String, String), Target<'c>)> = vec![];
    for &h in hosts {
        let Some(root) = h.remote_root.as_ref().filter(|r| !r.is_empty()) else { continue };
        if h.is_local() || cfg.is_mapped(h, local_dir) {
            continue;
        }
        let key = (h.group.clone().unwrap_or_else(|| h.name.clone()), root.clone());
        match out.iter_mut().find(|(k, _)| *k == key) {
            Some((_, t)) => t.members.push(h.name.clone()),
            None => {
                let path = cfg.remote_dir(h, &repo.root, Some(&repo.root));
                out.push((key, Target { host: h, members: vec![h.name.clone()], path }));
            }
        }
    }
    out.into_iter().map(|(_, t)| t).collect()
}

// ── Checking a remote copy ────────────────────────────────────────────────────

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RemoteState {
    /// missing | notgit | ok | nogit | unreachable
    pub state: String,
    pub tree: String,
    pub head: String,
    pub dirty: i64,
    pub error: String,
}

pub fn check_script(path: &str) -> String {
    let p = sh_path(path);
    format!(
        r#"{FIND_GIT}command -v git >/dev/null 2>&1 || {{ echo "STATE nogit"; exit 0; }}
P={p}
if [ ! -e "$P" ]; then echo "STATE missing"
elif [ "$(git -C "$P" rev-parse --show-toplevel 2>/dev/null)" != "$(cd "$P" && pwd -P)" ]; then
  echo "STATE notgit"
else
  echo "STATE ok"
  echo "TREE $(git -C "$P" rev-parse -q --verify 'HEAD^{{tree}}' 2>/dev/null)"
  echo "HEAD $(git -C "$P" rev-parse -q --verify HEAD 2>/dev/null)"
  echo "DIRTY $(git -C "$P" status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')"
fi
"#
    )
}

/// Ask a target's host what state its copy is in.
pub fn check(transport: &Transport, target: &Target) -> RemoteState {
    let r = transport.run(target.host, &check_script(&target.path), CHECK_TIMEOUT);
    let mut fields: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    for line in py::splitlines(&r.stdout) {
        if !py::strip(line).is_empty() {
            let (k, v) = line.split_once(' ').unwrap_or((line, ""));
            fields.insert(k, v);
        }
    }
    let Some(state) = fields.get("STATE").filter(|_| r.rc == 0) else {
        let err = py::strip(&r.stderr);
        let msg =
            py::splitlines(err).last().filter(|_| !err.is_empty()).map_or(format!("exit {}", r.rc), |l| l.to_string());
        return RemoteState { state: "unreachable".into(), error: msg, ..Default::default() };
    };
    let dirty = fields.get("DIRTY").map_or(0, |d| if d.is_empty() { 0 } else { py::int(d).unwrap_or(0) });
    RemoteState {
        state: state.to_string(),
        tree: fields.get("TREE").unwrap_or(&"").to_string(),
        head: fields.get("HEAD").unwrap_or(&"").to_string(),
        dirty,
        error: String::new(),
    }
}

/// None if the remote copy matches the local working tree, else why not.
pub fn staleness(repo: &LocalRepo, target: &Target, st: &RemoteState) -> Option<String> {
    let place = format!("{}:{}", target.label(), target.path);
    Some(match st.state.as_str() {
        "unreachable" => format!("could not check {place}: {}", st.error),
        "nogit" => format!("git is not installed on {}", target.host.name),
        "missing" => format!("repo '{}' doesn't exist on {place}; run `task sync`", repo.name()),
        "notgit" => format!("{place} exists but is not a git repo"),
        _ if st.dirty != 0 => format!(
            "{place} has {} uncommitted change(s) on the server; `task sync` will refuse to overwrite conflicting files",
            st.dirty
        ),
        _ if st.tree != repo.tree() => {
            let head: String = st.head.chars().take(7).collect();
            format!(
                "{place} is at {}, but you're at {}; run `task sync`",
                if head.is_empty() { "no commit".into() } else { head },
                repo.describe()
            )
        }
        _ => return None,
    })
}

// ── Syncing ───────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SyncResult<'c> {
    pub target: Target<'c>,
    pub ok: bool,
    pub message: String,
}

pub fn prepare_script(path: &str) -> String {
    let p = sh_path(path);
    format!(
        r#"{FIND_GIT}command -v git >/dev/null 2>&1 || {{ echo "git is not installed" >&2; exit 3; }}
P={p}
if [ ! -e "$P" ]; then
  mkdir -p "$P" && git init -q "$P" || exit 1
elif [ "$(git -C "$P" rev-parse --show-toplevel 2>/dev/null)" != "$(cd "$P" && pwd -P)" ]; then
  if [ -n "$(ls -A "$P")" ]; then echo "$P exists and is not a git repo" >&2; exit 4; fi
  git init -q "$P" || exit 1
fi
echo READY
"#
    )
}

pub fn checkout_script(path: &str, commit: &str) -> String {
    format!(
        "{FIND_GIT}cd {} || exit 1\ngit checkout -q --detach {} || exit 5\necho SYNCED\n",
        sh_path(path),
        crate::shell::quote(commit)
    )
}

fn last_line(text: &str, rc: i32) -> String {
    let t = py::strip(text);
    py::splitlines(t).last().filter(|_| !t.is_empty()).map_or(format!("exit {rc}"), |l| l.to_string())
}

/// Push the local working tree snapshot to one target and check it out.
pub fn sync<'c>(transport: &Transport, repo: &LocalRepo, target: &Target<'c>) -> SyncResult<'c> {
    let result = |ok: bool, message: String| SyncResult { target: target.clone(), ok, message };
    let commit = repo.snapshot().to_string();
    if commit.is_empty() {
        return result(false, "the local repo has no commits yet".into());
    }
    let host = target.host;

    let r = transport.run(host, &prepare_script(&target.path), Duration::from_secs(60));
    if r.rc != 0 || !r.stdout.contains("READY") {
        return result(false, last_line(&r.stderr, r.rc));
    }

    let ssh_config = match transport.ssh_config(None) {
        Ok(p) => p,
        Err(e) => return result(false, format!("git push failed: {e}")),
    };
    let push = Command::new("git")
        .args(["-C", &repo.root, "push", "-q", "--force", "--no-verify"])
        .arg(format!("--receive-pack={RECEIVE_PACK}"))
        .arg(format!("{}:{}", host.alias(), target.path))
        .arg(format!("{commit}:{SYNC_REF}"))
        .env("GIT_SSH_COMMAND", format!("ssh -F {}", crate::shell::quote(&ssh_config.display().to_string())))
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn();
    let mut push = match push {
        Ok(p) => p,
        Err(e) => return result(false, format!("git push failed: {e}")),
    };
    // stderr is read on a thread, so a chatty git can't fill the pipe while we wait.
    let mut err_pipe = push.stderr.take().expect("piped stderr");
    let reader = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = std::io::Read::read_to_string(&mut err_pipe, &mut s);
        s
    });
    use wait_timeout::ChildExt;
    let status = match push.wait_timeout(PUSH_TIMEOUT) {
        Ok(Some(s)) => s,
        _ => {
            let _ = push.kill();
            let _ = push.wait();
            return result(false, format!("git push timed out after {}s", PUSH_TIMEOUT.as_secs()));
        }
    };
    let stderr = reader.join().unwrap_or_default();
    if !status.success() {
        return result(false, format!("git push failed: {}", last_line(&stderr, status.code().unwrap_or(-1))));
    }

    let r = transport.run(host, &checkout_script(&target.path, &commit), Duration::from_secs(120));
    if r.rc != 0 || !r.stdout.contains("SYNCED") {
        let lines = py::splitlines(py::strip(&r.stderr));
        let detail = lines[lines.len().saturating_sub(3)..].join(" ");
        let detail = if detail.is_empty() { format!("exit {}", r.rc) } else { detail };
        return result(false, format!("checkout failed on the server: {detail}"));
    }
    result(true, format!("{} now at {}", target.path, repo.describe()))
}
