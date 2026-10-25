//! Detached jobs.
//!
//! Each launch attempt gets a job directory on the host that runs it,
//! $HOME/.gpu-spooler/jobs/<job_key>/, holding run.sh, pid, out.log and (once the command exits)
//! rc. Jobs are started in their own session, so they keep running when the ssh connection that
//! launched them goes away; the daemon polls the directory to sync the log and pick up the exit
//! code. The scripts are the Python version's byte for byte, so either daemon can adopt the
//! other's jobs.

use crate::py;
use crate::shell::{quote, sh_path};
use base64::Engine;
use indexmap::IndexMap;

pub const JOBS_DIR: &str = "$HOME/.gpu-spooler/jobs";
/// Max log bytes fetched per job per poll.
pub const CHUNK_BYTES: usize = 1048576;

/// git may only be on a login shell's PATH (on ECS it's in /usr/pkg/bin), and both our `bash -s`
/// scripts and the remote end of `git push` run non-login.
pub const FIND_GIT: &str =
    "command -v git >/dev/null 2>&1 || PATH=\"$(bash -lc 'printf %s \"$PATH\"' </dev/null 2>/dev/null):$PATH\"\n";

/// taura's environment probe (provenance::PROBE), run in the job's own interpreter.
pub const PROBE: &str = include_str!("remote/taura_probe.py");
/// Marks the probe's output line in the job's log.
pub const ENV_MARK: &str = "TAURA_ENV ";

/// `<task id>-<6 random hex digits>`.
pub fn new_job_key(task_id: i64) -> String {
    let mut bytes = [0u8; 3];
    getrandom::fill(&mut bytes).expect("the OS random number generator");
    format!("{task_id}-{:02x}{:02x}{:02x}", bytes[0], bytes[1], bytes[2])
}

/// Script (for `bash -s`) that starts a job detached and prints `TS_PID <pid>`.
///
/// `probe`: when the task is recorded with taura, how to run taura's environment probe like the
/// job's command (e.g. `PYTHONPATH=src python3`); the job then prints a `TAURA_ENV {...}` line first.
pub fn launch_script(key: &str, wdir: &str, gpus: &str, command: &str, delay: bool, probe: Option<&str>) -> String {
    let q_wdir = sh_path(wdir);
    let tag = key.replace('-', "_");
    let eof = format!("TS_EOF_{tag}");
    let (probe_file, probe_run) = match probe {
        Some(probe) => {
            let peof = format!("TS_PROBE_{tag}");
            (
                format!("cat > \"$D/taura_probe.py\" <<'{peof}'\n{}\n{peof}\n", PROBE.trim_matches('\n')),
                format!("{probe} \"$D/taura_probe.py\" 2>/dev/null | sed \"s/^/{ENV_MARK}/\" || true\n"),
            )
        }
        None => (String::new(), String::new()),
    };
    let gpus = quote(gpus);
    let pause = if delay { "sleep 2" } else { ":" };
    format!(
        r#"D="{JOBS_DIR}/{key}"
mkdir -p "$D" || exit 1
{probe_file}cat > "$D/run.sh" <<'{eof}'
D="{JOBS_DIR}/{key}"
source ~/.bashrc 2>/dev/null || true
ls {q_wdir} > /dev/null 2>&1
cd {q_wdir} || {{ echo "gpu-spooler: cannot cd to "{q_wdir} >&2; echo 1 > "$D/rc.tmp"; mv "$D/rc.tmp" "$D/rc"; exit 1; }}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES={gpus}
{pause}
{probe_run}( {command}
)
rc=$?
echo "$rc" > "$D/rc.tmp" && mv "$D/rc.tmp" "$D/rc"
exit $rc
{eof}
ts_detach() {{
  trap '' HUP     # like nohup: an ignored SIGHUP survives exec (macOS nohup
                  # refuses to run outside a console session)
  if command -v setsid >/dev/null 2>&1; then exec setsid "$@"
  else exec python3 -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' "$@"; fi
}}
cd "$HOME"
# The commit the job will run, for run provenance (taura): HEAD and uncommitted changes.
# git may only be on a login shell's PATH (on ECS it's in /usr/pkg/bin).
{FIND_GIT}( cd {q_wdir} 2>/dev/null && h=$(git rev-parse -q --verify HEAD 2>/dev/null)   && echo "TS_GIT $h $(git status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')"   && git status --porcelain --untracked-files=no 2>/dev/null | head -50 | sed 's/^.../TS_DIRTY /' )
ts_detach bash -l "$D/run.sh" > "$D/out.log" 2>&1 < /dev/null &
echo $! > "$D/pid"
echo "TS_PID $!"
"#
    )
}

/// The PID the launch script printed.
pub fn parse_launch(stdout: &str) -> Option<i64> {
    let line = py::splitlines(stdout).into_iter().find(|l| l.starts_with("TS_PID "))?;
    py::split(line).get(1).and_then(|p| py::int(p))
}

/// (commit, uncommitted changes) the launch script saw in the working directory, if a git repo.
pub fn parse_git(stdout: &str) -> Option<(String, i64)> {
    for line in py::splitlines(stdout) {
        let parts = py::split(line);
        if parts.len() == 3 && parts[0] == "TS_GIT" {
            return py::int(parts[2]).map(|n| (parts[1].to_string(), n));
        }
    }
    None
}

/// The changed tracked files the launch script listed (at most 50): taura tells code from outputs.
pub fn parse_dirty(stdout: &str) -> Vec<String> {
    py::splitlines(stdout)
        .into_iter()
        .filter_map(|l| l.strip_prefix("TS_DIRTY "))
        .map(|rest| {
            let last = rest.rsplit(" -> ").next().unwrap_or(rest);
            py::strip(last).trim_matches('"').to_string()
        })
        .collect()
}

/// Script reporting state and new log bytes for (job_key, offset) pairs.
pub fn status_script(specs: &[(String, i64)]) -> String {
    let items: Vec<String> = specs.iter().map(|(k, off)| quote(&format!("{k}:{off}"))).collect();
    format!(
        r#"for spec in {}; do
  k=${{spec%%:*}}; off=${{spec##*:}}; D="{JOBS_DIR}/$k"
  printf '@@JOB %s\n' "$k"
  if [ -f "$D/rc" ]; then printf 'STATE done %s\n' "$(cat "$D/rc")"
  elif [ ! -d "$D" ]; then echo 'STATE missing'
  elif [ -f "$D/pid" ] && kill -0 "$(cat "$D/pid")" 2>/dev/null; then echo 'STATE running'
  elif [ -f "$D/rc" ]; then printf 'STATE done %s\n' "$(cat "$D/rc")"
  else echo 'STATE gone'; fi
  printf 'SIZE %s\n' "$(wc -c < "$D/out.log" 2>/dev/null | tr -d ' ' || echo 0)"
  printf 'DATA '; tail -c +$((off+1)) "$D/out.log" 2>/dev/null | head -c {CHUNK_BYTES} | base64 | tr -d '\n'; echo
done
echo '@@END'
"#,
        items.join(" ")
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobStatus {
    /// running | done | missing | gone
    pub state: String,
    pub rc: Option<i64>,
    pub size: i64,
    pub data: Vec<u8>,
}

/// Parse status_script output; None if it was cut short.
pub fn parse_status(stdout: &str) -> Option<IndexMap<String, JobStatus>> {
    let mut out: IndexMap<String, JobStatus> = IndexMap::new();
    let mut cur: Option<String> = None;
    let mut complete = false;
    for line in py::splitlines(stdout) {
        if let Some(key) = line.strip_prefix("@@JOB ") {
            let key = py::strip(key).to_string();
            out.insert(key.clone(), JobStatus { state: "gone".into(), rc: None, size: 0, data: vec![] });
            cur = Some(key);
        } else if line == "@@END" {
            complete = true;
            break;
        } else if let Some(st) = cur.as_ref().and_then(|k| out.get_mut(k)) {
            if line.starts_with("STATE ") {
                let parts = py::split(line);
                let Some(state) = parts.get(1) else { continue };
                st.state = state.to_string();
                if st.state == "done" {
                    st.rc = Some(parts.get(2).and_then(|p| py::int(p)).unwrap_or(1));
                }
            } else if let Some(size) = line.strip_prefix("SIZE ") {
                let size = py::strip(size);
                st.size = if size.is_empty() { 0 } else { py::int(size).unwrap_or(0) };
            } else if line.starts_with("DATA") {
                let data = py::strip(line.get(5..).unwrap_or(""));
                st.data = base64::engine::general_purpose::STANDARD.decode(data).ok()?;
            }
        }
    }
    complete.then_some(out)
}

/// SIGTERM the job's process group, then SIGKILL it after `grace` seconds if it is still alive (in
/// a detached background step, so this returns at once). Prints KILLED, NOT_RUNNING or NO_JOB.
pub fn kill_script(key: &str, grace: u32) -> String {
    format!(
        r#"D="{JOBS_DIR}/{key}"
[ -f "$D/pid" ] || {{ echo NO_JOB; exit 0; }}
P=$(cat "$D/pid")
if kill -TERM -- -"$P" 2>/dev/null; then
  echo KILLED
  ( trap '' HUP; i=0
    while [ $i -lt {grace} ]; do sleep 1; kill -0 -- -"$P" 2>/dev/null || exit 0; i=$((i+1)); done
    kill -KILL -- -"$P" 2>/dev/null ) > /dev/null 2>&1 < /dev/null &
else
  echo NOT_RUNNING
fi
"#
    )
}

/// Remove job directories for every launch attempt of the given tasks.
pub fn cleanup_script(task_ids: &[i64]) -> String {
    let globs: Vec<String> = task_ids.iter().map(|t| format!("\"$J\"/{t}-*")).collect();
    format!("J=\"{JOBS_DIR}\"\nrm -rf {}\n", globs.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn bash(script: &str, home: &std::path::Path) -> String {
        let out = Command::new("bash")
            .arg("-s")
            .env("HOME", home)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .and_then(|mut c| {
                use std::io::Write;
                c.stdin.take().unwrap().write_all(script.as_bytes())?;
                c.wait_with_output()
            })
            .unwrap();
        String::from_utf8(out.stdout).unwrap()
    }

    #[test]
    fn job_key_shape() {
        let k = new_job_key(12);
        assert!(k.starts_with("12-") && k.len() == 9 && k[3..].bytes().all(|b| b.is_ascii_hexdigit()), "{k}");
        assert_ne!(new_job_key(1), new_job_key(1));
    }

    #[test]
    fn launch_runs_detached_and_status_reads_it_back() {
        let home = tempfile::tempdir().unwrap();
        let wdir = home.path().join("it's here");
        std::fs::create_dir(&wdir).unwrap();
        let out = bash(
            &launch_script("1-abcdef", wdir.to_str().unwrap(), "0", "echo 'a b'; pwd; exit 3", false, None),
            home.path(),
        );
        assert!(parse_launch(&out).is_some(), "{out}");
        let dir = home.path().join(".gpu-spooler/jobs/1-abcdef");
        for _ in 0..100 {
            if dir.join("rc").exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let st = parse_status(&bash(&status_script(&[("1-abcdef".into(), 0), ("2-gone00".into(), 0)]), home.path()))
            .unwrap();
        assert_eq!(st["1-abcdef"].state, "done");
        assert_eq!(st["1-abcdef"].rc, Some(3));
        let log = String::from_utf8(st["1-abcdef"].data.clone()).unwrap();
        assert_eq!(log, format!("a b\n{}\n", wdir.display()));
        assert_eq!(st["1-abcdef"].size as usize, log.len());
        assert_eq!(st["2-gone00"].state, "missing");
        // From an offset: only the new bytes.
        let st = parse_status(&bash(&status_script(&[("1-abcdef".into(), 4)]), home.path())).unwrap();
        assert_eq!(st["1-abcdef"].data, &log.as_bytes()[4..]);
        bash(&cleanup_script(&[1]), home.path());
        assert!(!dir.exists());
    }
}
