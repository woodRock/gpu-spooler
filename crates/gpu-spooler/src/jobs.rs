//! Detached jobs.
//!
//! Each launch attempt gets a job directory on the host that runs it,
//! $HOME/.gpu-spooler/jobs/<job_key>/, holding run.sh, pid, out.log and (once the command exits)
//! rc. Jobs are started in their own session, so they keep running when the ssh connection that
//! launched them goes away; the daemon polls the directory to sync the log and pick up the exit
//! code. The scripts are the Python version's byte for byte, so either daemon can adopt the
//! other's jobs.

use crate::config::Host;
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

/// The lines that write taura's probe into the job directory, and the line that runs it.
fn probe_steps(tag: &str, probe: Option<&str>) -> (String, String) {
    match probe {
        Some(probe) => {
            let peof = format!("TS_PROBE_{tag}");
            (
                format!("cat > \"$D/taura_probe.py\" <<'{peof}'\n{}\n{peof}\n", PROBE.trim_matches('\n')),
                format!("{probe} \"$D/taura_probe.py\" 2>/dev/null | sed \"s/^/{ENV_MARK}/\" || true\n"),
            )
        }
        None => (String::new(), String::new()),
    }
}

/// Lines that print the commit the job will run: see [`parse_git`] and [`parse_dirty`].
fn git_report(q_wdir: &str) -> String {
    format!(
        r#"# The commit the job will run, for run provenance (taura): HEAD and uncommitted changes.
# git may only be on a login shell's PATH (on ECS it's in /usr/pkg/bin).
{FIND_GIT}( cd {q_wdir} 2>/dev/null && h=$(git rev-parse -q --verify HEAD 2>/dev/null)   && echo "TS_GIT $h $(git status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')"   && git status --porcelain --untracked-files=no 2>/dev/null | head -50 | sed 's/^.../TS_DIRTY /' )
"#
    )
}

/// Script (for `bash -s`) that starts a job detached and prints `TS_PID <pid>`.
///
/// `probe`: when the task is recorded with taura, how to run taura's environment probe like the
/// job's command (e.g. `PYTHONPATH=src python3`); the job then prints a `TAURA_ENV {...}` line first.
pub fn launch_script(key: &str, wdir: &str, gpus: &str, command: &str, delay: bool, probe: Option<&str>) -> String {
    let q_wdir = sh_path(wdir);
    let tag = key.replace('-', "_");
    let eof = format!("TS_EOF_{tag}");
    let (probe_file, probe_run) = probe_steps(&tag, probe);
    let gpus = quote(gpus);
    let pause = if delay { "sleep 2" } else { ":" };
    let git = git_report(&q_wdir);
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
{git}ts_detach bash -l "$D/run.sh" > "$D/out.log" 2>&1 < /dev/null &
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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct JobStatus {
    /// running | done | missing | gone
    pub state: String,
    pub rc: Option<i64>,
    pub size: i64,
    pub data: Vec<u8>,
    /// SGE jobs: the job's ID and its state in qstat, while qstat lists it
    pub sge: Option<(i64, String)>,
    /// SGE jobs: when the job began running on a grid machine
    pub started: Option<f64>,
    /// SGE jobs: which machine
    pub machine: Option<String>,
    /// SGE jobs: the commit report from that machine, as [`parse_git`] and [`parse_dirty`] read it
    pub git: String,
    /// SGE jobs: seconds it has spent suspended
    pub suspended: Option<f64>,
    /// SGE jobs: when the command ended, which the next poll may be well after
    pub finished: Option<f64>,
}

/// Parse status_script output; None if it was cut short.
pub fn parse_status(stdout: &str) -> Option<IndexMap<String, JobStatus>> {
    let mut out: IndexMap<String, JobStatus> = IndexMap::new();
    let mut cur: Option<String> = None;
    let mut complete = false;
    for line in py::splitlines(stdout) {
        if let Some(key) = line.strip_prefix("@@JOB ") {
            let key = py::strip(key).to_string();
            out.insert(key.clone(), JobStatus { state: "gone".into(), ..Default::default() });
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
            } else if line.starts_with("SGE ") {
                let parts = py::split(line);
                if let (Some(id), Some(state)) = (parts.get(1).and_then(|p| py::int(p)), parts.get(2)) {
                    st.sge = Some((id, state.to_string()));
                }
            } else if line.starts_with("STARTED ") {
                let parts = py::split(line);
                st.started = parts.get(1).and_then(|ts| ts.parse().ok());
                st.machine = parts.get(2).map(|m| m.to_string());
            } else if let Some(ts) = line.strip_prefix("FINISHED ") {
                st.finished = py::strip(ts).parse().ok();
            } else if let Some(secs) = line.strip_prefix("SUSPENDED ") {
                st.suspended = py::strip(secs).parse().ok();
            } else if line.starts_with("TS_GIT ") || line.starts_with("TS_DIRTY ") {
                st.git += &format!("{line}\n");
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

// ── Sun Grid Engine ───────────────────────────────────────────────────────────

/// The first 32 lines of the ECS Grid submission template. The tech note says to change nothing
/// here but the username: https://ecs.wgtn.ac.nz/Support/TechNoteEcsGrid
fn sge_template(user: &str, wd: &str) -> String {
    format!(
        r#"#!/bin/sh
#
# Force Bourne Shell if not Sun Grid Engine default shell (you never know!)
#
#$ -S /bin/sh
#
# I know I have a directory here so I'll use it as my initial working directory
#
#$ -wd {wd}
#
# End of the setup directives
#
# Now let's do something useful, but first change into the job-specific directory that should
#  have been created for us
#
# Check we have somewhere to work now and if we don't, exit nicely.
#
if [ -d /local/tmp/{user}/$JOB_ID ]; then
        cd /local/tmp/{user}/$JOB_ID
else
        echo "Uh oh ! There's no job directory to change into "
        echo "Something is broken. I should inform the programmers"
        echo "Save some information that may be of use to them"
        echo "Here's LOCAL TMP "
        ls -la /local/tmp
        echo "AND LOCAL TMP {user} "
        ls -la /local/tmp/{user}
        echo "Exiting"
        exit 1
fi
#
# Now we are in the job-specific directory so now can do something useful
"#
    )
}

/// Seconds between the job's looks at the clock, see [`sge_watch`].
const SGE_WATCH_SECS: u32 = 5;

/// Lines that keep the job's suspended seconds in $D/suspended, in the background ($ts_watch).
/// SGE suspends a job by stopping its processes, this loop included, so the time it loses is the
/// time the job was stopped, less up to `every` seconds for each stop (a stopped `sleep` still
/// runs out). qacct would say this, but the ECS Grid keeps no accounting file.
fn sge_watch(every: u32) -> String {
    let least = every / 2 + 1; // less is a busy machine, not a stop
    format!(
        r#"( ts_last=$(date +%s); ts_off=0
  while sleep {every}; do
    ts_now=$(date +%s); ts_gap=$((ts_now - ts_last - {every})); ts_last=$ts_now
    if [ "$ts_gap" -ge {least} ]; then ts_off=$((ts_off + ts_gap)); echo "$ts_off" > "$D/suspended"; fi
  done ) > /dev/null 2>&1 < /dev/null &
ts_watch=$!
disown "$ts_watch"
"#
    )
}

/// Script (for `bash -s` on a submit host) that hands a job to SGE and prints `TS_SGE_JOB <id>`.
/// The job directory is on NFS, so the submit host can follow the job like any other. The command
/// runs in `wdir`, with the machine's scratch directory (deleted after the job) in $TASK_SCRATCH.
/// The job can wait in the queue for hours, so what a GPU job reports at launch this one reports
/// when it starts: `started` holds the time and machine, once `git` holds the commit it will run.
/// A job SGE restarts runs all of this again.
pub fn sge_submit_script(host: &Host, key: &str, wdir: &str, command: &str, probe: Option<&str>) -> String {
    let q_wdir = sh_path(wdir);
    let tag = key.replace('-', "_");
    let eof = format!("TS_EOF_{tag}");
    let job_eof = format!("TS_JOB_{tag}");
    let (probe_file, probe_run) = probe_steps(&tag, probe);
    let git = git_report(&q_wdir);
    let user = host.user.as_deref().unwrap_or_default();
    let template = sge_template(user, &host.sge_wd);
    let (settings, queue) = (sh_path(&host.sge_settings), quote(&host.sge_queue));
    let watch = sge_watch(SGE_WATCH_SECS);
    format!(
        r#". {settings} || exit 1
D="{JOBS_DIR}/{key}"
mkdir -p "$D" || exit 1
{probe_file}cat > "$D/run.sh" <<'{eof}'
D="{JOBS_DIR}/{key}"
ts_started=$(date +%s)
ts_machine=$(hostname -s 2>/dev/null || hostname)
ts_report() {{ echo "$ts_started $ts_machine" > "$D/started.tmp" && mv "$D/started.tmp" "$D/started"; }}
rm -f "$D/finished" "$D/suspended"
source ~/.bashrc 2>/dev/null || true
ls {q_wdir} > /dev/null 2>&1
cd {q_wdir} || {{ ts_report; echo "gpu-spooler: cannot cd to "{q_wdir} >&2; date +%s > "$D/finished"; echo 1 > "$D/rc.tmp"; mv "$D/rc.tmp" "$D/rc"; exit 1; }}
export TASK_SCRATCH="/local/tmp/{user}/$JOB_ID"
(
{git}) > "$D/git" 2>/dev/null
ts_report
{watch}{probe_run}( {command}
)
rc=$?
kill "$ts_watch" 2>/dev/null
date +%s > "$D/finished"
echo "$rc" > "$D/rc.tmp" && mv "$D/rc.tmp" "$D/rc"
exit $rc
{eof}
cat > "$D/job.sh" <<'{job_eof}'
{template}bash -l "{JOBS_DIR}/{key}/run.sh"
{job_eof}
{git}O=$(qsub -N ts-{key} -q {queue} -j y -o "$D/out.log" "$D/job.sh") || exit 1
J=$(printf '%s\n' "$O" | sed -n 's/^Your job \([0-9][0-9]*\).*/\1/p')
[ -n "$J" ] || {{ echo "gpu-spooler: unexpected qsub output: $O" >&2; exit 1; }}
echo "$J" > "$D/sgeid"
echo "TS_SGE_JOB $J"
"#
    )
}

/// The SGE job ID the submit script printed.
pub fn parse_sge_submit(stdout: &str) -> Option<i64> {
    let line = py::splitlines(stdout).into_iter().find(|l| l.starts_with("TS_SGE_JOB "))?;
    py::split(line).get(1).and_then(|p| py::int(p))
}

/// [`status_script`] for SGE jobs: a job is alive while qstat lists it. Fails if qstat does, as
/// the state of the jobs is then unknown.
pub fn sge_status_script(host: &Host, specs: &[(String, i64)]) -> String {
    let items: Vec<String> = specs.iter().map(|(k, off)| quote(&format!("{k}:{off}"))).collect();
    format!(
        r#". {settings} || exit 1
Q=$(qstat -u "$(id -un)") || exit 1
for spec in {items}; do
  k=${{spec%%:*}}; off=${{spec##*:}}; D="{JOBS_DIR}/$k"
  printf '@@JOB %s\n' "$k"
  J=$(cat "$D/sgeid" 2>/dev/null)
  S=""
  [ -z "$J" ] || S=$(printf '%s\n' "$Q" | awk -v j="$J" '$1 == j {{ print $5; exit }}')
  if [ -f "$D/rc" ]; then printf 'STATE done %s\n' "$(cat "$D/rc")"
  elif [ ! -d "$D" ]; then echo 'STATE missing'
  elif [ -n "$S" ]; then echo 'STATE running'
  else echo 'STATE gone'; fi
  [ -z "$S" ] || printf 'SGE %s %s\n' "$J" "$S"
  [ ! -f "$D/started" ] || printf 'STARTED %s\n' "$(cat "$D/started")"
  [ ! -f "$D/git" ] || cat "$D/git"
  [ ! -f "$D/suspended" ] || printf 'SUSPENDED %s\n' "$(cat "$D/suspended")"
  [ ! -f "$D/finished" ] || printf 'FINISHED %s\n' "$(cat "$D/finished")"
  printf 'SIZE %s\n' "$(wc -c < "$D/out.log" 2>/dev/null | tr -d ' ' || echo 0)"
  printf 'DATA '; tail -c +$((off+1)) "$D/out.log" 2>/dev/null | head -c {CHUNK_BYTES} | base64 | tr -d '\n'; echo
done
echo '@@END'
"#,
        settings = sh_path(&host.sge_settings),
        items = items.join(" ")
    )
}

/// Delete the job from SGE. Prints KILLED, NOT_RUNNING or NO_JOB.
pub fn sge_kill_script(host: &Host, key: &str) -> String {
    format!(
        r#". {settings} || exit 1
J=$(cat "{JOBS_DIR}/{key}/sgeid" 2>/dev/null)
[ -n "$J" ] || {{ echo NO_JOB; exit 0; }}
Q=$(qstat -u "$(id -un)") || exit 1
printf '%s\n' "$Q" | awk -v j="$J" '$1 == j {{ found = 1 }} END {{ exit !found }}' || {{ echo NOT_RUNNING; exit 0; }}
qdel "$J" > /dev/null || exit 1
echo KILLED
"#,
        settings = sh_path(&host.sge_settings)
    )
}

/// What a qstat state means, where the task's status doesn't say: `r` needs no words.
pub fn sge_note(state: &str) -> Option<&'static str> {
    if state.contains('E') {
        Some("error")
    } else if state.contains(['s', 'S', 'T']) {
        Some("suspended")
    } else if state.contains('h') {
        Some("held")
    } else if state.contains('q') {
        Some("waiting")
    } else {
        None
    }
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

    fn sge_host(settings: &std::path::Path) -> Host {
        let text = format!(
            "[hosts.grid]\nkind = \"sge\"\nuser = \"fred\"\nsge_settings = {:?}\nsge_wd = \"/vol/fred\"\n",
            settings.display()
        );
        let home = std::path::Path::new("/nowhere");
        let data = toml::from_str(&text).unwrap();
        crate::config::parse_config(&data, &crate::paths::Paths::for_home(home)).unwrap().hosts["grid"].clone()
    }

    #[test]
    fn sge_script_keeps_the_tech_note_template() {
        let host = sge_host(std::path::Path::new("/s.sh"));
        let script = sge_submit_script(&host, "1-abcdef", "/w", "echo hi", None);
        let job = script.split("<<'TS_JOB_1_abcdef'\n").nth(1).unwrap();
        let lines: Vec<&str> = job.lines().collect();
        assert_eq!(lines[0], "#!/bin/sh");
        assert_eq!(lines[8], "#$ -wd /vol/fred");
        assert_eq!(lines[17], "if [ -d /local/tmp/fred/$JOB_ID ]; then");
        assert_eq!(lines[26], "        ls -la /local/tmp/fred");
        assert_eq!(lines[31], "# Now we are in the job-specific directory so now can do something useful");
    }

    #[test]
    fn sge_status_follows_qstat() {
        let home = tempfile::tempdir().unwrap();
        let settings = home.path().join("settings.sh");
        let qstat = "job-ID prior name user state\n---\n 77 0.5 ts-1 fred qw\n 78 0.5 ts-2 fred Eqw\n";
        std::fs::write(&settings, format!("qstat() {{ printf '%s' '{qstat}'; return ${{QS:-0}}; }}\n")).unwrap();
        let host = sge_host(&settings);
        let jobs = home.path().join(".gpu-spooler/jobs");
        for (key, id) in [("1-aaaaaa", "77"), ("2-bbbbbb", "78"), ("3-cccccc", "79"), ("4-dddddd", "80")] {
            std::fs::create_dir_all(jobs.join(key)).unwrap();
            std::fs::write(jobs.join(key).join("sgeid"), format!("{id}\n")).unwrap();
        }
        std::fs::write(jobs.join("1-aaaaaa/out.log"), "hi\n").unwrap();
        std::fs::write(jobs.join("1-aaaaaa/started"), "1700000000 big-thumb\n").unwrap();
        std::fs::write(jobs.join("1-aaaaaa/git"), "TS_GIT abc 1\nTS_DIRTY results/a.json\n").unwrap();
        std::fs::write(jobs.join("1-aaaaaa/suspended"), "120\n").unwrap();
        std::fs::write(jobs.join("2-bbbbbb/started"), "1700000000\n").unwrap(); // from an older version
        std::fs::write(jobs.join("4-dddddd/rc"), "5\n").unwrap();
        let specs: Vec<(String, i64)> =
            ["1-aaaaaa", "2-bbbbbb", "3-cccccc", "4-dddddd", "5-eeeeee"].iter().map(|k| (k.to_string(), 0)).collect();
        let st = parse_status(&bash(&sge_status_script(&host, &specs), home.path())).unwrap();
        assert_eq!(st["1-aaaaaa"].state, "running");
        assert_eq!(st["1-aaaaaa"].sge, Some((77, "qw".to_string())));
        assert_eq!(st["1-aaaaaa"].started, Some(1700000000.0));
        assert_eq!(st["1-aaaaaa"].machine.as_deref(), Some("big-thumb"));
        assert_eq!(parse_git(&st["1-aaaaaa"].git), Some(("abc".to_string(), 1)));
        assert_eq!(parse_dirty(&st["1-aaaaaa"].git), ["results/a.json"]);
        assert_eq!(st["1-aaaaaa"].suspended, Some(120.0));
        assert_eq!(st["2-bbbbbb"].started, Some(1700000000.0));
        assert_eq!(st["2-bbbbbb"].machine, None);
        assert_eq!(st["1-aaaaaa"].data, b"hi\n");
        assert_eq!(st["2-bbbbbb"].sge, Some((78, "Eqw".to_string())));
        assert_eq!(st["3-cccccc"].state, "gone");
        assert_eq!((st["4-dddddd"].state.as_str(), st["4-dddddd"].rc), ("done", Some(5)));
        assert_eq!(st["5-eeeeee"].state, "missing");

        // qstat failing says nothing about the jobs.
        let broken = sge_status_script(&host, &specs).replacen(". ", "QS=1; . ", 1);
        assert_eq!(parse_status(&bash(&broken, home.path())), None);
    }

    #[test]
    fn sge_watch_counts_only_stopped_time() {
        use std::os::unix::process::CommandExt;
        let dir = tempfile::tempdir().unwrap();
        let script = format!("D={:?}\n{}sleep 8\nkill \"$ts_watch\"\n", dir.path().display(), sge_watch(2));
        let mut child = Command::new("bash").arg("-c").arg(script).process_group(0).spawn().unwrap();
        let group = -(child.id() as libc::pid_t);
        let signal = |sig| assert_eq!(unsafe { libc::kill(group, sig) }, 0);
        std::thread::sleep(std::time::Duration::from_secs(2));
        assert!(!dir.path().join("suspended").exists(), "nothing stopped it yet");
        signal(libc::SIGSTOP);
        std::thread::sleep(std::time::Duration::from_secs(4));
        signal(libc::SIGCONT);
        child.wait().unwrap();
        let secs: i64 = std::fs::read_to_string(dir.path().join("suspended")).unwrap().trim().parse().unwrap();
        assert!((2..=6).contains(&secs), "{secs}");
    }

    #[test]
    fn sge_notes() {
        let notes: Vec<_> = ["r", "qw", "hqw", "Eqw", "s", "S", "T", "dr"].iter().map(|s| sge_note(s)).collect();
        assert_eq!(
            notes,
            [
                None,
                Some("waiting"),
                Some("held"),
                Some("error"),
                Some("suspended"),
                Some("suspended"),
                Some("suspended"),
                None
            ]
        );
    }
}
