//! `task gpus`: query every configured host in parallel, run nvidia-smi, and show free/busy GPUs,
//! users and processes. The tables match the Python `gpu-status` character for character
//! (padding counts the ANSI codes too, as Python's format() does).

use crate::color::{colorize, BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW};
use gpu_spooler::config::Config;
use gpu_spooler::gpu::{gpu_model, gpu_type_matches, query_host, GpuProcess, ServerResult};
use gpu_spooler::idle::{fmt_idle, idle_gpus, Idle, IDLE_UTIL_PCT};
use gpu_spooler::paths::Paths;
use gpu_spooler::py;
use gpu_spooler::transport::{now, Transport};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Write};
use std::sync::Mutex;

#[derive(clap::Args, Debug)]
pub struct Args {
    /// One row per GPU instead of one per server
    #[arg(short, long)]
    pub detail: bool,
    /// Only servers / GPUs with free capacity
    #[arg(short, long)]
    pub free: bool,
    /// Hosts, groups or tags to query (comma separated)
    #[arg(short = 'H', long, value_name = "SPEC")]
    pub hosts: Option<String>,
    /// Parallel SSH connections
    #[arg(long, value_name = "N", default_value_t = 16)]
    pub workers: usize,
    /// Print raw SSH output for each server
    #[arg(long)]
    pub debug: bool,
}

fn mb_to_gb(mb: i64) -> String {
    format!("{:.1}", mb as f64 / 1024.0)
}

/// (host, configured gpu_type, what nvidia-smi reports) where the two disagree.
pub fn label_mismatches(results: &[ServerResult], cfg: &Config) -> Vec<(String, String, String)> {
    let mut sorted: Vec<&ServerResult> = results.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let mut out = vec![];
    for r in sorted {
        let Some(configured) = cfg.hosts.get(&r.name).and_then(|h| h.gpu_type.as_deref()).filter(|t| !t.is_empty())
        else {
            continue;
        };
        if !r.reachable || r.gpus.is_empty() {
            continue;
        }
        let models = r.models();
        if !models.iter().all(|m| gpu_type_matches(Some(configured), m)) {
            out.push((r.name.clone(), configured.to_string(), models.join(", ")));
        }
    }
    out
}

fn held_idle(idle: Option<&Idle>, host: &str) -> usize {
    idle.map_or(0, |i| i.keys().filter(|(h, _)| h == host).count())
}

/// Unreachable hosts last, then by name.
fn by_reach_then_name(results: &[ServerResult]) -> Vec<&ServerResult> {
    let mut sorted: Vec<&ServerResult> = results.iter().collect();
    sorted.sort_by(|a, b| (!a.reachable, &a.name).cmp(&(!b.reachable, &b.name)));
    sorted
}

/// One row per server. GPU TYPE is what nvidia-smi reports (the configured gpu_type only for hosts
/// that reported no GPUs); * marks a configured gpu_type that disagrees. With gpu-detective's
/// history (`idle`), hosts with GPUs held but idle say how many.
pub fn print_summary_table(
    out: &mut dyn Write,
    results: &[ServerResult],
    cfg: &Config,
    free_only: bool,
    idle: Option<&Idle>,
) -> io::Result<()> {
    let mismatched: BTreeSet<String> = label_mismatches(results, cfg).into_iter().map(|m| m.0).collect();
    let header = format!(
        "{:<14} {:<24} {:>4}  {:>4}  {:>10}  {:>10}  {}",
        "SERVER", "GPU TYPE", "GPUs", "FREE", "VRAM FREE", "VRAM TOTAL", "USERS / PROCESSES"
    );
    let sep = "─".repeat(header.chars().count());
    writeln!(out, "\n{BOLD}{header}{RESET}")?;
    writeln!(out, "{sep}")?;

    for r in by_reach_then_name(results) {
        if free_only && r.reachable && r.free_gpus() == 0 {
            continue;
        }
        let configured = cfg.hosts.get(&r.name).and_then(|h| h.gpu_type.clone()).filter(|t| !t.is_empty());
        let mut gpu_type = r.models().join("/");
        if gpu_type.is_empty() {
            gpu_type = configured.unwrap_or_else(|| "-".into());
        }
        if mismatched.contains(&r.name) {
            gpu_type.push('*');
        }

        if !r.reachable {
            writeln!(out, "{:<14} {:<33}  {}", r.name, colorize("UNREACHABLE", RED), py::head(&r.error, 60))?;
            continue;
        }

        let (total, free) = (r.total_gpus(), r.free_gpus());
        let color = if free == total {
            GREEN
        } else if free == 0 {
            RED
        } else {
            YELLOW
        };
        let free_str = colorize(&free.to_string(), color);

        // Unique users and their processes
        let mut user_procs: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for g in &r.gpus {
            for p in &g.processes {
                user_procs.entry(&p.user).or_default().insert(&p.command);
            }
        }
        let mut who = user_procs
            .iter()
            .map(|(u, cmds)| format!("{}: {}", colorize(u, CYAN), cmds.iter().copied().collect::<Vec<_>>().join(", ")))
            .collect::<Vec<_>>()
            .join("; ");
        if who.is_empty() {
            who = colorize("idle", DIM);
        }
        let n_idle = held_idle(idle, &r.name);
        if n_idle > 0 {
            who = colorize(&format!("[{n_idle} held idle] "), YELLOW) + &who;
        }

        let vram_free = format!("{:>5} GB", mb_to_gb(r.free_vram_mb()));
        let vram_total = format!("{:>5} GB", mb_to_gb(r.total_vram_mb()));
        writeln!(
            out,
            "{:<14} {:<24} {:>4}  {:>12}  {:>10}  {:>10}  {}",
            r.name, gpu_type, total, free_str, vram_free, vram_total, who
        )?;
    }

    writeln!(out, "{sep}")?;
    print_label_warnings(out, results, cfg)
}

pub fn print_label_warnings(out: &mut dyn Write, results: &[ServerResult], cfg: &Config) -> io::Result<()> {
    let rows = label_mismatches(results, cfg);
    if rows.is_empty() {
        return Ok(());
    }
    writeln!(out, "{}", colorize(&format!("* gpu_type in {} disagrees with nvidia-smi:", cfg.source_label()), YELLOW))?;
    for (name, configured, live) in rows {
        writeln!(out, "{}", colorize(&format!("    {name:<14} configured '{configured}', reports '{live}'"), YELLOW))?;
    }
    writeln!(
        out,
        "{}",
        colorize(
            "  --gpu-type and -m use what nvidia-smi reports; update or remove these gpu_type (and \
             gpu_mem_gb/num_gpus) entries.",
            YELLOW
        )
    )
}

/// One row per GPU. GPUs held but idle (from gpu-detective's history) are marked IDLE.
pub fn print_detail_table(
    out: &mut dyn Write,
    results: &[ServerResult],
    free_only: bool,
    idle: Option<&Idle>,
) -> io::Result<()> {
    let header = format!(
        "{:<14} {:>3}  {:<24}  {:>9}  {:>7}  {:>5}  {:<8}  {:<12}  {}",
        "SERVER", "GPU", "GPU NAME", "MEM USED", "MEM TOT", "UTIL", "STATUS", "USER", "PROCESS / SCRIPT"
    );
    let sep = "─".repeat(header.chars().count());
    writeln!(out, "\n{BOLD}{header}{RESET}")?;
    writeln!(out, "{sep}")?;

    for r in by_reach_then_name(results) {
        if !r.reachable {
            writeln!(out, "{:<14} {}  {}", r.name, colorize("UNREACHABLE", RED), py::head(&r.error, 60))?;
            continue;
        }
        let mut gpus: Vec<_> = r.gpus.iter().collect();
        gpus.sort_by_key(|g| g.index);
        for g in gpus {
            if free_only && !g.is_free() {
                continue;
            }
            let mem_used = format!("{:>6} GB", mb_to_gb(g.mem_used_mb));
            let mem_tot = format!("{:>4} GB", mb_to_gb(g.mem_total_mb));
            let util = if g.utilization_pct >= 0 { format!("{:>3}%", g.utilization_pct) } else { "  N/A".into() };
            let model = gpu_model(&g.name);
            let lead =
                format!("{:<14} {:>3}  {:<24}  {:>9}  {:>7}  {:>5}  ", r.name, g.index, model, mem_used, mem_tot, util);

            if g.is_free() {
                writeln!(out, "{lead}{:<16}  {:<12}  ", colorize("FREE", GREEN), "")?;
                continue;
            }
            let ig = idle.and_then(|i| i.get(&(r.name.clone(), g.index)));
            let status = if ig.is_some() { colorize("IDLE", YELLOW) } else { colorize("IN USE", RED) };
            // One line per process (the first on the GPU's row)
            let unknown = [GpuProcess { pid: 0, user: "?".into(), mem_used_mb: 0, command: "?".into() }];
            let procs = if g.processes.is_empty() { &unknown[..] } else { &g.processes[..] };
            for (i, p) in procs.iter().enumerate() {
                let user = colorize(&p.user, CYAN);
                if i == 0 {
                    let note = ig.map_or(String::new(), |ig| colorize(&format!("  ({})", fmt_idle(ig)), YELLOW));
                    writeln!(out, "{lead}{status:<16}  {user:<20}  {}{note}", p.command)?;
                } else {
                    writeln!(
                        out,
                        "{:14} {:>3}  {:24}  {:>9}  {:>7}  {:>5}  {:16}  {user:<20}  {}",
                        "", "", "", "", "", "", "", p.command
                    )?;
                }
            }
        }
    }
    writeln!(out, "{sep}")
}

pub fn print_totals(
    out: &mut dyn Write,
    results: &[ServerResult],
    idle: Option<&Idle>,
    idle_minutes: i64,
) -> io::Result<()> {
    let reachable: Vec<&ServerResult> = results.iter().filter(|r| r.reachable).collect();
    let total_gpus: usize = reachable.iter().map(|r| r.total_gpus()).sum();
    let free_gpus: usize = reachable.iter().map(|r| r.free_gpus()).sum();
    let total_vram: i64 = reachable.iter().map(|r| r.total_vram_mb()).sum();
    let free_vram: i64 = reachable.iter().map(|r| r.free_vram_mb()).sum();
    let unreachable = results.len() - reachable.len();
    let unreachable = if unreachable > 0 { colorize(&unreachable.to_string(), RED) } else { colorize("0", GREEN) };
    let idle_part = idle.map_or(String::new(), |i| {
        let color = if i.is_empty() { GREEN } else { YELLOW };
        format!("  |  {} GPUs held idle {idle_minutes}m+ (gpu-detective)", colorize(&i.len().to_string(), color))
    });
    writeln!(
        out,
        "\n{BOLD}Totals:{RESET}  {} / {total_gpus} GPUs free  |  {} / {} GB VRAM free  |  {unreachable} servers \
         unreachable{idle_part}\n",
        colorize(&free_gpus.to_string(), GREEN),
        mb_to_gb(free_vram),
        mb_to_gb(total_vram),
    )
}

/// Keep only what this query agrees with: the history can be a sweep behind, and a GPU that is
/// busy again, or held by someone else, now is not idle.
pub fn confirm_idle(idle: Option<Idle>, results: &[ServerResult]) -> Option<Idle> {
    let idle = idle?;
    let live: BTreeMap<(String, i64), _> = results
        .iter()
        .filter(|r| r.reachable)
        .flat_map(|r| r.gpus.iter().map(|g| ((r.name.clone(), g.index), g)))
        .collect();
    Some(
        idle.into_iter()
            .filter(|(key, ig)| {
                live.get(key).is_some_and(|g| {
                    let users: BTreeSet<&str> = g.processes.iter().map(|p| p.user.as_str()).collect();
                    (0..IDLE_UTIL_PCT).contains(&g.utilization_pct)
                        && users.into_iter().collect::<Vec<_>>()
                            == ig.users.iter().map(String::as_str).collect::<Vec<_>>()
                })
            })
            .collect(),
    )
}

/// Idle-held GPUs from gpu-detective's history; None when [detective] is off.
pub fn load_idle(out: &mut dyn Write, cfg: &Config) -> io::Result<Option<Idle>> {
    let Some(path) = cfg.detective_path() else { return Ok(None) };
    if !path.exists() {
        let msg = format!("[detective] is on but {} doesn't exist: run `detective watch`.", path.display());
        writeln!(out, "{}", colorize(&msg, YELLOW))?;
        return Ok(None);
    }
    Ok(Some(idle_gpus(&path, (cfg.idle_minutes * 60) as f64, now())))
}

/// `task gpus`. Exit code 1 (with `gpu-status: ...` on stderr) for a config or -H problem.
pub fn run(args: &Args) -> i32 {
    let paths = Paths::from_env();
    let loaded = gpu_spooler::config::load_config(&paths).and_then(|cfg| {
        cfg.select(args.hosts.as_deref())
            .map(|t| t.into_iter().map(|h| h.name.clone()).collect::<Vec<_>>())
            .map(|t| (cfg, t))
    });
    let (cfg, targets) = match loaded {
        Ok(x) => x,
        Err(e) => {
            eprintln!("gpu-status: {e}");
            return 1;
        }
    };
    let transport = Transport::new(cfg, paths);
    let cfg = &transport.cfg;
    let hosts: Vec<_> = targets.iter().map(|n| &cfg.hosts[n]).collect();
    let mut stdout = io::stdout();
    let res = (|| -> io::Result<()> {
        writeln!(
            stdout,
            "{BOLD}Querying {} server(s)...{RESET}  {DIM}network: {}{RESET}",
            hosts.len(),
            transport.describe_mode(true)
        )?;
        let done = Mutex::new(0usize);
        let results: Vec<ServerResult> = transport
            .fan_out(
                &hosts,
                |host| {
                    let r = query_host(&transport, host, args.debug);
                    let mut n = done.lock().unwrap();
                    *n += 1;
                    let status =
                        if r.reachable { "ok".to_string() } else { format!("ERR: {}", py::head(&r.error, 60)) };
                    println!("  [{:>2}/{}] {:<16} {status}", *n, hosts.len(), host.name);
                    r
                },
                args.workers,
            )
            .into_values()
            .collect();
        let idle = confirm_idle(load_idle(&mut stdout, cfg)?, &results);
        if args.detail {
            print_detail_table(&mut stdout, &results, args.free, idle.as_ref())?;
            print_label_warnings(&mut stdout, &results, cfg)?;
        } else {
            print_summary_table(&mut stdout, &results, cfg, args.free, idle.as_ref())?;
        }
        print_totals(&mut stdout, &results, idle.as_ref(), cfg.idle_minutes)
    })();
    match res {
        Ok(()) => 0,
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => 0,
        Err(e) => {
            eprintln!("gpu-status: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_spooler::idle::IdleGpu;
    use serde_json::Value;
    use std::path::PathBuf;

    fn golden(name: &str) -> String {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/parity").join(name);
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
    }

    /// The fixture's config and results, with its {HOME} placeholder as the home folder.
    fn setup() -> (Config, Vec<ServerResult>) {
        let cases: Value = serde_json::from_str(&golden("gpu_status.json")).unwrap();
        let input = &cases[0]["input"];
        let paths = Paths::for_home("{HOME}");
        let mut cfg =
            gpu_spooler::config::parse_config(&input["config"].as_str().unwrap().parse().unwrap(), &paths).unwrap();
        cfg.source = Some(paths.config.clone());
        let results = serde_json::from_value(input["results"].clone()).unwrap();
        (cfg, results)
    }

    fn capture(f: impl FnOnce(&mut dyn Write) -> io::Result<()>) -> String {
        let mut buf = Vec::new();
        f(&mut buf).unwrap();
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn tables_match_python() {
        let (cfg, results) = setup();
        for free in [false, true] {
            let sfx = if free { ".free" } else { "" };
            let got = capture(|o| print_summary_table(o, &results, &cfg, free, None));
            assert_eq!(got, golden(&format!("gpu_status/summary{sfx}.txt")), "summary{sfx}");
            let got = capture(|o| print_detail_table(o, &results, free, None));
            assert_eq!(got, golden(&format!("gpu_status/detail{sfx}.txt")), "detail{sfx}");
        }
        assert_eq!(capture(|o| print_label_warnings(o, &results, &cfg)), golden("gpu_status/label_warnings.txt"));
        assert_eq!(capture(|o| print_totals(o, &results, None, 30)), golden("gpu_status/totals.txt"));
        let up: Vec<ServerResult> = results.iter().filter(|r| r.reachable).cloned().collect();
        assert_eq!(capture(|o| print_totals(o, &up, None, 30)), golden("gpu_status/totals.unreachable0.txt"));
    }

    fn gpu(idx: i64, util: i64, user: &str) -> gpu_spooler::gpu::GpuInfo {
        gpu_spooler::gpu::GpuInfo {
            index: idx,
            name: "A40".into(),
            mem_used_mb: 9000,
            mem_free_mb: 0,
            mem_total_mb: 46068,
            utilization_pct: util,
            processes: vec![GpuProcess { pid: 1, user: user.into(), mem_used_mb: 9000, command: "x".into() }],
        }
    }

    fn idle_gpu(host: &str, i: i64, user: &str) -> ((String, i64), IdleGpu) {
        (
            (host.into(), i),
            IdleGpu { host: host.into(), index: i, idle_seconds: 3600.0, users: vec![user.into()], capped: false },
        )
    }

    #[test]
    fn confirm_idle_drops_what_the_live_query_contradicts() {
        let results = [ServerResult::up("a", vec![gpu(0, 0, "alice"), gpu(1, 68, "bob"), gpu(2, 0, "erin")])];
        let mut idle: Idle =
            ["alice", "bob", "carol"].iter().enumerate().map(|(i, u)| idle_gpu("a", i as i64, u)).collect();
        idle.extend([idle_gpu("b", 0, "dave")]); // host not queried this time
        let kept = confirm_idle(Some(idle), &results).unwrap();
        assert_eq!(kept.keys().cloned().collect::<Vec<_>>(), [("a".to_string(), 0)]);
        assert!(confirm_idle(None, &results).is_none());
    }

    #[test]
    fn idle_shows_in_tables() {
        let results = vec![ServerResult::up("a", vec![gpu(0, 0, "alice"), gpu(1, 68, "bob")])];
        let idle: Idle = [idle_gpu("a", 0, "alice")].into_iter().collect();
        let cfg = gpu_spooler::config::builtin_config(std::path::Path::new("/h"));
        let summary = capture(|o| print_summary_table(o, &results, &cfg, false, Some(&idle)));
        assert!(summary.contains(&colorize("[1 held idle] ", YELLOW)), "{summary}");
        let detail = capture(|o| print_detail_table(o, &results, false, Some(&idle)));
        assert!(detail.contains(&colorize("IDLE", YELLOW)) && detail.contains("(idle 1h00m)"), "{detail}");
        let totals = capture(|o| print_totals(o, &results, Some(&idle), 30));
        assert!(totals.contains("GPUs held idle 30m+ (gpu-detective)"), "{totals}");
    }
}
