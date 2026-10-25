//! The cluster and its configuration: `task hosts`, `task exec`, `task ssh`, `task connect`,
//! `task config init|path` and `task daemon status`.

use crate::color::{colorize, BOLD, DIM, GREEN, RED, RESET};
use crate::ctx::{CmdResult, Ctx, Fail};
use gpu_spooler::config::{config_template, Host, Role};
use gpu_spooler::fmt::truncate;
use gpu_spooler::gpu::query_hosts;
use gpu_spooler::py;
use gpu_spooler::shell;
use gpu_spooler::transport::{auth_hint, Mode};
use std::io::Write;
use std::sync::Mutex;
use std::time::Duration;

#[derive(clap::Args, Debug)]
pub struct ExecArgs {
    /// Only these hosts, groups or tags (comma separated)
    #[arg(short = 'H', long, value_name = "SPEC")]
    pub hosts: Option<String>,
    #[arg(value_name = "-- CMD", trailing_var_arg = true, allow_hyphen_values = true)]
    pub command: Vec<String>,
}

#[derive(clap::Args, Debug)]
pub struct SshArgs {
    #[arg(value_name = "HOST")]
    pub host: String,
}

#[derive(clap::Args, Debug)]
pub struct DaemonArgs {
    #[arg(value_name = "ACTION", value_parser = ["start", "stop", "status"])]
    pub action: String,
}

#[derive(clap::Args, Debug)]
pub struct ConfigArgs {
    #[arg(value_name = "ACTION", value_parser = ["init", "path"])]
    pub action: String,
    /// With init: your username on the servers (default: your local one)
    #[arg(short, long, value_name = "NAME")]
    pub user: Option<String>,
}

pub fn hosts(ctx: &Ctx) -> CmdResult {
    let transport = ctx.transport()?;
    let cfg = &transport.cfg;
    let hosts: Vec<&Host> = cfg.hosts.values().filter(|h| h.role == Role::Worker).collect();
    let mode = transport.mode(false);
    let mut out = std::io::stdout();
    writeln!(out, "{BOLD}Querying {} host(s)...{RESET}", hosts.len())?;
    let enabled: Vec<&Host> = hosts.iter().copied().filter(|h| h.enabled).collect();
    let results = query_hosts(transport, &enabled, 16, false);

    let hdr = format!(
        "{:<14} {:<5} {:<8} {:<16} {:<11} {:>4} {:>3}  PATH MAP",
        "HOST", "KIND", "GROUP", "VIA", "STATUS", "GPUS", "CPU"
    );
    let sep = "─".repeat(hdr.chars().count().max(80));
    writeln!(out, "\n{BOLD}{hdr}{RESET}")?;
    writeln!(out, "{sep}")?;
    for h in hosts {
        let chain = cfg.chain(h);
        let via = if h.is_local() {
            "local".to_string()
        } else if !chain.is_empty() && mode == Mode::Jump {
            chain.iter().map(|x| x.alias()).collect::<Vec<_>>().join("→")
        } else {
            "direct".to_string()
        };
        let r = results.get(&h.name);
        let status = if !h.enabled {
            colorize(&format!("{:<11}", "disabled"), DIM)
        } else if r.is_some_and(|r| r.reachable) {
            colorize(&format!("{:<11}", "ok"), GREEN)
        } else {
            colorize(&format!("{:<11}", "unreachable"), RED)
        };
        let gpus = r.filter(|r| r.reachable).map_or("-".to_string(), |r| r.total_gpus().to_string());
        let pmap = h.path_map.iter().map(|(l, rem)| format!("{l} → {rem}")).collect::<Vec<_>>().join(", ");
        let cpu = if h.cpu_slots != 0 { h.cpu_slots.to_string() } else { "-".into() };
        let kind = if h.is_local() { "local" } else { "ssh" };
        writeln!(
            out,
            "{:<14} {kind:<5} {:<8} {:<16} {status} {gpus:>4} {cpu:>3}  {}",
            h.name,
            h.group.as_deref().filter(|g| !g.is_empty()).unwrap_or("-"),
            truncate(&via, 16),
            if pmap.is_empty() { "-" } else { &pmap },
        )?;
        if let Some(r) = r.filter(|r| !r.reachable && !r.error.is_empty()) {
            writeln!(out, "{:14} {}", "", colorize(&truncate(&r.error, 90), DIM))?;
        }
    }
    writeln!(out, "{sep}")?;
    writeln!(out, "network: {}   config: {}\n", transport.describe_mode(true), cfg.source_label())?;
    Ok(0)
}

/// Run a command on one host in a login shell: (success, combined output).
fn run_on_server(ctx: &Ctx, host: &Host, command: &str) -> Result<(bool, String), Fail> {
    let inner = format!("source ~/.bashrc 2>/dev/null || true; {command}");
    let script = format!("exec bash -l -c {} < /dev/null\n", shell::quote(&inner));
    let r = ctx.transport()?.run(host, &script, Duration::from_secs(300));
    if r.rc == 124 {
        return Ok((false, "timed out after 300s".into()));
    }
    let output = py::strip(&format!("{}{}", r.stdout, r.stderr)).to_string();
    let hint = if r.rc == 255 { auth_hint(&r.stderr) } else { None };
    Ok((
        r.rc == 0,
        match hint {
            Some(hint) => format!("{output}\n({hint})"),
            None => output,
        },
    ))
}

pub fn exec(ctx: &Ctx, args: &ExecArgs) -> CmdResult {
    if args.command.is_empty() {
        return Err(Fail::new("task exec: no command given, e.g.: task exec -- nvidia-smi -L"));
    }
    let command = shell::command_from_argv(&args.command);
    let transport = ctx.transport()?;
    let hosts = transport
        .cfg
        .select(args.hosts.as_deref())
        .map_err(|e| Fail::new(format!("task exec: -H {}: {e}", args.hosts.as_deref().unwrap_or("None"))))?;
    let n = hosts.len();
    println!(
        "{BOLD}Running on {n} servers:{RESET} {command}  {DIM}network: {}{RESET}\n",
        transport.describe_mode(true)
    );

    let results: Mutex<Vec<(String, bool, String)>> = Mutex::new(vec![]);
    transport.fan_out(
        &hosts,
        |host| {
            let (ok, output) = run_on_server(ctx, host, &command).unwrap_or_else(|e| (false, e.message));
            let mut results = results.lock().unwrap();
            // Show the last non-empty line as a quick summary
            let last = py::splitlines(&output).into_iter().rev().find(|l| !py::strip(l).is_empty()).unwrap_or("");
            let status = if ok { colorize("OK  ", GREEN) } else { colorize("FAIL", RED) };
            println!("  [{:>2}/{n}] {:<16} {status}  {}", results.len() + 1, host.name, truncate(last, 60));
            results.push((host.name.clone(), ok, output));
        },
        16,
    );

    let results = results.into_inner().unwrap();
    let ok_count = results.iter().filter(|r| r.1).count();
    let fail_count = n - ok_count;
    // Full output for failed servers
    if fail_count > 0 {
        println!("\n{}", colorize(&format!("{fail_count} server(s) failed — full output:"), RED));
        for (server, _, out) in results.iter().filter(|r| !r.1) {
            println!("\n{BOLD}── {server} ──{RESET}");
            println!("{}", if out.is_empty() { "(no output)" } else { out });
        }
    }
    let failed =
        if fail_count > 0 { format!("  {}", colorize(&format!("{fail_count} failed"), RED)) } else { String::new() };
    println!("\n{}/{n} servers succeeded{failed}\n", colorize(&ok_count.to_string(), GREEN));
    Ok(0)
}

pub fn ssh(ctx: &Ctx, args: &SshArgs) -> CmdResult {
    use std::os::unix::process::CommandExt;
    let transport = ctx.transport()?;
    let host = transport.cfg.get(&args.host).map_err(|e| Fail::new(format!("task: {e}")))?;
    let argv = transport.interactive_argv(host)?;
    let err = std::process::Command::new(&argv[0]).args(&argv[1..]).exec();
    Err(Fail::new(format!("task: {}: {err}", argv[0])))
}

pub fn connect(ctx: &Ctx) -> CmdResult {
    let transport = ctx.transport()?;
    if transport.mode(true) != Mode::Jump {
        println!("Network mode is {}; no jump hosts to connect.", transport.describe_mode(true));
        return Ok(0);
    }
    if transport.connect(&transport.cfg.workers())? > 0 {
        return Err(Fail::silent(1));
    }
    println!("Jump connections are open; the daemon will reuse them.");
    Ok(0)
}

pub fn config(ctx: &Ctx, args: &ConfigArgs) -> CmdResult {
    let path = &ctx.paths.config;
    if args.action == "path" {
        if path.exists() {
            println!("{}", path.display());
        } else {
            println!("{} (does not exist; built-in defaults are used)", path.display());
        }
        return Ok(0);
    }
    if path.exists() {
        println!("{} already exists; not overwriting.", path.display());
        return Err(Fail::silent(1));
    }
    let user = args.user.clone().unwrap_or_else(gpu_spooler::core::getuser);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, config_template(&user))?;
    let how = if args.user.is_some() { "" } else { " (your local username; use --user if your ECS one differs)" };
    println!("Wrote {} for user '{user}'{how}. Check it with: task hosts", path.display());
    Ok(0)
}

pub fn daemon(ctx: &Ctx, args: &DaemonArgs) -> CmdResult {
    if args.action != "status" {
        return Err(Fail {
            message: format!(
                "task daemon {}: not in the Rust port yet; use the Python `task daemon {0}` for now",
                args.action
            ),
            code: 2,
        });
    }
    match gpu_spooler::core::daemon_pid(&ctx.paths).filter(|p| *p != 0) {
        Some(pid) => println!("Daemon running (PID {pid}). Logs: {}", ctx.paths.daemon_log().display()),
        None => println!("Daemon is not running (it starts on the next submit)."),
    }
    Ok(0)
}
