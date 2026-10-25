//! The port against the golden fixtures from the Python implementation (dev/gen_golden.py says
//! what each file holds). Each migration step adds the modules it ports.

mod common;

use common::{cases, text, Home};
use gpu_spooler::config::{self, Config, Kind, Num, Role};
use gpu_spooler::gpu::{self, GpuInfo, ServerResult};
use gpu_spooler::transport::{Mode, Transport};
use gpu_spooler::{fmt, shell};
use serde_json::{json, Value};
use std::collections::BTreeSet;

// ── The fixtures themselves ──────────────────────────────────────────────────

#[test]
fn index_lists_every_fixture() {
    let listed: BTreeSet<String> = common::index().into_iter().collect();
    let mut found = BTreeSet::new();
    let mut stack = vec![common::dir()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let rel = path.strip_prefix(common::dir()).unwrap().to_string_lossy().into_owned();
                if rel != "INDEX" {
                    found.insert(rel);
                }
            }
        }
    }
    assert_eq!(listed, found, "tests/golden/parity/INDEX is stale; run dev/gen_golden.py");
}

#[test]
fn json_fixtures_are_named_cases() {
    for name in common::index().iter().filter(|n| n.ends_with(".json")) {
        let cases = cases(name);
        assert!(!cases.is_empty(), "{name}: no cases");
        let names: BTreeSet<&str> = cases.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names.len(), cases.len(), "{name}: duplicate case names");
    }
}

fn s(v: &Value) -> &str {
    v.as_str().unwrap_or_else(|| panic!("not a string: {v}"))
}

fn opt_s(v: &Value) -> Option<&str> {
    v.as_str()
}

// ── shell, fmt ───────────────────────────────────────────────────────────────

#[test]
fn shell_quoting() {
    for c in cases("shell_quote.json") {
        assert_eq!(shell::quote(s(&c.input)), s(&c.output), "quote {}", c.name);
    }
    for c in cases("command_from_argv.json") {
        let argv: Vec<&str> = c.input.as_array().unwrap().iter().map(s).collect();
        assert_eq!(shell::command_from_argv(&argv), s(&c.output), "{}", c.name);
    }
    for c in cases("sh_path.json") {
        assert_eq!(shell::sh_path(s(&c.input)), s(&c.output), "sh_path {}", c.name);
    }
}

#[test]
fn formatting() {
    for c in cases("fmt_duration.json") {
        assert_eq!(fmt::fmt_duration(c.input.as_f64()), s(&c.output), "fmt_duration {}", c.name);
    }
    for c in cases("fmt_ts.json") {
        assert_eq!(fmt::fmt_ts_in(c.input.as_f64(), &chrono::Utc), s(&c.output), "fmt_ts {}", c.name);
    }
    for c in cases("truncate.json") {
        let n = c.input[1].as_u64().unwrap() as usize;
        assert_eq!(fmt::truncate(s(&c.input[0]), n), s(&c.output), "truncate {}", c.name);
    }
    for c in cases("one_line.json") {
        assert_eq!(fmt::one_line(s(&c.input)), s(&c.output), "one_line {}", c.name);
    }
    for c in cases("gpu_model.json") {
        assert_eq!(gpu::gpu_model(s(&c.input)), s(&c.output), "gpu_model {}", c.name);
    }
    for c in cases("gpu_type_matches.json") {
        let got = gpu::gpu_type_matches(opt_s(&c.input[0]), s(&c.input[1]));
        assert_eq!(got, c.output.as_bool().unwrap(), "gpu_type_matches {}", c.name);
    }
}

// ── config ───────────────────────────────────────────────────────────────────

fn num(n: Option<Num>) -> Value {
    match n {
        None => Value::Null,
        Some(Num::Int(i)) => json!(i),
        Some(Num::Float(f)) => json!(f),
    }
}

/// A Config as the generator writes it (dataclasses.asdict, hosts as a list in order).
fn config_json(cfg: &Config) -> Value {
    let hosts: Vec<Value> = cfg
        .hosts
        .values()
        .map(|h| {
            json!({
                "name": h.name,
                "kind": match h.kind { Kind::Ssh => "ssh", Kind::Local => "local" },
                "hostname": h.hostname, "user": h.user, "port": h.port, "jump": h.jump,
                "role": match h.role { Role::Worker => "worker", Role::Jump => "jump" },
                "tags": h.tags, "group": h.group,
                "path_map": h.path_map.iter().map(|(l, r)| json!([l, r])).collect::<Vec<_>>(),
                "cpu_slots": h.cpu_slots, "remote_root": h.remote_root, "enabled": h.enabled,
                "gpu_type": h.gpu_type, "num_gpus": h.num_gpus, "gpu_mem_gb": num(h.gpu_mem_gb),
                "cpu_cores": h.cpu_cores, "sys_mem_gb": num(h.sys_mem_gb),
            })
        })
        .collect();
    json!({
        "hosts": hosts,
        "jumps": cfg.jumps.iter().map(|(k, v)| (k.clone(), json!(v))).collect::<serde_json::Map<_, _>>(),
        "mode": cfg.mode.as_str(), "probe": cfg.probe, "identity_file": cfg.identity_file,
        "control_persist": cfg.control_persist, "jump_control_persist": cfg.jump_control_persist,
        "extra_options": cfg.extra_options, "builtin": cfg.builtin, "max_gpus": cfg.max_gpus,
        "taura": cfg.taura, "detective_history": cfg.detective_history, "idle_minutes": cfg.idle_minutes,
        "workers": cfg.workers().iter().map(|h| h.name.as_str()).collect::<Vec<_>>(),
    })
}

fn parse(home: &Home, toml_text: &str) -> Result<Config, config::ConfigError> {
    let data: toml::Table = toml_text.parse().map_err(|_| config::ConfigError("invalid TOML".into()))?;
    config::parse_config(&data, &home.paths())
}

#[test]
fn parse_config_matches_python() {
    let home = Home::new();
    for c in cases("parse_config.json") {
        let got = if c.input.is_null() {
            config_json(&config::builtin_config(home.dir.path()))
        } else {
            match parse(&home, s(&c.input)) {
                Ok(cfg) => config_json(&cfg),
                Err(e) => json!({"error": "ConfigError", "message": e.0}),
            }
        };
        let got: Value = serde_json::from_str(&home.scrub(&got.to_string())).unwrap();
        assert_eq!(got, c.output, "parse_config {}", c.name);
    }
}

#[test]
fn config_template_matches_python() {
    assert_eq!(config::config_template("woodj4"), text("config_template.toml"));
}

#[test]
fn select_matches_python() {
    let home = Home::new();
    let all = cases("select.json");
    let cfg = parse(&home, s(&all[0].input)).unwrap();
    for c in &all[1..] {
        let got = match cfg.select(opt_s(&c.input)) {
            Ok(hosts) => json!(hosts.iter().map(|h| h.name.as_str()).collect::<Vec<_>>()),
            Err(e) => json!({"error": "ConfigError", "message": e.0}),
        };
        assert_eq!(got, c.output, "select {}", c.name);
    }
}

#[test]
fn path_mapping_matches_python() {
    let home = Home::new();
    for c in cases("map_path.json") {
        let cfg = parse(&home, s(&c.input[0])).unwrap();
        let host = cfg.get(s(&c.input[1])).unwrap();
        let path = home.unscrub(s(&c.input[2]));
        assert_eq!(cfg.map_path(host, &path), s(&c.output["map_path"]), "map_path {}", c.name);
        assert_eq!(cfg.is_mapped(host, &path), c.output["is_mapped"].as_bool().unwrap(), "is_mapped {}", c.name);
    }
    for c in cases("remote_dir.json") {
        let cfg = parse(&home, s(&c.input[0])).unwrap();
        let host = cfg.get(s(&c.input[1])).unwrap();
        assert_eq!(cfg.remote_dir(host, s(&c.input[2]), opt_s(&c.input[3])), s(&c.output), "remote_dir {}", c.name);
    }
}

// ── transport ────────────────────────────────────────────────────────────────

#[test]
fn ssh_configs_match_python() {
    let template = config::config_template("woodj4");
    let fake = &cases("parse_config.json").into_iter().find(|c| c.name == "fake_cluster").unwrap().input;
    let merging = &cases("parse_config.json").into_iter().find(|c| c.name == "merging").unwrap().input;
    let fixtures = [
        ("builtin.direct", None, Mode::Direct, false),
        ("builtin.include", None, Mode::Direct, true),
        ("template.direct", Some(template.as_str()), Mode::Direct, false),
        ("template.jump", Some(template.as_str()), Mode::Jump, false),
        ("fake_cluster.jump", Some(s(fake)), Mode::Jump, false),
        ("fake_cluster.direct", Some(s(fake)), Mode::Direct, false),
        ("merging.jump", Some(s(merging)), Mode::Jump, false),
    ];
    for (name, src, mode, include) in fixtures {
        let home = Home::new();
        if include {
            std::fs::create_dir(home.dir.path().join(".ssh")).unwrap();
            std::fs::write(home.dir.path().join(".ssh/config"), "Host *\n").unwrap();
        }
        let paths = home.paths();
        let cfg = match src {
            None => config::builtin_config(&paths.home),
            Some(src) => {
                let mut cfg = parse(&home, src).unwrap();
                cfg.source = Some(paths.config.clone());
                cfg
            }
        };
        let t = Transport::new(cfg, paths);
        assert_eq!(home.scrub(&t.render_ssh_config(mode)), text(&format!("ssh_config/{name}")), "ssh_config/{name}");
    }
    // The fixture the Python tests already use.
    let home = Home::new();
    let t = Transport::new(config::builtin_config(home.dir.path()), home.paths());
    let golden = std::fs::read_to_string(common::dir().join("../ssh_config.builtin")).unwrap();
    assert_eq!(home.scrub(&t.render_ssh_config(Mode::Direct)), golden);
}

// ── gpu ──────────────────────────────────────────────────────────────────────

#[test]
fn remote_gpu_query_is_byte_identical() {
    assert_eq!(gpu::REMOTE_CMD, text("gpu/remote_cmd.sh"));
}

fn gpu_json(g: &GpuInfo) -> Value {
    let mut v = serde_json::to_value(g).unwrap();
    v["is_free"] = json!(g.is_free());
    v["mem_total_gb"] = json!(g.mem_total_gb());
    v
}

#[test]
fn parse_output_matches_python() {
    for c in cases("gpu_parse_output.json") {
        let got: Vec<Value> = gpu::parse_output(s(&c.input)).iter().map(gpu_json).collect();
        let want = c.output.as_array().unwrap();
        assert_eq!(got.len(), want.len(), "parse_output {}", c.name);
        for (g, w) in got.iter().zip(want) {
            let (gt, wt) = (g["mem_total_gb"].as_f64().unwrap(), w["mem_total_gb"].as_f64().unwrap());
            assert!((gt - wt).abs() < 1e-9, "parse_output {}: mem_total_gb {gt} vs {wt}", c.name);
            let strip = |v: &Value| {
                let mut v = v.clone();
                v.as_object_mut().unwrap().remove("mem_total_gb");
                v
            };
            assert_eq!(strip(g), strip(w), "parse_output {}", c.name);
        }
    }
}

#[test]
fn server_results_match_python() {
    for c in cases("server_result.json") {
        let r: ServerResult = serde_json::from_value(c.input.clone()).unwrap();
        let got = json!({
            "total_gpus": r.total_gpus(), "free_gpus": r.free_gpus(), "busy_gpus": r.busy_gpus(),
            "total_vram_mb": r.total_vram_mb(), "free_vram_mb": r.free_vram_mb(), "models": r.models(),
        });
        assert_eq!(got, c.output, "server_result {}", c.name);
    }
}
