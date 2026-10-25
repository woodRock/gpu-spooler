//! What the tui shows, as styled text: the queue table's cells and the GPU heatmap. Pure
//! functions of the data, so they can be compared with the Python tui's output.

use gpu_spooler::core::runtime;
use gpu_spooler::db::Task;
use gpu_spooler::fmt::{fmt_duration, fmt_ts, one_line, truncate};
use gpu_spooler::gpu::ServerResult;
use gpu_spooler::py;
use ratatui::style::{Color, Modifier, Style};
use serde_json::Value;
use std::collections::HashMap;

pub const COLUMNS: [&str; 9] =
    ["ID", "STATUS", "G", "HOST", "SERVER", "GPU", "SUBMITTED", "RUNTIME", "LABEL / COMMAND"];

pub fn bold() -> Style {
    Style::new().add_modifier(Modifier::BOLD)
}

pub fn dim() -> Style {
    Style::new().add_modifier(Modifier::DIM)
}

fn on(fg: Color, bg: Color) -> Style {
    Style::new().fg(fg).bg(bg)
}

/// How a status is shown (the Python tui's STATUS_STYLE).
pub fn status_style(status: &str) -> Style {
    match status {
        "queued" => dim(),
        "running" => bold().fg(Color::Green),
        "done" => Style::new().fg(Color::Cyan),
        "failed" => bold().fg(Color::Red),
        "killed" | "cancelled" => Style::new().fg(Color::Yellow),
        "lost" => Style::new().fg(Color::Indexed(208)), // rich's dark_orange
        _ => Style::new(),
    }
}

/// Cells for one task in the queue table; the status cell carries its style.
pub fn task_row(t: &Task) -> Vec<(String, Style)> {
    let or_dash = |v: &Option<String>| v.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| "-".into());
    let name = t.name.as_deref().filter(|n| !n.is_empty()).unwrap_or(&t.command);
    let plain = |s: String| (s, Style::new());
    vec![
        plain(t.id.to_string()),
        (t.status.clone(), status_style(&t.status)),
        plain(t.num_gpus.to_string()),
        plain(or_dash(&t.host_spec)),
        plain(or_dash(&t.server)),
        plain(or_dash(&t.gpu_indices)),
        plain(fmt_ts(Some(t.submitted_at))),
        plain(fmt_duration(runtime(t))),
        plain(one_line(name)),
    ]
}

/// Hosts as rows, GPUs as cells: free, ours (the task id) or another user's. `running` are our
/// running tasks. As text runs with their styles; lines end in "\n".
pub fn gpu_map(
    snapshot: Option<&Value>,
    running: &[Task],
    workers: &[String],
    now: f64,
    daemon_running: bool,
) -> Vec<(String, Style)> {
    let mut t: Vec<(String, Style)> = vec![];
    let mut push = |s: String, style: Style| t.push((s, style));
    let hosts_json = snapshot.and_then(|s| s.get("hosts")).and_then(Value::as_object).filter(|h| !h.is_empty());
    let (Some(snapshot), Some(hosts_json)) = (snapshot, hosts_json) else {
        push("No GPU snapshot yet. ".into(), bold());
        let stopped = if daemon_running { "" } else { " (daemon stopped: press d)" };
        push(format!("The daemon writes one while tui is open{stopped}."), dim());
        return t;
    };

    let mut ours: HashMap<(String, i64), i64> = HashMap::new();
    for r in running {
        for idx in r.gpu_indices.as_deref().unwrap_or("").split(',') {
            if py::isdigit(py::strip(idx)) {
                if let Some(i) = py::int(idx) {
                    ours.insert((r.server.clone().unwrap_or_default(), i), r.id);
                }
            }
        }
    }

    push("  free ".into(), on(Color::Black, Color::Green));
    push(" ".into(), Style::new());
    push(" ours ".into(), on(Color::White, Color::Blue));
    push(" ".into(), Style::new());
    push(" in use ".into(), on(Color::White, Color::Red));
    let ts = snapshot.get("ts").and_then(Value::as_f64).unwrap_or(now);
    let mode = snapshot.get("mode").and_then(Value::as_str).unwrap_or("?");
    push(format!("   updated {} ago · network {mode}\n\n", fmt_duration(Some(now - ts))), dim());

    let hosts: HashMap<&String, ServerResult> =
        hosts_json.iter().filter_map(|(n, h)| serde_json::from_value(h.clone()).ok().map(|r| (n, r))).collect();
    let mut names: Vec<&String> = workers.iter().filter(|n| hosts.contains_key(n)).collect();
    let mut extra: Vec<&String> = hosts.keys().copied().filter(|n| !workers.contains(n)).collect();
    extra.sort();
    names.extend(extra);
    let width = names.iter().map(|n| n.chars().count()).max().unwrap_or(6) + 2;
    for name in names {
        let h = &hosts[name];
        push(format!("{name:<width$}"), bold());
        if !h.reachable {
            push("unreachable".into(), dim().fg(Color::Red));
            if !h.error.is_empty() {
                push(format!("  {}", truncate(&h.error, 60)), dim());
            }
            push("\n".into(), Style::new());
            continue;
        }
        if h.gpus.is_empty() {
            push("no GPUs (CPU only)".into(), dim());
        }
        for g in &h.gpus {
            match ours.get(&(name.clone(), g.index)) {
                Some(tid) => push(format!(" #{tid:<4}"), on(Color::White, Color::Blue)),
                None if g.is_free() => push(format!(" {:<5}", g.index), on(Color::Black, Color::Green)),
                None => {
                    let user = g.processes.iter().map(|p| p.user.as_str()).find(|u| *u != "?").unwrap_or("?");
                    push(format!(" {:<5}", truncate(user, 5)), on(Color::White, Color::Red));
                }
            }
            push(" ".into(), Style::new());
        }
        if let Some(first) = h.gpus.first() {
            let free = h.gpus.iter().filter(|g| g.is_free() && !ours.contains_key(&(name.clone(), g.index))).count();
            push(format!("  {free}/{} free  {}", h.total_gpus(), first.name), dim());
        }
        push("\n".into(), Style::new());
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

    fn golden(name: &str) -> Value {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/parity").join(name);
        serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
    }

    /// A style by the name rich gives it ("bold green", "white on blue", ...).
    fn rich(style: Style) -> String {
        let color = |c: Color| match c {
            Color::Black => "black",
            Color::White => "white",
            Color::Green => "green",
            Color::Red => "red",
            Color::Blue => "blue",
            Color::Cyan => "cyan",
            Color::Yellow => "yellow",
            Color::Indexed(208) => "dark_orange",
            other => panic!("no rich name for {other:?}"),
        };
        let mut words = vec![];
        if style.add_modifier.contains(Modifier::BOLD) {
            words.push("bold");
        }
        if style.add_modifier.contains(Modifier::DIM) {
            words.push("dim");
        }
        if let Some(fg) = style.fg {
            words.push(color(fg));
        }
        let mut name = words.join(" ");
        if let Some(bg) = style.bg {
            name = format!("{name} on {}", color(bg));
        }
        name
    }

    fn task(v: &Value) -> Task {
        let mut v = v.clone();
        v.as_object_mut().unwrap().retain(|_, x| !x.is_null());
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn task_rows_match_python() {
        for c in golden("tui_task_row.json").as_array().unwrap() {
            let got: Vec<Value> = task_row(&task(&c["input"]))
                .into_iter()
                .enumerate()
                .map(|(i, (text, style))| if i == 1 { json!([text, rich(style)]) } else { json!(text) })
                .collect();
            // fmt_ts is local time, the fixture UTC: compare the SUBMITTED column by shape only.
            let mut want = c["output"].as_array().unwrap().clone();
            let mut got = got;
            assert_eq!(got[6].as_str().unwrap().len(), want[6].as_str().unwrap().len());
            got[6] = json!("<ts>");
            want[6] = json!("<ts>");
            assert_eq!(got, want, "{}", c["name"]);
        }
    }

    #[test]
    fn gpu_map_matches_python() {
        for c in golden("tui_gpu_map.json").as_array().unwrap() {
            let i = &c["input"];
            let running: Vec<Task> = i["running"].as_array().unwrap().iter().map(task).collect();
            let workers: Vec<String> =
                i["workers"].as_array().unwrap().iter().map(|w| w.as_str().unwrap().into()).collect();
            let snapshot = Some(&i["snapshot"]).filter(|s| !s.is_null());
            let segs =
                gpu_map(snapshot, &running, &workers, i["now"].as_f64().unwrap(), i["daemon"].as_bool().unwrap());
            // rich keeps unstyled runs together: do the same before comparing
            let mut got: Vec<(String, String)> = vec![];
            for (text, style) in segs {
                let name = rich(style);
                match got.last_mut() {
                    Some((t, n)) if n.is_empty() && name.is_empty() => t.push_str(&text),
                    _ => got.push((text, name)),
                }
            }
            let want: Vec<(String, String)> = c["output"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| (s[0].as_str().unwrap().into(), s[1].as_str().unwrap().into()))
                .collect();
            assert_eq!(got, want, "{}", c["name"]);
        }
    }
}
