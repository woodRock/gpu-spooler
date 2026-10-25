//! The tui driven by keys and rendered to a TestBackend (ported from tests/test_tui.py, which used
//! Textual's test pilot).

use super::app::{App, Modal, Tab};
use super::view::draw;
use gpu_spooler::core::{self, SubmitOpts};
use gpu_spooler::paths::Paths;
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui::style::Color;
use ratatui::Terminal;
use std::time::Duration;

/// An empty state dir with a local-only config; no daemon.
struct State {
    _dir: tempfile::TempDir,
    paths: Paths,
}

impl State {
    fn new() -> State {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::for_home(dir.path());
        std::fs::create_dir_all(&paths.state).unwrap();
        std::fs::write(&paths.config, "[hosts.local]\nkind = \"local\"\ncpu_slots = 1\n").unwrap();
        State { _dir: dir, paths }
    }

    fn transport(&self) -> gpu_spooler::transport::Transport {
        let cfg = gpu_spooler::config::load_config(&self.paths).unwrap();
        gpu_spooler::transport::Transport::new(cfg, self.paths.clone())
    }

    fn submit(&self, command: &str, num_gpus: i64, f: impl FnOnce(&mut SubmitOpts)) -> core::Submitted {
        let mut opts = SubmitOpts {
            command: command.into(),
            num_gpus,
            working_dir: Some(self.paths.home.display().to_string()),
            ..Default::default()
        };
        f(&mut opts);
        core::submit(&self.transport(), &opts).unwrap()
    }

    fn task(&self, id: i64) -> gpu_spooler::db::Task {
        core::get_task(&gpu_spooler::db::open(&self.paths).unwrap(), id).unwrap()
    }

    fn app(&self) -> App {
        let mut app = App::new(self.paths.clone(), self.paths.home.display().to_string());
        app.start_daemon = false;
        app.tick(true);
        app
    }
}

fn press(app: &mut App, keys: &str) {
    for c in keys.chars() {
        let code = match c {
            '\n' => KeyCode::Enter,
            '\x1b' => KeyCode::Esc,
            '\t' => KeyCode::Tab,
            '\x08' => KeyCode::Backspace,
            '↓' => KeyCode::Down,
            '↑' => KeyCode::Up,
            c => KeyCode::Char(c),
        };
        app.key(KeyEvent::from(code));
    }
}

fn screen(app: &App) -> (Vec<String>, ratatui::buffer::Buffer) {
    let mut t = Terminal::new(TestBackend::new(140, 40)).unwrap();
    t.draw(|f| draw(f, app)).unwrap();
    let buf = t.backend().buffer().clone();
    let w = buf.area.width as usize;
    let lines = buf.content.chunks(w).map(|row| row.iter().map(|c| c.symbol()).collect::<String>()).collect();
    (lines, buf)
}

fn text(app: &App) -> String {
    screen(app).0.join("\n")
}

#[test]
fn table_shows_tasks_and_kill_cancels() {
    let s = State::new();
    let a = s.submit("echo one", 1, |_| {}).task_id;
    let b = s.submit("echo two", 0, |o| o.name = Some("cpu job".into())).task_id;
    let mut app = s.app();
    assert_eq!(app.rows.len(), 2);
    assert_eq!(app.selected_task().unwrap().id, b); // newest first
    assert!(app.subtitle.contains("2 queued") && app.subtitle.contains("DAEMON STOPPED (d)"), "{}", app.subtitle);

    let (lines, buf) = screen(&app);
    assert!(lines[0].contains("tui — 0 running · 2 queued"), "{}", lines[0]);
    let head = lines.iter().position(|l| l.contains("ID") && l.contains("LABEL / COMMAND")).unwrap();
    let row = lines.iter().position(|l| l.contains("cpu job")).unwrap();
    assert_eq!(row, head + 1);
    assert_eq!(buf[(30, row as u16)].bg, Color::Blue); // the selected row is highlighted
    assert!(lines.last().unwrap().contains(" k  Kill") && lines.last().unwrap().contains(" q  Quit"));

    press(&mut app, "k");
    assert!(text(&app).contains("Cancel task 2: cpu job?"));
    press(&mut app, "y");
    let n = app.wait_for_note(Duration::from_secs(10)).unwrap();
    assert_eq!(n.text, format!("Task {b} cancelled."));
    assert_eq!(s.task(b).status, "cancelled");
    assert_eq!(s.task(a).status, "queued");
    press(&mut app, "↓kn"); // "no" leaves it alone
    assert!(app.modal.is_none());
    assert_eq!(s.task(a).status, "queued");
    app.tick(true);
    press(&mut app, "↑k");
    assert!(app.modal.is_none(), "a cancelled task can't be killed");
    assert!(app.notes.iter().any(|n| n.text == "Select a queued or running task."));
}

#[test]
fn resubmit_and_new_task_form() {
    let s = State::new();
    let orig = s.submit("python3 x.py", 2, |o| {
        o.min_mem = 40;
        o.name = Some("lbl".into());
        o.host_spec = Some("local".into());
    });
    let mut app = s.app();
    press(&mut app, "r");
    assert!(app.wait_for_note(Duration::from_secs(10)).unwrap().text.starts_with("Queued task 2: python3 x.py"));
    let copy = s.task(orig.task_id + 1);
    assert_eq!(
        (copy.command.as_str(), copy.num_gpus, copy.min_mem, copy.name.as_deref(), copy.host_spec.as_deref()),
        ("python3 x.py", 2, 40, Some("lbl"), Some("local"))
    );
    assert_eq!(copy.working_dir, orig.working_dir);

    press(&mut app, "n");
    let shown = text(&app);
    assert!(shown.contains("New task") && shown.contains("GPUs (-G)") && shown.contains("Directory (-d)"), "{shown}");
    press(&mut app, "\n"); // no command: nothing queued
    assert!(app.bell && matches!(app.modal, Some(Modal::Submit(_))));
    press(&mut app, "echo hi && echo there\t\x080\t\t\tfrom tui\n");
    let n = app.wait_for_note(Duration::from_secs(10)).unwrap();
    assert!(n.text.starts_with("Queued task 3: echo hi && echo there"), "{}", n.text);
    let new = s.task(orig.task_id + 2);
    assert_eq!(
        (new.command.as_str(), new.num_gpus, new.name.as_deref()),
        ("echo hi && echo there", 0, Some("from tui"))
    );
    press(&mut app, "n\x1b");
    assert!(app.modal.is_none());
}

#[test]
fn log_follow_and_info_panes() {
    let s = State::new();
    let sub = s.submit("echo hi", 1, |_| {});
    let log = s.paths.logs().join(format!("{}.log", sub.task_id));
    std::fs::write(&log, "# header\nfirst line\n").unwrap();
    let con = gpu_spooler::db::open(&s.paths).unwrap();
    con.execute(
        "UPDATE tasks SET status='running', server='local', started_at=?, log_file=? WHERE id=?",
        rusqlite::params![gpu_spooler::transport::now(), log.display().to_string(), sub.task_id],
    )
    .unwrap();
    let mut app = s.app();
    assert!(text(&app).contains("first line"));
    std::fs::OpenOptions::new()
        .append(true)
        .open(&log)
        .and_then(|mut f| std::io::Write::write_all(&mut f, b"second line\n"))
        .unwrap();
    app.refresh_log();
    assert!(text(&app).contains("second line"));

    press(&mut app, "f");
    assert!(app.follow && app.tab == Tab::Log);
    let full = text(&app);
    assert!(full.contains("first line") && !full.contains("LABEL / COMMAND"), "the log fills the screen");
    press(&mut app, "f");
    assert!(!app.follow && text(&app).contains("LABEL / COMMAND"));

    press(&mut app, "i");
    let info = text(&app);
    assert!(info.contains("Status:      running") && info.contains("Server:      local"), "{info}");

    // A rewritten log (a retry) starts over.
    std::fs::write(&log, "retry\n").unwrap();
    app.refresh_log();
    assert_eq!(app.log, "retry\n");
}

#[test]
fn gpu_map_from_the_snapshot() {
    let s = State::new();
    let sub = s.submit("train", 1, |_| {});
    let con = gpu_spooler::db::open(&s.paths).unwrap();
    con.execute("UPDATE tasks SET status='running', server='cuda01', gpu_indices='1' WHERE id=?", [sub.task_id])
        .unwrap();
    let gpu = |i: i64, used: i64, procs: serde_json::Value| {
        serde_json::json!({"index": i, "name": "RTX A5000", "mem_used_mb": used, "mem_free_mb": 0,
                           "mem_total_mb": 24564, "utilization_pct": 0, "processes": procs})
    };
    let snapshot = serde_json::json!({"ts": gpu_spooler::transport::now() - 5.0, "mode": "jump", "hosts": {
        "cuda01": {"name": "cuda01", "reachable": true, "error": "", "gpus": [
            gpu(0, 9000, serde_json::json!([{"pid": 1, "user": "alice", "mem_used_mb": 9000, "command": "py"}])),
            gpu(1, 1, serde_json::json!([])), gpu(2, 1, serde_json::json!([]))]},
        "cuda02": {"name": "cuda02", "reachable": false, "error": "timed out", "gpus": []},
    }});
    std::fs::write(s.paths.snapshot(), snapshot.to_string()).unwrap();
    let mut app = s.app();
    assert!(s.paths.snapshot_request().exists(), "tui asks the daemon for fresh snapshots");
    press(&mut app, "g");
    let (lines, buf) = screen(&app);
    let shown = lines.join("\n");
    assert!(shown.contains("alice") && shown.contains(&format!("#{}", sub.task_id)), "{shown}");
    assert!(shown.contains("1/3 free") && shown.contains("unreachable") && shown.contains("network jump"), "{shown}");
    let row = lines.iter().position(|l| l.starts_with(" cuda01")).unwrap();
    let col = |s: &str| lines[row].find(s).unwrap() as u16;
    assert_eq!(buf[(col("alice"), row as u16)].bg, Color::Red);
    assert_eq!(buf[(col(&format!("#{}", sub.task_id)), row as u16)].bg, Color::Blue);
    assert_eq!(buf[(col(" 2 ") + 1, row as u16)].bg, Color::Green);
}

#[test]
fn gpu_map_without_a_snapshot() {
    let s = State::new();
    let mut app = s.app();
    press(&mut app, "g");
    assert!(
        text(&app).contains("No GPU snapshot yet. The daemon writes one while tui is open (daemon stopped: press d).")
    );
}

#[test]
fn clear_asks_first_and_quit() {
    let s = State::new();
    let done = s.submit("true", 1, |_| {}).task_id;
    gpu_spooler::db::open(&s.paths).unwrap().execute("UPDATE tasks SET status='done' WHERE id=?", [done]).unwrap();
    let mut app = s.app();
    press(&mut app, "c");
    assert!(text(&app).contains("Remove all finished tasks, their logs and job directories?"));
    press(&mut app, "\x1b");
    assert!(app.modal.is_none());
    press(&mut app, "cy");
    assert_eq!(app.wait_for_note(Duration::from_secs(10)).unwrap().text, "Removed 1 finished task(s).");
    app.tick(true);
    assert!(app.rows.is_empty() && text(&app).contains("0 running · 0 queued"));
    press(&mut app, "q");
    assert!(app.quit);
}
