//! The menu bar item and the app's windows: a template icon with the number of running tasks
//! beside it, a popover under it on click, and notifications when a task ends. Closing the main
//! window hides it; the app keeps running in the menu bar until Quit.

use crate::queue::QueueState;
use gpu_spooler::db::Task;
use serde::Serialize;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::image::Image;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager, PhysicalPosition, Rect, WebviewWindow};
use tauri_plugin_autostart::ManagerExt as _;
use tauri_plugin_notification::NotificationExt as _;
use tauri_specta::Event;

pub const MAIN: &str = "main";
pub const POPOVER: &str = "popover";
const TRAY: &str = "tray";
/// Passed by the login item: start in the menu bar only.
pub const HIDDEN_ARG: &str = "--hidden";

/// When the popover last hid itself on losing focus. A click on the menu bar icon takes the focus
/// first, so the click that follows must not show it again at once.
#[derive(Default)]
pub struct PopoverState(Mutex<Option<Instant>>);

/// Asks the main window to show a task, or the New Task sheet.
#[derive(Clone, Debug, Serialize, specta::Type, tauri_specta::Event)]
pub struct ShowRequest {
    pub task: Option<i64>,
    pub new_task: bool,
}

pub fn setup(app: &AppHandle) -> tauri::Result<()> {
    let open = MenuItem::with_id(app, "open", "Open gpu-spooler", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit gpu-spooler", true, Some("CmdOrCtrl+Q"))?;
    let menu = Menu::with_items(app, &[&open, &PredefinedMenuItem::separator(app)?, &quit])?;
    TrayIconBuilder::with_id(TRAY)
        .icon(Image::from_bytes(include_bytes!("../icons/tray.png"))?)
        .icon_as_template(true)
        .tooltip("gpu-spooler")
        .menu(&menu)
        // Left click: the popover. Right click: the menu.
        .show_menu_on_left_click(false)
        .on_menu_event(|app, e| match e.id().as_ref() {
            "open" => show_main(app),
            "quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, e| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left, button_state: MouseButtonState::Up, rect, ..
            } = e
            {
                toggle_popover(tray.app_handle(), rect);
            }
        })
        .build(app)?;
    Ok(())
}

pub fn show_main(app: &AppHandle) {
    if let Some(w) = app.get_webview_window(MAIN) {
        let _ = w.unminimize();
        let _ = w.show();
        let _ = w.set_focus();
    }
    if let Some(p) = app.get_webview_window(POPOVER) {
        let _ = p.hide();
    }
}

fn toggle_popover(app: &AppHandle, icon: Rect) {
    let Some(w) = app.get_webview_window(POPOVER) else { return };
    // Hide only a popover that is on screen and in use. One macOS still counts as visible but
    // that lost focus without telling us (another app came forward) is brought back instead:
    // otherwise the click looks like it did nothing.
    if w.is_visible().unwrap_or(false) && w.is_focused().unwrap_or(false) {
        let _ = w.hide();
        return;
    }
    let just_hid =
        app.state::<PopoverState>().0.lock().unwrap().is_some_and(|t| t.elapsed() < Duration::from_millis(300));
    if just_hid {
        return;
    }
    place_under(&w, icon);
    let _ = w.show();
    let _ = w.set_focus();
}

/// Centre the popover under the menu bar icon, kept on screen.
fn place_under(w: &WebviewWindow, icon: Rect) {
    let scale = w.scale_factor().unwrap_or(1.0);
    let pos = icon.position.to_physical::<f64>(scale);
    let size = icon.size.to_physical::<f64>(scale);
    let width = w.outer_size().map_or(0.0, |s| s.width as f64);
    let mut x = pos.x + size.width / 2.0 - width / 2.0;
    if let Ok(Some(m)) = w.current_monitor() {
        let left = m.position().x as f64;
        let right = left + m.size().width as f64;
        x = x.clamp(left + 8.0 * scale, (right - width - 8.0 * scale).max(left));
    }
    let _ = w.set_position(PhysicalPosition::new(x, pos.y + size.height + 4.0 * scale));
}

/// Window events: the popover hides when it loses focus; closing the main window hides it.
pub fn on_window_event(w: &tauri::Window, e: &tauri::WindowEvent) {
    match (w.label(), e) {
        (POPOVER, tauri::WindowEvent::Focused(false)) => {
            let _ = w.hide();
            *w.app_handle().state::<PopoverState>().0.lock().unwrap() = Some(Instant::now());
        }
        (MAIN, tauri::WindowEvent::CloseRequested { api, .. }) => {
            api.prevent_close();
            let _ = w.hide();
        }
        _ => {}
    }
}

/// The running count beside the icon, and the queue in its tooltip.
pub fn update_tray(app: &AppHandle, q: &QueueState) {
    let Some(tray) = app.tray_by_id(TRAY) else { return };
    let count = |s: &str| q.tasks.iter().filter(|t| t.status == s).count();
    let (running, queued) = (count("running"), count("queued"));
    let _ = tray.set_title((running > 0).then(|| running.to_string()));
    let mut tip = format!("gpu-spooler: {running} running, {queued} queued");
    if q.daemon_pid.is_none() {
        tip += " (daemon stopped)";
    }
    let _ = tray.set_tooltip(Some(tip));
}

/// Tasks that ended between two reads: done, failed or lost, having been queued or running
/// before. Killed and cancelled are left out (the user did that); so is anything not in `prev`,
/// so the first read after launch says nothing.
pub fn ended<'a>(prev: &QueueState, next: &'a QueueState) -> Vec<&'a Task> {
    next.tasks
        .iter()
        .filter(|t| matches!(t.status.as_str(), "done" | "failed" | "lost"))
        .filter(|t| prev.tasks.iter().any(|p| p.id == t.id && matches!(p.status.as_str(), "queued" | "running")))
        .collect()
}

/// (title, body) of a task's notification.
pub fn notice(t: &Task) -> (String, String) {
    let title = match t.status.as_str() {
        "done" => format!("Task {} done", t.id),
        "failed" => match t.exit_code {
            Some(c) => format!("Task {} failed (exit {c})", t.id),
            None => format!("Task {} failed", t.id),
        },
        _ => format!("Task {} lost", t.id),
    };
    let what = t.name.as_ref().filter(|n| !n.is_empty()).unwrap_or(&t.command);
    let mut body = gpu_spooler::fmt::truncate(what, 80);
    if let Some(server) = t.server.as_ref().filter(|s| !s.is_empty()) {
        body += &format!(" · {server}");
    }
    let took = gpu_spooler::core::runtime(t);
    if took.is_some() {
        body += &format!(" · {}", gpu_spooler::fmt::fmt_duration(took));
    }
    (title, body)
}

pub fn notify_ended(app: &AppHandle, prev: Option<&QueueState>, next: &QueueState) {
    let Some(prev) = prev else { return };
    for t in ended(prev, next) {
        let (title, body) = notice(t);
        let _ = app.notification().builder().title(title).body(body).show();
    }
}

/// Show the main window, on a task or the New Task sheet.
#[tauri::command]
#[specta::specta]
pub fn open_main(app: AppHandle, task: Option<i64>, new_task: bool) -> Result<(), String> {
    show_main(&app);
    if task.is_some() || new_task {
        ShowRequest { task, new_task }.emit_to(&app, MAIN).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Esc in the popover.
#[tauri::command]
#[specta::specta]
pub fn hide_popover(app: AppHandle) {
    if let Some(p) = app.get_webview_window(POPOVER) {
        let _ = p.hide();
    }
}

#[tauri::command]
#[specta::specta]
pub fn quit(app: AppHandle) {
    app.exit(0);
}

/// Whether the app starts at login (a LaunchAgent, in the menu bar only).
#[tauri::command]
#[specta::specta]
pub fn launch_at_login(app: AppHandle) -> Result<bool, String> {
    app.autolaunch().is_enabled().map_err(|e| e.to_string())
}

#[tauri::command]
#[specta::specta]
pub fn set_launch_at_login(app: AppHandle, enabled: bool) -> Result<(), String> {
    let l = app.autolaunch();
    if enabled { l.enable() } else { l.disable() }.map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(tasks: &[(i64, &str)]) -> QueueState {
        QueueState {
            tasks: tasks.iter().map(|&(id, s)| Task { id, status: s.into(), ..Default::default() }).collect(),
            daemon_pid: None,
            gpus_in_use: 0,
            max_gpus: None,
            error: None,
        }
    }

    #[test]
    fn notifies_tasks_that_ended_since_the_last_read() {
        let prev = q(&[(1, "running"), (2, "running"), (3, "queued"), (4, "running"), (5, "done")]);
        let next = q(&[(1, "done"), (2, "failed"), (3, "lost"), (4, "killed"), (5, "done"), (6, "done")]);
        let ids: Vec<i64> = ended(&prev, &next).iter().map(|t| t.id).collect();
        assert_eq!(ids, [1, 2, 3]);
        assert!(ended(&next, &next).is_empty());
    }

    #[test]
    fn notices_say_how_it_ended() {
        let t = Task {
            id: 12,
            status: "failed".into(),
            exit_code: Some(3),
            name: Some("train".into()),
            server: Some("cuda14".into()),
            started_at: Some(100.0),
            finished_at: Some(165.0),
            ..Default::default()
        };
        assert_eq!(notice(&t), ("Task 12 failed (exit 3)".into(), "train · cuda14 · 1m 05s".into()));
        let t = Task { id: 3, status: "done".into(), command: "python3 eval.py".into(), ..Default::default() };
        assert_eq!(notice(&t), ("Task 3 done".into(), "python3 eval.py".into()));
    }
}
