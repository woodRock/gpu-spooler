"""
tui - interactive view of the gpu-spooler queue, logs and GPUs.

A front end over gpu_spooler.core, like the `task` CLI: it reads the same
database and the GPU snapshot the daemon writes, and never runs its own SSH
fan-out (kill and clear do talk to the job's host, exactly as `task kill` does).

Keys: k kill/cancel · r resubmit · n new task · s sync git repo
      f follow log (full screen) · l log · g GPUs · i info
      c clear finished · d start daemon · q quit
"""

import sys
import time
from typing import Optional

try:
    from rich.text import Text
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Grid, Horizontal, Vertical
    from textual.screen import ModalScreen
    from textual.widgets import (
        Button, DataTable, Footer, Header, Input, Label, Log, Static, TabbedContent, TabPane,
    )
except ImportError:          # pragma: no cover - exercised without the extra
    App = None

from gpu_spooler import core
from gpu_spooler.config import ConfigError
from gpu_spooler.gpu import result_from_dict

TASK_REFRESH     = 1.0     # seconds between queue table refreshes
LOG_REFRESH      = 0.5
SNAPSHOT_REFRESH = 2.0     # re-read gpu_snapshot.json
SNAPSHOT_PING    = 30.0    # ask the daemon to keep the snapshot fresh

STATUS_STYLE = {
    "queued":    "dim",
    "running":   "bold green",
    "done":      "cyan",
    "failed":    "bold red",
    "killed":    "yellow",
    "cancelled": "yellow",
    "lost":      "dark_orange",
}

COLUMNS = ("ID", "STATUS", "G", "HOST", "SERVER", "GPU", "SUBMITTED", "RUNTIME", "LABEL / COMMAND")


def task_row(r) -> tuple:
    """Cells for one task in the queue table."""
    return (
        str(r["id"]),
        Text(r["status"], style=STATUS_STYLE.get(r["status"], "")),
        str(r["num_gpus"]),
        r["host_spec"] or "-",
        r["server"] or "-",
        r["gpu_indices"] or "-",
        core.fmt_ts(r["submitted_at"]),
        core.fmt_duration(core.runtime(r)),
        core.one_line(r["name"] or r["command"]),
    )


def render_gpu_map(snapshot: Optional[dict], running_rows, worker_names: list[str],
                   now: Optional[float] = None) -> "Text":
    """
    Hosts as rows, GPUs as cells: free, ours (task id) or another user's.
    `running_rows` are our running tasks (server, gpu_indices, id).
    """
    now = now or time.time()
    t = Text()
    if not snapshot or not snapshot.get("hosts"):
        t.append("No GPU snapshot yet. ", style="bold")
        t.append("The daemon writes one while tui is open"
                 + ("" if core.daemon_is_running() else " (daemon stopped: press d)") + ".",
                 style="dim")
        return t

    ours: dict[tuple[str, int], int] = {}
    for r in running_rows:
        for idx in (r["gpu_indices"] or "").split(","):
            if idx.strip().isdigit():
                ours[(r["server"], int(idx))] = r["id"]

    t.append("  free ", style="black on green")
    t.append(" ")
    t.append(" ours ", style="white on blue")
    t.append(" ")
    t.append(" in use ", style="white on red")
    t.append(f"   updated {core.fmt_duration(now - snapshot.get('ts', now))} ago"
             f" · network {snapshot.get('mode', '?')}\n\n", style="dim")

    hosts = {n: result_from_dict(h) for n, h in snapshot["hosts"].items()}
    names = [n for n in worker_names if n in hosts] + sorted(set(hosts) - set(worker_names))
    width = max((len(n) for n in names), default=6) + 2
    for name in names:
        h = hosts[name]
        t.append(f"{name:<{width}}", style="bold")
        if not h.reachable:
            t.append("unreachable", style="dim red")
            if h.error:
                t.append(f"  {core.truncate(h.error, 60)}", style="dim")
            t.append("\n")
            continue
        if not h.gpus:
            t.append("no GPUs (CPU only)", style="dim")
        for g in h.gpus:
            tid = ours.get((name, g.index))
            if tid is not None:
                t.append(f" #{tid:<4}", style="white on blue")
            elif g.is_free:
                t.append(f" {g.index:<5}", style="black on green")
            else:
                user = next((p.user for p in g.processes if p.user != "?"), "?")
                t.append(f" {core.truncate(user, 5):<5}", style="white on red")
            t.append(" ")
        if h.gpus:
            free = sum(1 for g in h.gpus if g.is_free and (name, g.index) not in ours)
            t.append(f"  {free}/{h.total_gpus} free  {h.gpus[0].name}", style="dim")
        t.append("\n")
    return t


if App is not None:

    class ConfirmScreen(ModalScreen[bool]):
        BINDINGS = [("y", "answer(True)", "Yes"), ("n,escape", "answer(False)", "No")]

        def __init__(self, question: str):
            super().__init__()
            self.question = question

        def compose(self) -> ComposeResult:
            with Vertical(id="dialog"):
                yield Label(self.question)
                with Horizontal(id="buttons"):
                    yield Button("Yes (y)", variant="error", id="yes")
                    yield Button("No (n)", id="no")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            self.dismiss(event.button.id == "yes")

        def action_answer(self, value: bool) -> None:
            self.dismiss(value)

    class SubmitScreen(ModalScreen[Optional[dict]]):
        """Form with the same fields as `task submit -g/-m/-H/-n/-C CMD`."""

        BINDINGS = [("escape", "cancel", "Cancel")]

        def compose(self) -> ComposeResult:
            with Vertical(id="dialog"):
                yield Label("New task", classes="title")
                with Grid(id="form"):
                    yield Label("Command")
                    yield Input(placeholder="python3 train.py --epochs 50", id="command")
                    yield Label("GPUs (-G)")
                    yield Input("1", id="gpus", type="integer")
                    yield Label("Min VRAM GB (-m)")
                    yield Input("0", id="min_mem", type="integer")
                    yield Label("Hosts (-H)")
                    yield Input(placeholder="all workers, or e.g. cuda14,local", id="host")
                    yield Label("Label (-n)")
                    yield Input(id="name")
                    yield Label("Directory (-d)")
                    yield Input(core.logical_cwd(), id="dir")
                with Horizontal(id="buttons"):
                    yield Button("Queue", variant="primary", id="ok")
                    yield Button("Cancel", id="cancel")

        def on_mount(self) -> None:
            self.query_one("#command", Input).focus()

        def _value(self, wid: str) -> str:
            return self.query_one(f"#{wid}", Input).value.strip()

        def _submit(self) -> None:
            if not self._value("command"):
                self.app.bell()
                return
            self.dismiss({
                "command": self._value("command"),
                "num_gpus": int(self._value("gpus") or 0),
                "min_mem": int(self._value("min_mem") or 0),
                "host_spec": self._value("host") or None,
                "name": self._value("name") or None,
                "working_dir": self._value("dir") or None,
            })

        def on_input_submitted(self, event: Input.Submitted) -> None:
            self._submit()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "ok":
                self._submit()
            else:
                self.dismiss(None)

        def action_cancel(self) -> None:
            self.dismiss(None)

    class SpoolerApp(App):
        TITLE = "tui"
        SUB_TITLE = "task spooler"
        CSS = """
        #tasks { height: 1fr; }
        #bottom { height: 1fr; }
        #gpus, #info { padding: 0 1; }
        ConfirmScreen, SubmitScreen { align: center middle; }
        #dialog {
            width: 80; height: auto; padding: 1 2;
            border: thick $accent; background: $surface;
        }
        #dialog .title { text-style: bold; margin-bottom: 1; }
        #form { grid-size: 2; grid-columns: 18 1fr; grid-rows: 3; height: auto; }
        #form Label { padding-top: 1; }
        #buttons { height: auto; margin-top: 1; align-horizontal: right; }
        #buttons Button { margin-left: 2; }
        """
        BINDINGS = [
            Binding("k", "kill", "Kill"),
            Binding("r", "resubmit", "Resubmit"),
            Binding("n", "new", "New"),
            Binding("s", "sync", "Sync repo"),
            Binding("f", "follow", "Follow"),
            Binding("l", "tab('log')", "Log"),
            Binding("g", "tab('gpus')", "GPUs"),
            Binding("i", "tab('info')", "Info"),
            Binding("c", "clear", "Clear done"),
            Binding("d", "start_daemon", "Start daemon"),
            Binding("q", "quit", "Quit"),
        ]

        def __init__(self):
            super().__init__()
            self.con = core.db_open()
            self._rows: dict[int, object] = {}
            self._table_sig: Optional[tuple] = None
            self._log_task: Optional[int] = None
            self._log_offset = 0

        def compose(self) -> ComposeResult:
            yield Header()
            yield DataTable(id="tasks", cursor_type="row", zebra_stripes=True)
            with TabbedContent(id="bottom", initial="log"):
                with TabPane("Log", id="log"):
                    yield Log(id="logview", max_lines=20000)
                with TabPane("GPUs", id="gpus"):
                    yield Static(id="gpumap")
                with TabPane("Info", id="info"):
                    yield Static(id="details")
            yield Footer()

        def on_mount(self) -> None:
            self.query_one(DataTable).add_columns(*COLUMNS)
            core.request_gpu_snapshot()
            self.refresh_tasks()
            self.refresh_gpus()
            self.set_interval(TASK_REFRESH, self.refresh_tasks)
            self.set_interval(LOG_REFRESH, self.refresh_log)
            self.set_interval(SNAPSHOT_REFRESH, self.refresh_gpus)
            self.set_interval(SNAPSHOT_PING, core.request_gpu_snapshot)

        # ── data refresh ─────────────────────────────────────────────────────

        def selected_id(self) -> Optional[int]:
            table = self.query_one(DataTable)
            if not table.row_count:
                return None
            try:
                key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
            except Exception:
                return None
            return int(key.value) if key and key.value else None

        def selected_row(self):
            tid = self.selected_id()
            return self._rows.get(tid) if tid is not None else None

        def refresh_tasks(self) -> None:
            rows = core.list_tasks(200, self.con)
            self._rows = {r["id"]: r for r in rows}
            cells = [task_row(r) for r in rows]
            sig = tuple(tuple(str(c) for c in row) for row in cells)
            if sig != self._table_sig:
                self._table_sig = sig
                table = self.query_one(DataTable)
                keep = self.selected_id()
                table.clear()
                for r, row in zip(rows, cells):
                    table.add_row(*row, key=str(r["id"]))
                if keep in self._rows:
                    table.move_cursor(row=table.get_row_index(str(keep)))
            self.refresh_header()
            self.refresh_details()

        def refresh_header(self) -> None:
            c = core.task_counts(self.con)
            daemon = "daemon running" if core.daemon_is_running() else "DAEMON STOPPED (d)"
            try:
                net = core.get_transport().describe_mode(probe=False)
            except ConfigError:
                net = "?"
            self.sub_title = (f"{c['running']} running · {c['queued']} queued · {daemon} · "
                              f"network {net}")

        def refresh_details(self) -> None:
            r = self.selected_row()
            details = self.query_one("#details", Static)
            if r is None:
                details.update("No task selected.")
                return
            t = Text()
            for label, value in core.task_details(r):
                t.append(f"{label + ':':<13}", style="bold")
                t.append(value + "\n", style=STATUS_STYLE.get(value, "") if label == "Status" else "")
            details.update(t)

        def refresh_log(self) -> None:
            log = self.query_one(Log)
            r = self.selected_row()
            tid = r["id"] if r is not None else None
            path = core.log_path(r) if r is not None else None
            if tid != self._log_task:
                self._log_task, self._log_offset = tid, 0
                log.clear()
                if path is None or not path.exists():
                    log.write_line("(no output yet)" if r is not None else "")
                    self._log_offset = -1
                    return
                text, self._log_offset = core.read_log_tail(path)
                log.write(text)
                return
            if path is None or not path.exists():
                return
            if self._log_offset < 0:             # log appeared since we looked
                log.clear()
                self._log_offset = 0
            if path.stat().st_size < self._log_offset:   # rewritten (retry)
                log.clear()
                self._log_offset = 0
            text, self._log_offset = core.read_log_from(path, self._log_offset)
            if text:
                log.write(text)

        def refresh_gpus(self) -> None:
            running = [r for r in self._rows.values() if r["status"] == "running"]
            try:
                workers = [h.name for h in core.get_cfg().workers()]
            except ConfigError:
                workers = []
            self.query_one("#gpumap", Static).update(
                render_gpu_map(core.read_gpu_snapshot(), running, workers))

        def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
            self.refresh_log()
            self.refresh_details()

        # ── actions ──────────────────────────────────────────────────────────

        def action_tab(self, tab: str) -> None:
            self.query_one(TabbedContent).active = tab
            if tab == "gpus":
                core.request_gpu_snapshot()

        def action_follow(self) -> None:
            self.query_one(TabbedContent).active = "log"
            log = self.query_one(Log)
            if self.screen.maximized is not None:
                self.screen.minimize()
            else:
                self.screen.maximize(log)
                log.scroll_end(animate=False)

        def action_kill(self) -> None:
            r = self.selected_row()
            if r is None or r["status"] not in core.ACTIVE:
                self.notify("Select a queued or running task.", severity="warning")
                return
            verb = "Cancel" if r["status"] == "queued" else "Kill"
            label = core.truncate(core.one_line(r["name"] or r["command"]), 50)

            def answer(yes: Optional[bool]) -> None:
                if yes:
                    self.do_kill(r["id"])
            self.push_screen(ConfirmScreen(f"{verb} task {r['id']}: {label}?"), answer)

        @work(thread=True)
        def do_kill(self, task_id: int) -> None:
            res = core.kill(task_id)
            self.call_from_thread(
                self.notify, res.message,
                severity="information" if res.confirmed else "warning",
                timeout=8 if res.confirmed else 20)

        def action_resubmit(self) -> None:
            r = self.selected_row()
            if r is None:
                return
            self.do_submit(lambda: core.resubmit(r["id"]))

        def action_new(self) -> None:
            def submitted(form: Optional[dict]) -> None:
                if form:
                    self.do_submit(lambda: core.submit(**form))
            self.push_screen(SubmitScreen(), submitted)

        @work(thread=True)
        def do_submit(self, fn) -> None:
            try:
                sub = fn()
            except (core.TaskError, ConfigError) as e:
                self.call_from_thread(self.notify, str(e), severity="error")
                return
            msg = f"Queued task {sub.task_id}: {core.truncate(sub.command, 60)}"
            for w in sub.warnings:
                msg += f"\nWarning: {w}"
            self.call_from_thread(self.notify, msg,
                                  severity="warning" if sub.warnings else "information")

        def action_sync(self) -> None:
            """Same as `task sync` from the directory tui was started in."""
            self.notify("Syncing git repo...")
            self.do_sync()

        @work(thread=True, exclusive=True, group="sync")
        def do_sync(self) -> None:
            try:
                info = core.repo_for()
                results = core.sync_repo(info)
            except (core.TaskError, ConfigError) as e:
                self.call_from_thread(self.notify, str(e), severity="error", timeout=10)
                return
            lines = [f"{'OK' if r.ok else 'FAILED'} {r.target.label}: {r.message}" for r in results]
            busy = core.running_in([t.path for t in info.copies])
            if busy:
                lines.append(f"{busy} running task(s) use this checkout.")
            ok = all(r.ok for r in results)
            self.call_from_thread(self.notify, "\n".join(lines),
                                  severity="information" if ok else "error", timeout=12)

        def action_clear(self) -> None:
            def answer(yes: Optional[bool]) -> None:
                if yes:
                    self.do_clear()
            self.push_screen(ConfirmScreen(
                "Remove all finished tasks, their logs and job directories?"), answer)

        @work(thread=True)
        def do_clear(self) -> None:
            res = core.clear_finished()
            msg = f"Removed {res.tasks} finished task(s)."
            if res.failed_hosts:
                msg += f" Could not clean job directories on: {', '.join(res.failed_hosts)}"
            self.call_from_thread(self.notify, msg)

        def action_start_daemon(self) -> None:
            pid = core.start_daemon()
            self.notify(f"Daemon started (PID {pid})." if pid else "Daemon is already running.")
            core.request_gpu_snapshot()


def main() -> None:
    if App is None:
        sys.exit("tui needs Textual. Install it with:\n"
                 "  pip install -e '.[tui]'     (from the gpu-spooler checkout)\n"
                 "or re-run install.sh.")
    try:
        core.get_cfg()
    except ConfigError as e:
        sys.exit(f"tui: config error: {e}")
    SpoolerApp().run()


if __name__ == "__main__":
    main()
