//! Drawing the tui: header, queue table, the Log / GPUs / Info tabs, footer, notes and dialogs.

use super::app::{App, Level, Modal, Tab};
use super::model::{bold, dim, task_row, COLUMNS};
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Row, Table, TableState, Wrap};
use ratatui::Frame;

const KEYS: [(&str, &str); 11] = [
    ("k", "Kill"),
    ("r", "Resubmit"),
    ("n", "New"),
    ("s", "Sync repo"),
    ("f", "Follow"),
    ("l", "Log"),
    ("g", "GPUs"),
    ("i", "Info"),
    ("c", "Clear done"),
    ("d", "Start daemon"),
    ("q", "Quit"),
];

fn header(app: &App) -> Paragraph<'static> {
    let line = Line::from(vec![Span::styled("tui", bold()), Span::raw(" — "), Span::raw(app.subtitle.clone())]);
    Paragraph::new(line).centered().style(Style::new().bg(Color::Blue).fg(Color::White))
}

fn footer() -> Paragraph<'static> {
    let mut spans = vec![];
    for (k, what) in KEYS {
        spans.push(Span::styled(
            format!(" {k} "),
            Style::new().bg(Color::DarkGray).fg(Color::White).add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::raw(format!(" {what}  ")));
    }
    Paragraph::new(Line::from(spans))
}

fn table(app: &App) -> Table<'static> {
    let rows: Vec<Row> = app
        .rows
        .iter()
        .map(|t| Row::new(task_row(t).into_iter().map(|(text, style)| Span::styled(text, style)).collect::<Vec<_>>()))
        .collect();
    let widths = [
        Constraint::Length(5),
        Constraint::Length(10),
        Constraint::Length(2),
        Constraint::Length(10),
        Constraint::Length(14),
        Constraint::Length(7),
        Constraint::Length(11),
        Constraint::Length(8),
        Constraint::Fill(1),
    ];
    Table::new(rows, widths)
        .header(Row::new(COLUMNS.map(|c| Span::styled(c, bold()))))
        .row_highlight_style(Style::new().bg(Color::Blue).fg(Color::White).add_modifier(Modifier::BOLD))
        .column_spacing(1)
}

fn tabs(app: &App) -> Line<'static> {
    let tab = |name: &str, key: &str, t: Tab| {
        let style = if app.tab == t {
            Style::new().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            dim()
        };
        vec![Span::styled(format!(" {name} ({key}) "), style), Span::raw(" ")]
    };
    Line::from([tab("Log", "l", Tab::Log), tab("GPUs", "g", Tab::Gpus), tab("Info", "i", Tab::Info)].concat())
}

/// The log's last lines that fit, `scroll` lines up from the end.
fn log_view(app: &App, area: Rect) -> Paragraph<'static> {
    let lines: Vec<&str> = app.log.lines().collect();
    let h = area.height as usize;
    let end = lines.len().saturating_sub(app.log_scroll.min(lines.len().saturating_sub(h.min(lines.len()))));
    let start = end.saturating_sub(h);
    Paragraph::new(lines[start..end].iter().map(|l| Line::raw(l.to_string())).collect::<Vec<_>>())
}

fn gpu_view(app: &App) -> Paragraph<'static> {
    let mut lines: Vec<Line> = vec![Line::default()];
    for (text, style) in &app.gpu_map {
        let mut parts = text.split('\n').peekable();
        while let Some(part) = parts.next() {
            if !part.is_empty() {
                lines.last_mut().unwrap().spans.push(Span::styled(part.to_string(), *style));
            }
            if parts.peek().is_some() {
                lines.push(Line::default());
            }
        }
    }
    Paragraph::new(Text::from(lines))
}

fn info_view(app: &App) -> Paragraph<'static> {
    if app.selected_task().is_none() {
        return Paragraph::new("No task selected.");
    }
    let lines: Vec<Line> = app
        .details
        .iter()
        .map(|(label, value)| {
            let style = if label == "Status" { super::model::status_style(value) } else { Style::new() };
            Line::from(vec![
                Span::styled(format!("{:<13}", format!("{label}:")), bold()),
                Span::styled(value.clone(), style),
            ])
        })
        .collect();
    Paragraph::new(lines).wrap(Wrap { trim: false })
}

fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let [v] = Layout::vertical([Constraint::Length(h.min(area.height))]).flex(Flex::Center).areas(area);
    let [r] = Layout::horizontal([Constraint::Length(w.min(area.width))]).flex(Flex::Center).areas(v);
    r
}

fn dialog(f: &mut Frame, app: &App) {
    let Some(modal) = &app.modal else { return };
    let block = Block::default().borders(Borders::ALL).border_style(Style::new().fg(Color::Cyan)).style(Style::new());
    match modal {
        Modal::Confirm { question, .. } => {
            let area = centered(f.area(), 80, 6);
            f.render_widget(Clear, area);
            let body = vec![
                Line::raw(question.clone()),
                Line::default(),
                Line::from(vec![
                    Span::styled(
                        " Yes (y) ",
                        Style::new().bg(Color::Red).fg(Color::White).add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("  "),
                    Span::styled(" No (n) ", Style::new().bg(Color::DarkGray).fg(Color::White)),
                ])
                .right_aligned(),
            ];
            f.render_widget(
                Paragraph::new(body)
                    .wrap(Wrap { trim: false })
                    .block(block.padding(ratatui::widgets::Padding::horizontal(1))),
                area,
            );
        }
        Modal::Submit(form) => {
            let area = centered(f.area(), 80, form.fields.len() as u16 + 6);
            f.render_widget(Clear, area);
            let mut body = vec![Line::styled("New task", bold()), Line::default()];
            for (i, (label, value, placeholder)) in form.fields.iter().enumerate() {
                let focused = i == form.focus;
                let shown = if value.is_empty() && !focused {
                    Span::styled(placeholder.to_string(), dim())
                } else {
                    Span::styled(
                        format!("{value}{}", if focused { "▏" } else { "" }),
                        if focused { Style::new().add_modifier(Modifier::UNDERLINED) } else { Style::new() },
                    )
                };
                let marker = if focused { Span::styled("› ", Style::new().fg(Color::Cyan)) } else { Span::raw("  ") };
                body.push(Line::from(vec![marker, Span::styled(format!("{label:<18}"), bold()), shown]));
            }
            body.push(Line::default());
            body.push(Line::styled("Enter queue · Tab/↑↓ move · Esc cancel", dim()).right_aligned());
            f.render_widget(Paragraph::new(body).block(block.padding(ratatui::widgets::Padding::horizontal(1))), area);
        }
    }
}

fn notes(f: &mut Frame, app: &App, above: u16) {
    let mut bottom = above;
    for n in app.notes.iter().rev().take(3) {
        let text: Vec<Line> = n.text.lines().map(|l| Line::raw(l.to_string())).collect();
        let w = (n.text.lines().map(|l| l.chars().count()).max().unwrap_or(0) as u16 + 4).clamp(20, 70);
        let h = text.len() as u16 + 2;
        if bottom < h + 1 {
            break;
        }
        let area = f.area();
        let rect = Rect { x: area.width.saturating_sub(w + 1), y: bottom - h, width: w.min(area.width), height: h };
        let color = match n.level {
            Level::Info => Color::Green,
            Level::Warning => Color::Yellow,
            Level::Error => Color::Red,
        };
        f.render_widget(Clear, rect);
        let block = Block::default()
            .borders(Borders::LEFT | Borders::TOP | Borders::BOTTOM)
            .border_style(Style::new().fg(color));
        f.render_widget(Paragraph::new(text).wrap(Wrap { trim: false }).block(block), rect);
        bottom -= h;
    }
}

pub fn draw(f: &mut Frame, app: &App) {
    let [top, body, foot] =
        Layout::vertical([Constraint::Length(1), Constraint::Fill(1), Constraint::Length(1)]).areas(f.area());
    f.render_widget(header(app), top);
    f.render_widget(footer(), foot);
    if app.follow {
        f.render_widget(log_view(app, body), body);
    } else {
        let [list, bar, pane] =
            Layout::vertical([Constraint::Fill(1), Constraint::Length(1), Constraint::Fill(1)]).areas(body);
        let mut state = TableState::default().with_selected((!app.rows.is_empty()).then_some(app.selected));
        f.render_stateful_widget(table(app), list, &mut state);
        f.render_widget(Paragraph::new(tabs(app)), bar);
        let inner = Rect { x: pane.x + 1, width: pane.width.saturating_sub(2), ..pane };
        match app.tab {
            Tab::Log => f.render_widget(log_view(app, inner), inner),
            Tab::Gpus => f.render_widget(gpu_view(app), inner),
            Tab::Info => f.render_widget(info_view(app), inner),
        }
    }
    notes(f, app, foot.y);
    dialog(f, app);
}
