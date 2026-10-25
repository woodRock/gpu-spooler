//! `task tui`: an interactive view of the queue, logs and GPUs.
//!
//! A front end over gpu_spooler::core, like the CLI: it reads the same database and the GPU
//! snapshot the daemon writes, and never runs its own SSH fan-out (kill, clear and sync do talk to
//! the servers, exactly as their commands do, on a worker thread).
//!
//! Keys: k kill/cancel · r resubmit · n new task · s sync git repo · f follow log (full screen) ·
//! l log · g GPUs · i info · c clear finished · d start daemon · q quit

mod app;
mod model;
mod view;

use crate::ctx::{CmdResult, Ctx, Fail};
use ratatui::crossterm::event::{self, Event, KeyEventKind};
use std::time::Duration;

pub fn run(ctx: &Ctx) -> CmdResult {
    if let Err(e) = gpu_spooler::config::load_config(&ctx.paths) {
        return Err(Fail::new(format!("tui: config error: {e}")));
    }
    let mut app = app::App::new(ctx.paths.clone(), gpu_spooler::core::logical_cwd());
    app.tick(true);
    ratatui::run(|t| -> std::io::Result<()> {
        while !app.quit {
            t.draw(|f| view::draw(f, &app))?;
            if app.bell {
                app.bell = false;
                print!("\x07");
            }
            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(k) = event::read()? {
                    if k.kind == KeyEventKind::Press {
                        app.key(k);
                    }
                }
            }
            app.tick(false);
        }
        Ok(())
    })?;
    Ok(0)
}

#[cfg(test)]
mod tests;
