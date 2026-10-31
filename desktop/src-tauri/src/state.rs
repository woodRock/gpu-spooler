//! What the commands and the watcher share: the state paths and one SQLite connection.

use gpu_spooler::db;
use gpu_spooler::paths::Paths;
use rusqlite::Connection;
use std::sync::{mpsc, Mutex};

pub struct AppState {
    pub paths: Paths,
    /// Opened on first use and kept for the app's lifetime: opening and closing one per read would
    /// checkpoint the WAL whenever the daemon is stopped, and that is itself a change to the
    /// folder the watcher watches.
    con: Mutex<Option<Connection>>,
    /// Wakes the watcher for an immediate read (set once it runs).
    wake: Mutex<Option<mpsc::Sender<()>>>,
}

impl AppState {
    pub fn new(paths: Paths) -> AppState {
        AppState { paths, con: Mutex::new(None), wake: Mutex::new(None) }
    }

    pub fn set_wake(&self, tx: mpsc::Sender<()>) {
        *self.wake.lock().unwrap() = Some(tx);
    }

    /// Read again now, after an action changed the queue: file events can take a second or
    /// more to arrive, and the window should show the change as soon as the toast reports it.
    pub fn refresh(&self) {
        if let Some(tx) = self.wake.lock().unwrap().as_ref() {
            let _ = tx.send(());
        }
    }

    /// Run `f` on the shared connection, opening it if needed.
    pub fn with_con<T>(&self, f: impl FnOnce(&Connection) -> T) -> Result<T, String> {
        let mut con = self.con.lock().unwrap();
        if con.is_none() {
            *con =
                Some(db::open(&self.paths).map_err(|e| format!("could not open {}: {e}", self.paths.db().display()))?);
        }
        Ok(f(con.as_ref().expect("opened above")))
    }
}
