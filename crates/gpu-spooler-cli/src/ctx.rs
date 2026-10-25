//! What a command works with: the state paths, and the config and ssh transport, loaded the
//! first time a command needs them (as core.get_cfg does, so a broken hosts.toml only stops the
//! commands that read it).

use gpu_spooler::config::{load_config, Config};
use gpu_spooler::paths::Paths;
use gpu_spooler::transport::Transport;
use std::sync::OnceLock;

/// How a command fails: the message for stderr (already prefixed, e.g. `task: ...`) and the
/// exit code.
#[derive(Debug)]
pub struct Fail {
    pub message: String,
    pub code: i32,
}

impl Fail {
    pub fn new(message: impl Into<String>) -> Fail {
        Fail { message: message.into(), code: 1 }
    }

    /// Exit with this code and print nothing (sys.exit(n)).
    pub fn silent(code: i32) -> Fail {
        Fail { message: String::new(), code }
    }
}

impl From<gpu_spooler::core::TaskError> for Fail {
    fn from(e: gpu_spooler::core::TaskError) -> Fail {
        Fail::new(format!("task: {e}"))
    }
}

impl From<rusqlite::Error> for Fail {
    fn from(e: rusqlite::Error) -> Fail {
        Fail::new(format!("task: {e}"))
    }
}

impl From<std::io::Error> for Fail {
    fn from(e: std::io::Error) -> Fail {
        Fail::new(format!("task: {e}"))
    }
}

pub type CmdResult = Result<i32, Fail>;

pub struct Ctx {
    pub paths: Paths,
    transport: OnceLock<Result<Transport, String>>,
}

impl Ctx {
    pub fn from_env() -> Ctx {
        Ctx { paths: Paths::from_env(), transport: OnceLock::new() }
    }

    pub fn transport(&self) -> Result<&Transport, Fail> {
        self.transport
            .get_or_init(|| {
                load_config(&self.paths).map(|cfg| Transport::new(cfg, self.paths.clone())).map_err(|e| e.0)
            })
            .as_ref()
            .map_err(|e| Fail::new(format!("task: config error: {e}")))
    }

    pub fn cfg(&self) -> Result<&Config, Fail> {
        Ok(&self.transport()?.cfg)
    }

    pub fn db(&self) -> Result<rusqlite::Connection, Fail> {
        Ok(gpu_spooler::db::open(&self.paths)?)
    }
}
