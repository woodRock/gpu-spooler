//! Where gpu-spooler keeps its state: ~/.gpu-spooler/, and the hosts.toml in use.

use std::env;
use std::path::{Path, PathBuf};

/// The home folder and what is derived from it. Built from the environment ([`Paths::from_env`]),
/// or from any folder in tests ([`Paths::for_home`]), so nothing reads `$HOME` behind your back.
#[derive(Clone, Debug)]
pub struct Paths {
    pub home: PathBuf,
    /// ~/.gpu-spooler
    pub state: PathBuf,
    /// The config file: `$GPU_SPOOLER_CONFIG`, or ~/.gpu-spooler/hosts.toml.
    pub config: PathBuf,
    /// `$GPU_DETECTIVE_DB`, gpu-detective's history (the [detective] default).
    pub detective_db: Option<String>,
}

impl Paths {
    pub fn from_env() -> Paths {
        let home = env::var_os("HOME").map(PathBuf::from).filter(|h| !h.as_os_str().is_empty());
        #[allow(deprecated)] // fine on Unix, the only platform this runs on
        let home = home.or_else(env::home_dir).unwrap_or_else(|| PathBuf::from("/"));
        let mut paths = Paths::for_home(home);
        if let Some(c) = env::var("GPU_SPOOLER_CONFIG").ok().filter(|c| !c.is_empty()) {
            paths.config = PathBuf::from(expand_user(&c, &paths.home));
        }
        paths.detective_db = env::var("GPU_DETECTIVE_DB").ok().filter(|d| !d.is_empty());
        paths
    }

    pub fn for_home(home: impl Into<PathBuf>) -> Paths {
        let home = home.into();
        let state = home.join(".gpu-spooler");
        Paths { config: state.join("hosts.toml"), state, home, detective_db: None }
    }

    pub fn network_cache(&self) -> PathBuf {
        self.state.join("network.json")
    }

    pub fn expand_user(&self, path: &str) -> String {
        expand_user(path, &self.home)
    }
}

/// `os.path.expanduser` for `~` and `~/...` (a `~user` prefix is left as it is).
pub fn expand_user(path: &str, home: &Path) -> String {
    if path != "~" && !path.starts_with("~/") {
        return path.to_string();
    }
    let home = home.to_string_lossy();
    let home = home.trim_end_matches('/');
    format!("{home}{}", &path[1..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_only_own_home() {
        let h = Path::new("/home/me/");
        assert_eq!(expand_user("~", h), "/home/me");
        assert_eq!(expand_user("~/x/y", h), "/home/me/x/y");
        assert_eq!(expand_user("~bob/x", h), "~bob/x");
        assert_eq!(expand_user("/abs", h), "/abs");
    }
}
