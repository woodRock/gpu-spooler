//! Access to the golden fixtures in tests/golden/parity/, written by `dev/gen_golden.py` from the
//! Python implementation. Placeholders: `{HOME}` for the home folder the generator used and
//! `{CM_DIR}` for its ControlMaster directory; times are UTC.

#![allow(dead_code)]

use serde_json::Value;
use std::fs;
use std::path::PathBuf;

pub fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/parity")
}

/// A text fixture, e.g. `text("jobs/kill.sh")`.
pub fn text(name: &str) -> String {
    let path = dir().join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// One case of a JSON fixture.
pub struct Case {
    pub name: String,
    pub input: Value,
    pub output: Value,
}

/// The cases of a JSON fixture, e.g. `cases("fmt_duration.json")`.
pub fn cases(name: &str) -> Vec<Case> {
    let value: Value = serde_json::from_str(&text(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let rows = value.as_array().unwrap_or_else(|| panic!("{name}: not a list of cases"));
    rows.iter()
        .map(|row| Case {
            name: row["name"].as_str().unwrap_or_else(|| panic!("{name}: case without a name")).to_string(),
            input: row["input"].clone(),
            output: row["output"].clone(),
        })
        .collect()
}

/// Every fixture the generator wrote, from its INDEX.
pub fn index() -> Vec<String> {
    text("INDEX").lines().map(str::to_string).collect()
}

/// A home folder for the port under test, and the placeholders the fixtures use for it.
pub struct Home {
    pub dir: tempfile::TempDir,
}

impl Home {
    pub fn new() -> Home {
        Home { dir: tempfile::tempdir().unwrap() }
    }

    pub fn paths(&self) -> gpu_spooler::paths::Paths {
        gpu_spooler::paths::Paths::for_home(self.dir.path())
    }

    /// Output of the port, with this home and its ControlMaster directory written as the fixtures
    /// write them.
    pub fn scrub(&self, s: &str) -> String {
        let cm = gpu_spooler::transport::cm_dir(&self.paths());
        s.replace(&cm.display().to_string(), "{CM_DIR}").replace(&self.dir.path().display().to_string(), "{HOME}")
    }

    /// A fixture input, with `{HOME}` pointing at this home.
    pub fn unscrub(&self, s: &str) -> String {
        s.replace("{HOME}", &self.dir.path().display().to_string())
    }
}
