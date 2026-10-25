//! GPUs held but idle, read from gpu-detective's history.
//!
//! Opt-in with a [detective] table in hosts.toml. gpu-spooler never links gpu-detective: the
//! contract is the history's SQLite schema (snapshots, gpus, procs), opened read-only. A missing or
//! unreadable history just means no idle information.

use rusqlite::{Connection, OpenFlags};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::time::Duration;

/// Same threshold as gpu-detective.
pub const IDLE_UTIL_PCT: i64 = 5;
/// A host's latest snapshot must be this recent to count.
pub const STALE: f64 = 600.0;
pub const MIN_LOOKBACK: f64 = 6.0 * 3600.0;

#[derive(Clone, Debug, PartialEq)]
pub struct IdleGpu {
    pub host: String,
    pub index: i64,
    pub idle_seconds: f64,
    pub users: Vec<String>,
    /// idle for at least the lookback: idle_seconds is a lower bound
    pub capped: bool,
}

/// (host, gpu index) → idle GPU.
pub type Idle = BTreeMap<(String, i64), IdleGpu>;

/// Every GPU that, in the latest sweep, holds processes at under IDLE_UTIL_PCT utilisation and has
/// done so, with the same users, in every sweep for at least `min_seconds`.
pub fn idle_gpus(path: &Path, min_seconds: f64, now: f64) -> Idle {
    read(path, min_seconds, now).unwrap_or_default()
}

type Users = HashMap<i64, HashMap<i64, BTreeSet<String>>>;

fn read(path: &Path, min_seconds: f64, now: f64) -> rusqlite::Result<Idle> {
    let lookback = MIN_LOOKBACK.max(2.0 * min_seconds);
    let con = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)?;
    con.busy_timeout(Duration::from_secs(2))?;
    let snaps: Vec<(i64, f64, String, bool)> = con
        .prepare("SELECT id, ts, host, reachable FROM snapshots WHERE ts >= ? ORDER BY host, ts")?
        .query_map([now - lookback], |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                r.get::<_, Option<i64>>(3)?.unwrap_or(0) != 0,
            ))
        })?
        .collect::<Result<_, _>>()?;
    let Some(lo) = snaps.iter().map(|s| s.0).min() else { return Ok(Idle::new()) };

    // snapshot id → gpu index → utilisation (None: NULL), in row order
    let mut util: HashMap<i64, Vec<(i64, Option<i64>)>> = HashMap::new();
    let mut stmt = con.prepare("SELECT snapshot_id, idx, util_pct FROM gpus WHERE snapshot_id >= ?")?;
    for row in stmt.query_map([lo], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, Option<i64>>(2)?)))? {
        let (sid, idx, u) = row?;
        let gpus = util.entry(sid).or_default();
        match gpus.iter_mut().find(|(i, _)| *i == idx) {
            Some(slot) => slot.1 = u,
            None => gpus.push((idx, u)),
        }
    }
    let mut users: Users = HashMap::new();
    let mut stmt = con.prepare("SELECT snapshot_id, gpu_idx, user FROM procs WHERE snapshot_id >= ?")?;
    for row in
        stmt.query_map([lo], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, Option<String>>(2)?)))?
    {
        let (sid, idx, user) = row?;
        users.entry(sid).or_default().entry(idx).or_default().insert(user.unwrap_or_default());
    }

    let mut series: BTreeMap<String, Vec<(i64, f64, bool)>> = BTreeMap::new();
    for (sid, ts, host, reachable) in snaps {
        series.entry(host).or_default().push((sid, ts, reachable));
    }
    let empty = BTreeSet::new();
    let holders = |sid: i64, idx: i64| users.get(&sid).and_then(|g| g.get(&idx)).unwrap_or(&empty);
    let is_idle = |sid: i64, idx: i64| {
        let u = util.get(&sid).and_then(|g| g.iter().find(|(i, _)| *i == idx)).and_then(|(_, u)| *u);
        matches!(u, Some(u) if (0..IDLE_UTIL_PCT).contains(&u)) && !holders(sid, idx).is_empty()
    };

    let mut out = Idle::new();
    for (host, ss) in &series {
        let &(sid, ts, reachable) = ss.last().expect("series are never empty");
        if !reachable || ts < now - STALE {
            continue;
        }
        for &(idx, _) in util.get(&sid).map_or(&[][..], |g| g.as_slice()) {
            if !is_idle(sid, idx) {
                continue;
            }
            let who = holders(sid, idx);
            let (mut since, mut capped) = (ts, true);
            for &(psid, pts, preach) in ss[..ss.len() - 1].iter().rev() {
                if !preach || !is_idle(psid, idx) || holders(psid, idx) != who {
                    capped = false;
                    break;
                }
                since = pts;
            }
            if ts - since >= min_seconds {
                out.insert(
                    (host.clone(), idx),
                    IdleGpu {
                        host: host.clone(),
                        index: idx,
                        idle_seconds: ts - since,
                        users: who.iter().cloned().collect(),
                        capped: capped && ss.len() > 1,
                    },
                );
            }
        }
    }
    Ok(out)
}

/// `idle 59m+`, `idle 1h05m`.
pub fn fmt_idle(g: &IdleGpu) -> String {
    let s = g.idle_seconds as i64;
    let text = if s < 3600 { format!("{}m", s / 60) } else { format!("{}h{:02}m", s / 3600, s % 3600 / 60) };
    format!("idle {text}{}", if g.capped { "+" } else { "" })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &str = "
        CREATE TABLE snapshots (id INTEGER PRIMARY KEY, ts REAL, host TEXT, reachable INTEGER, error TEXT DEFAULT '');
        CREATE TABLE gpus (snapshot_id INTEGER, idx INTEGER, name TEXT, mem_used_mb INTEGER,
                           mem_free_mb INTEGER, mem_total_mb INTEGER, util_pct INTEGER);
        CREATE TABLE procs (snapshot_id INTEGER, gpu_idx INTEGER, pid INTEGER, user TEXT,
                            mem_mb INTEGER, started REAL, command TEXT);";

    /// (gpu, util, users)
    type Gpus<'a> = &'a [(i64, i64, &'a [&'a str])];

    /// sweeps: (ts, host, reachable, gpus)
    fn history(path: &Path, sweeps: &[(f64, &str, bool, Gpus)]) {
        let con = Connection::open(path).unwrap();
        con.execute_batch(SCHEMA).unwrap();
        for (ts, host, reachable, gpus) in sweeps {
            con.execute("INSERT INTO snapshots (ts, host, reachable) VALUES (?, ?, ?)", (ts, host, *reachable as i64))
                .unwrap();
            let sid = con.last_insert_rowid();
            for (idx, util, users) in gpus.iter() {
                con.execute("INSERT INTO gpus VALUES (?, ?, 'A40', 1, 1, 2, ?)", (sid, idx, util)).unwrap();
                for (i, u) in users.iter().enumerate() {
                    con.execute("INSERT INTO procs VALUES (?, ?, ?, ?, 1, 0, 'x')", (sid, idx, 100 + i as i64, u))
                        .unwrap();
                }
            }
        }
    }

    #[test]
    fn idle_gpus_found() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("h.db");
        let now = 100000.0;
        history(
            &db,
            &[
                (now - 3600.0, "a", true, &[(0, 90, &["alice"]), (1, 0, &["bob"])]),
                (now - 2400.0, "a", true, &[(0, 0, &["alice"]), (1, 0, &["bob"])]),
                (now - 1200.0, "a", true, &[(0, 0, &["alice"]), (1, 0, &["bob"])]),
                (now - 60.0, "a", true, &[(0, 0, &["alice"]), (1, 0, &["bob"]), (2, 0, &[])]),
                (now - 60.0, "b", true, &[(0, 0, &["carol"])]), // one sweep only: too short
                (now - 4000.0, "c", true, &[(0, 0, &["dave"])]), // stale host
            ],
        );
        let idle = idle_gpus(&db, 30.0 * 60.0, now);
        assert_eq!(idle.keys().cloned().collect::<Vec<_>>(), [("a".to_string(), 0), ("a".to_string(), 1)]);
        let a0 = &idle[&("a".to_string(), 0)];
        assert_eq!(a0.idle_seconds, 2400.0 - 60.0);
        assert!(!a0.capped);
        assert_eq!(a0.users, ["alice"]);
        let a1 = &idle[&("a".to_string(), 1)];
        assert!(a1.capped);
        assert_eq!(fmt_idle(a1), "idle 59m+");
    }

    #[test]
    fn breaks_on_new_holder_or_outage() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("h.db");
        let now = 100000.0;
        history(
            &db,
            &[
                (now - 3000.0, "a", true, &[(0, 0, &["alice"]), (1, 0, &["bob"])]),
                (now - 2000.0, "a", false, &[]),
                (now - 1000.0, "a", true, &[(0, 0, &["erin"]), (1, 0, &["bob"])]),
                (now - 60.0, "a", true, &[(0, 0, &["erin"]), (1, 0, &["bob"])]),
            ],
        );
        assert_eq!(idle_gpus(&db, 600.0, now).len(), 2);
        assert!(idle_gpus(&db, 1200.0, now).is_empty());
    }

    #[test]
    fn missing_or_bad_history_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(idle_gpus(&dir.path().join("nope.db"), 60.0, 0.0).is_empty());
        std::fs::write(dir.path().join("junk.db"), "not sqlite").unwrap();
        assert!(idle_gpus(&dir.path().join("junk.db"), 60.0, 0.0).is_empty());
        assert!(!dir.path().join("nope.db").exists(), "opening read-only must not create the file");
    }

    #[test]
    fn formats() {
        let g = |s: f64, capped| IdleGpu { host: "h".into(), index: 0, idle_seconds: s, users: vec![], capped };
        assert_eq!(fmt_idle(&g(3900.0, false)), "idle 1h05m");
        assert_eq!(fmt_idle(&g(59.0, false)), "idle 0m");
    }
}
