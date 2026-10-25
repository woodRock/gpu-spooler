//! The golden fixtures from the Python implementation. Each migration step adds tests here that
//! compare a ported module against them (see dev/gen_golden.py for what each file holds).

mod common;

use std::collections::BTreeSet;

#[test]
fn index_lists_every_fixture() {
    let listed: BTreeSet<String> = common::index().into_iter().collect();
    let mut found = BTreeSet::new();
    let mut stack = vec![common::dir()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let rel = path.strip_prefix(common::dir()).unwrap().to_string_lossy().into_owned();
                if rel != "INDEX" {
                    found.insert(rel);
                }
            }
        }
    }
    assert_eq!(listed, found, "tests/golden/parity/INDEX is stale; run dev/gen_golden.py");
}

#[test]
fn json_fixtures_are_named_cases() {
    for name in common::index().iter().filter(|n| n.ends_with(".json")) {
        let cases = common::cases(name);
        assert!(!cases.is_empty(), "{name}: no cases");
        let names: BTreeSet<&str> = cases.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names.len(), cases.len(), "{name}: duplicate case names");
    }
}

#[test]
fn script_cases_point_at_text_fixtures() {
    for name in ["jobs_launch.json", "jobs_scripts.json"] {
        for case in common::cases(name) {
            let file = case.output.as_str().unwrap_or_else(|| panic!("{name}/{}: output is not a file", case.name));
            assert!(!common::text(file).is_empty(), "{file} is empty");
        }
    }
}
