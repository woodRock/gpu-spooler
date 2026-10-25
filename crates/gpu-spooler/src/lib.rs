//! gpu-spooler - queue jobs onto SSH-reachable GPU servers.
//!
//! The Rust port of the Python `gpu_spooler` package, in progress. Modules arrive one migration
//! step at a time; until then the Python version is the one that runs. Everything sent to the
//! servers must match the Python version byte for byte: `tests/golden.rs` checks the port against
//! fixtures generated from the Python code (`dev/gen_golden.py`).
