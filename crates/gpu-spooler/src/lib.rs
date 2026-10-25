//! gpu-spooler - queue jobs onto SSH-reachable GPU servers.
//!
//! The Rust port of the Python `gpu_spooler` package, in progress: modules arrive one migration
//! step at a time, and until the port is done the Python version is the one that runs.
//! Everything sent to the servers must match the Python version byte for byte; `tests/golden.rs`
//! checks the port against fixtures generated from the Python code (`dev/gen_golden.py`).
//!
//! gpu-detective uses [`config`] (hosts.toml), [`transport`] (the ssh route) and [`gpu`] (the
//! nvidia-smi query and its parser).

pub mod config;
pub mod core;
pub mod db;
pub mod fmt;
pub mod gpu;
pub mod idle;
pub mod paths;
pub mod py;
pub mod shell;
pub mod transport;
