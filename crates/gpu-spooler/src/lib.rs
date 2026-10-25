//! gpu-spooler - queue jobs onto SSH-reachable GPU servers.
//!
//! A rewrite of the Python gpu-spooler (0.1, tag `v0.1-python`) that keeps its state, config and
//! everything it sends to the servers byte for byte, so either version adopts the other's jobs.
//! `tests/golden.rs` checks that against fixtures generated from the Python code.
//!
//! gpu-detective uses [`config`] (hosts.toml), [`transport`] (the ssh route) and [`gpu`] (the
//! nvidia-smi query and its parser).

pub mod config;
pub mod core;
pub mod daemon;
pub mod db;
pub mod fetch;
pub mod fmt;
pub mod gpu;
pub mod idle;
pub mod jobs;
pub mod paths;
pub mod provenance;
pub mod py;
pub mod repo;
pub mod shell;
pub mod transport;
