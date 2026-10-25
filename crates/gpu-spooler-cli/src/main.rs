//! `task` - the gpu-spooler command line (Rust port, in progress).

use std::process::ExitCode;

fn main() -> ExitCode {
    eprintln!(
        "task {}: the Rust port of gpu-spooler is not usable yet; use the Python version (install.sh)",
        env!("CARGO_PKG_VERSION")
    );
    ExitCode::from(2)
}
