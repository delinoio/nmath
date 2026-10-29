//! Structured `tracing` logs on stderr.
//!
//! `NMATH_LOG` enables logging and takes an `EnvFilter` directive (for example
//! `debug` or `nmath_compiler=trace`). ANSI color is on by default; a non-empty
//! `NO_COLOR` turns it off. Logs never contain source text.

use tracing_subscriber::EnvFilter;

pub const LOG_ENV: &str = "NMATH_LOG";

pub fn init() {
    let Some(directive) = std::env::var(LOG_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
    else {
        return;
    };
    let ansi = std::env::var_os("NO_COLOR").is_none_or(|value| value.is_empty());
    let filter = EnvFilter::try_new(&directive).unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(ansi)
        .with_writer(std::io::stderr)
        .try_init();
}
