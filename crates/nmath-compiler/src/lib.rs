//! The NMath compiler: `check`, `build`, `lsp` and the internal `.nts` content
//! mapper.
//!
//! Behavior contract: `docs/crates-nmath-compiler-contract.md`.

pub mod cli;
pub mod content_mapper;
pub mod framing;
pub mod logging;
pub mod lsp;
pub mod project;
pub mod result;

pub use project::{MapperLaunch, ProjectCommand, ProjectOptions};
pub use result::{CompilerCode, CompilerDiagnostic, CompilerResult, ExitCode};

/// Type-checks a project.
pub fn check(options: &ProjectOptions) -> CompilerResult {
    project::run(ProjectCommand::Check, options)
}

/// Builds a project. Nothing is written unless the whole build succeeds.
pub fn build(options: &ProjectOptions) -> CompilerResult {
    project::run(ProjectCommand::Build, options)
}
