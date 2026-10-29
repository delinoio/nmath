//! Command-line parsing and dispatch.

use std::{io::Write, path::PathBuf};

use crate::{
    project::{MapperLaunch, ProjectCommand, ProjectOptions},
    result::ExitCode,
};

pub const USAGE: &str = "\
Usage:
  nmath-compiler check -p <tsconfig> [--json]
  nmath-compiler build -p <tsconfig> [--json]
  nmath-compiler lsp
";

/// A parsed command line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Invocation {
    Project {
        command: ProjectCommand,
        tsconfig: PathBuf,
        json: bool,
    },
    Lsp,
    ContentMapper,
    Help,
}

pub fn parse(args: &[String]) -> Result<Invocation, String> {
    let Some((subcommand, rest)) = args.split_first() else {
        return Err("missing command".to_string());
    };
    let command = match subcommand.as_str() {
        "check" => ProjectCommand::Check,
        "build" => ProjectCommand::Build,
        "lsp" | "content-mapper" => {
            // Editors pass `--stdio` to language servers; stdio is the only
            // transport.
            if let Some(extra) = rest.iter().find(|arg| arg.as_str() != "--stdio") {
                return Err(format!("unexpected argument '{extra}'"));
            }
            return Ok(if subcommand == "lsp" {
                Invocation::Lsp
            } else {
                Invocation::ContentMapper
            });
        }
        "-h" | "--help" | "help" => return Ok(Invocation::Help),
        other => return Err(format!("unknown command '{other}'")),
    };
    let mut tsconfig = None;
    let mut json = false;
    let mut iter = rest.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-p" | "--project" => {
                let value = iter
                    .next()
                    .ok_or_else(|| format!("'{arg}' requires a tsconfig path"))?;
                tsconfig = Some(PathBuf::from(value));
            }
            "--json" => json = true,
            other => return Err(format!("unexpected argument '{other}'")),
        }
    }
    let tsconfig = tsconfig.ok_or("missing -p <tsconfig>")?;
    Ok(Invocation::Project {
        command,
        tsconfig,
        json,
    })
}

/// Runs the CLI and returns the process exit code.
pub fn main(args: &[String]) -> i32 {
    let invocation = match parse(args) {
        Ok(invocation) => invocation,
        Err(message) => {
            eprintln!("error: {message}\n\n{USAGE}");
            return ExitCode::UsageOrConfig as i32;
        }
    };
    tracing::debug!(?invocation, "nmath-compiler started");
    match invocation {
        Invocation::Help => {
            print!("{USAGE}");
            ExitCode::Success as i32
        }
        Invocation::ContentMapper => {
            match crate::content_mapper::serve(std::io::stdin().lock(), std::io::stdout().lock()) {
                Ok(()) => 0,
                Err(err) => {
                    tracing::error!(error = %err, "content mapper failed");
                    1
                }
            }
        }
        Invocation::Lsp => crate::lsp::serve(std::io::stdin().lock(), std::io::stdout().lock())
            .unwrap_or_else(|err| {
                tracing::error!(error = %err, "language server failed");
                1
            }),
        Invocation::Project {
            command,
            tsconfig,
            json,
        } => {
            let options = ProjectOptions {
                tsconfig,
                mapper: MapperLaunch::CurrentExecutable,
            };
            let result = crate::project::run(command, &options);
            let mut stdout = std::io::stdout().lock();
            let text = if json {
                serde_json::to_string_pretty(&result).expect("results serialize") + "\n"
            } else {
                result.render()
            };
            let _ = stdout.write_all(text.as_bytes());
            let _ = stdout.flush();
            result.exit_code() as i32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parses_project_commands() {
        assert_eq!(
            parse(&args(&["check", "-p", "tsconfig.json", "--json"])),
            Ok(Invocation::Project {
                command: ProjectCommand::Check,
                tsconfig: "tsconfig.json".into(),
                json: true
            })
        );
        assert_eq!(
            parse(&args(&["build", "--project", "a/tsconfig.json"])),
            Ok(Invocation::Project {
                command: ProjectCommand::Build,
                tsconfig: "a/tsconfig.json".into(),
                json: false
            })
        );
        assert_eq!(parse(&args(&["lsp", "--stdio"])), Ok(Invocation::Lsp));
        assert_eq!(
            parse(&args(&["content-mapper"])),
            Ok(Invocation::ContentMapper)
        );
    }

    #[test]
    fn rejects_usage_errors() {
        assert!(parse(&args(&[])).is_err());
        assert!(parse(&args(&["check"])).is_err());
        assert!(parse(&args(&["check", "-p"])).is_err());
        assert!(parse(&args(&["frobnicate"])).is_err());
        assert!(parse(&args(&["build", "-p", "x", "--watch"])).is_err());
    }
}
