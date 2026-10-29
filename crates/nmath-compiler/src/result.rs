//! The structured compiler result printed by `--json` and returned by the
//! project API.

use std::fmt;

use nmath_nts::DiagnosticCode;
use serde::{Serialize, Serializer};

/// Stable compiler diagnostic codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompilerCode {
    NtsParse,
    NtsAmbiguousType,
    NtsUnsupportedSyntax,
    NtsAssignRequiresAsync,
    Config,
    TypeError,
    Emit,
}

impl CompilerCode {
    pub fn as_str(self) -> &'static str {
        match self {
            CompilerCode::NtsParse => "NMATH_NTS_PARSE",
            CompilerCode::NtsAmbiguousType => DiagnosticCode::AmbiguousType.as_str(),
            CompilerCode::NtsUnsupportedSyntax => DiagnosticCode::UnsupportedSyntax.as_str(),
            CompilerCode::NtsAssignRequiresAsync => "NMATH_NTS_ASSIGN_REQUIRES_ASYNC",
            CompilerCode::Config => "NMATH_COMPILER_CONFIG",
            CompilerCode::TypeError => "NMATH_COMPILER_TYPE_ERROR",
            CompilerCode::Emit => "NMATH_COMPILER_EMIT",
        }
    }

    pub fn recovery(self) -> &'static str {
        match self {
            CompilerCode::NtsParse => DiagnosticCode::Parse.recovery(),
            CompilerCode::NtsAmbiguousType => DiagnosticCode::AmbiguousType.recovery(),
            CompilerCode::NtsUnsupportedSyntax => DiagnosticCode::UnsupportedSyntax.recovery(),
            CompilerCode::NtsAssignRequiresAsync => DiagnosticCode::AssignRequiresAsync.recovery(),
            CompilerCode::Config => {
                "Pass an existing tsconfig.json with -p and fix the reported configuration error."
            }
            CompilerCode::TypeError => {
                "Fix the reported type error; narrow values that may be NMath or ordinary JS \
                 before using an operator."
            }
            CompilerCode::Emit => {
                "Check that the outDir is writable and has free space, then run the build again."
            }
        }
    }
}

impl From<DiagnosticCode> for CompilerCode {
    fn from(code: DiagnosticCode) -> CompilerCode {
        match code {
            DiagnosticCode::Parse => CompilerCode::NtsParse,
            DiagnosticCode::AmbiguousType => CompilerCode::NtsAmbiguousType,
            DiagnosticCode::UnsupportedSyntax => CompilerCode::NtsUnsupportedSyntax,
            DiagnosticCode::AssignRequiresAsync => CompilerCode::NtsAssignRequiresAsync,
        }
    }
}

impl fmt::Display for CompilerCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for CompilerCode {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

/// One diagnostic. `file` is empty and `line`/`column` are 0 for diagnostics
/// without a location.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CompilerDiagnostic {
    pub code: CompilerCode,
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub message: String,
    pub recovery: String,
}

impl CompilerDiagnostic {
    pub fn new(
        code: CompilerCode,
        file: impl Into<String>,
        line: u32,
        column: u32,
        message: impl Into<String>,
    ) -> Self {
        CompilerDiagnostic {
            code,
            file: file.into(),
            line,
            column,
            message: message.into(),
            recovery: code.recovery().to_string(),
        }
    }

    pub fn without_location(code: CompilerCode, message: impl Into<String>) -> Self {
        CompilerDiagnostic::new(code, "", 0, 0, message)
    }
}

/// `{ ok, diagnostics, outputFiles }`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CompilerResult {
    pub ok: bool,
    pub diagnostics: Vec<CompilerDiagnostic>,
    pub output_files: Vec<String>,
}

/// Process exit codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitCode {
    Success = 0,
    Diagnostics = 1,
    UsageOrConfig = 2,
}

impl CompilerResult {
    pub fn exit_code(&self) -> ExitCode {
        if self.ok {
            ExitCode::Success
        } else if self
            .diagnostics
            .iter()
            .any(|d| d.code == CompilerCode::Config)
        {
            ExitCode::UsageOrConfig
        } else {
            ExitCode::Diagnostics
        }
    }

    /// The human-readable report.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for d in &self.diagnostics {
            if d.file.is_empty() {
                out.push_str(&format!("error {}: {}\n", d.code, d.message));
            } else {
                out.push_str(&format!(
                    "{}:{}:{} - error {}: {}\n",
                    d.file, d.line, d.column, d.code, d.message
                ));
            }
            out.push_str(&format!("  recovery: {}\n", d.recovery));
        }
        for file in &self.output_files {
            out.push_str(&format!("wrote {file}\n"));
        }
        if !self.diagnostics.is_empty() {
            out.push_str(&format!("Found {} error(s).\n", self.diagnostics.len()));
        }
        out
    }
}
