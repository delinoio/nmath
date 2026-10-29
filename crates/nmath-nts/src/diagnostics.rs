//! Stable `.nts` transform diagnostics.

use std::fmt;

/// Stable public diagnostic codes reported by the `.nts` transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DiagnosticCode {
    /// The `.nts` file has a syntax error.
    Parse,
    AmbiguousType,
    UnsupportedSyntax,
    /// A bracket assignment appears outside an async function body or module
    /// top level.
    AssignRequiresAsync,
}

impl DiagnosticCode {
    pub const ALL: [DiagnosticCode; 4] = [
        DiagnosticCode::Parse,
        DiagnosticCode::AssignRequiresAsync,
        DiagnosticCode::AmbiguousType,
        DiagnosticCode::UnsupportedSyntax,
    ];

    /// The stable public code string.
    pub fn as_str(self) -> &'static str {
        match self {
            DiagnosticCode::Parse => "NMATH_NTS_PARSE",
            DiagnosticCode::AmbiguousType => "NMATH_NTS_AMBIGUOUS_TYPE",
            DiagnosticCode::UnsupportedSyntax => "NMATH_NTS_UNSUPPORTED_SYNTAX",
            DiagnosticCode::AssignRequiresAsync => "NMATH_NTS_ASSIGN_REQUIRES_ASYNC",
        }
    }

    /// The numeric code used for mapper-authored diagnostics in the tsrs
    /// content-mapper protocol.
    pub fn mapper_code(self) -> i32 {
        match self {
            DiagnosticCode::Parse => 1,
            DiagnosticCode::AmbiguousType => 3,
            DiagnosticCode::UnsupportedSyntax => 4,
            DiagnosticCode::AssignRequiresAsync => 2,
        }
    }

    pub fn from_mapper_code(code: i32) -> Option<DiagnosticCode> {
        DiagnosticCode::ALL
            .into_iter()
            .find(|candidate| candidate.mapper_code() == code)
    }

    /// English recovery guidance for this code.
    pub fn recovery(self) -> &'static str {
        match self {
            DiagnosticCode::AmbiguousType => {
                "Narrow the operand type or use an explicit NMath API."
            }
            DiagnosticCode::UnsupportedSyntax => "Use an explicit NMath API for this operation.",
            DiagnosticCode::Parse => {
                "Fix the TypeScript syntax at this location; .nts files use standard TypeScript \
                 syntax."
            }
            DiagnosticCode::AssignRequiresAsync => {
                "Move the assignment into an async function or the module top level, or call \
                 `await set(array, index, value)` explicitly."
            }
        }
    }
}

impl fmt::Display for DiagnosticCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A diagnostic located in the original `.nts` text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: DiagnosticCode,
    /// UTF-8 byte offset in the original text.
    pub start: u32,
    /// UTF-8 byte length in the original text.
    pub length: u32,
    /// 1-based line.
    pub line: u32,
    /// 1-based column in UTF-16 code units.
    pub column: u32,
    pub message: String,
    pub recovery: String,
}

impl Diagnostic {
    pub(crate) fn new(
        text: &str,
        code: DiagnosticCode,
        start: u32,
        length: u32,
        message: String,
    ) -> Diagnostic {
        let (line, column) = line_column(text, start);
        Diagnostic {
            code,
            start,
            length,
            line,
            column,
            message,
            recovery: code.recovery().to_string(),
        }
    }
}

/// The 1-based line and 1-based UTF-16 column of a UTF-8 byte offset. Offsets
/// past the end clamp to the end.
pub fn line_column(text: &str, offset: u32) -> (u32, u32) {
    let offset = (offset as usize).min(text.len());
    let mut line = 1;
    let mut line_start = 0;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < offset {
        match bytes[i] {
            b'\n' => {
                line += 1;
                line_start = i + 1;
            }
            b'\r' if bytes.get(i + 1) != Some(&b'\n') => {
                line += 1;
                line_start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    let prefix = text.get(line_start..offset).unwrap_or_else(|| {
        // The offset is inside a multi-byte character; count up to its start.
        let mut end = offset;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        &text[line_start..end]
    });
    let column = prefix.encode_utf16().count() as u32 + 1;
    (line, column)
}

/// The UTF-8 byte offset of a 0-based line and 0-based UTF-16 character (LSP
/// positions). Clamps to the line end.
pub fn offset_of_line_character(text: &str, line: u32, character: u32) -> u32 {
    let mut current = 0;
    let mut offset = 0;
    let bytes = text.as_bytes();
    while current < line && offset < bytes.len() {
        match bytes[offset] {
            b'\n' => current += 1,
            b'\r' if bytes.get(offset + 1) != Some(&b'\n') => current += 1,
            _ => {}
        }
        offset += 1;
    }
    let mut units = 0;
    for (index, ch) in text[offset..].char_indices() {
        if units >= character || ch == '\n' || ch == '\r' {
            return (offset + index) as u32;
        }
        units += ch.len_utf16() as u32;
    }
    text.len() as u32
}
