//! The NMath `.nts` content-mapping transform.
//!
//! `.nts` text is parsed with the standard tsrs TypeScript parser (no parser or
//! lexer extension) and lowered to TypeScript that calls the `__nmath` runtime
//! helpers. The result carries a span map from the lowered (virtual)
//! text back to the original text, in the shape the tsrs content-mapper
//! protocol expects.
//!
//! Behavior contract: `docs/crates-nmath-compiler-contract.md`.

mod diagnostics;
mod rewrite;
mod span_map;

pub use diagnostics::{Diagnostic, DiagnosticCode, line_column, offset_of_line_character};
pub use span_map::{Segment, SegmentKind, SpanMap};

/// The module specifier of the runtime helpers that lowered `.nts` output
/// imports.
pub const RUNTIME_MODULE: &str = "@delinoio/nmath/nts-runtime";
/// The local namespace name of the runtime helpers in lowered output.
pub const RUNTIME_NAMESPACE: &str = "__nmath";
/// The import line inserted at the top of every lowered module.
pub const RUNTIME_IMPORT: &str = "import * as __nmath from \"@delinoio/nmath/nts-runtime\";";
/// The source file extension handled by the transform.
pub const NTS_EXTENSION: &str = ".nts";

/// Options for [`transform_with_options`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransformOptions {
    /// Rewrite relative import and export specifiers ending in `.nts` to `.js`.
    ///
    /// Emitted output needs this. A content mapper that hands virtual text to
    /// the tsrs compiler turns it off so the compiler still resolves the
    /// `.nts` file and applies its own output-extension rewrite on emit.
    pub rewrite_import_extensions: bool,
}

impl Default for TransformOptions {
    fn default() -> TransformOptions {
        TransformOptions {
            rewrite_import_extensions: true,
        }
    }
}

/// The result of lowering one `.nts` file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransformOutput {
    /// The lowered TypeScript text. When the source has syntax errors this is
    /// the original text unchanged.
    pub text: String,
    /// Maps positions in `text` back to the original source.
    pub span_map: SpanMap,
    /// Transform diagnostics in original-source coordinates.
    pub diagnostics: Vec<Diagnostic>,
}

impl TransformOutput {
    pub fn has_errors(&self) -> bool {
        !self.diagnostics.is_empty()
    }
}

/// Lowers `.nts` source text with the default options.
pub fn transform(file_name: &str, source: &str) -> TransformOutput {
    transform_with_options(file_name, source, TransformOptions::default())
}

/// Lowers `.nts` source text.
pub fn transform_with_options(
    file_name: &str,
    source: &str,
    options: TransformOptions,
) -> TransformOutput {
    let output = rewrite::run(file_name, source, options);
    tracing::debug!(
        file = file_name,
        source_len = source.len(),
        output_len = output.text.len(),
        segments = output.span_map.segments().len(),
        diagnostics = output.diagnostics.len(),
        "lowered .nts file"
    );
    output
}

/// Whether `file_name` names a `.nts` file.
pub fn is_nts_file(file_name: &str) -> bool {
    file_name.ends_with(NTS_EXTENSION)
}

#[cfg(test)]
mod tests;
