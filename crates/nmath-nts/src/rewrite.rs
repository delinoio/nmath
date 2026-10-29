//! The `.nts` lowering pass: walks the standard TypeScript AST and records text
//! edits, then applies them while building the span map.

use tsrs_ast::{FunctionFlags, Kind, Node, NodeFlags, SourceFileParseOptions};
use tsrs_core::{P, ScriptKind, tspath};

use crate::{
    RUNTIME_IMPORT, TransformOptions, TransformOutput,
    diagnostics::{Diagnostic, DiagnosticCode},
    span_map::{Segment, SegmentKind, SpanMap},
};

/// Where an edit sorts among edits that start at the same offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum EditClass {
    /// Text inserted at the end of a rewritten node (deeper nodes close first).
    Close,
    /// Text inserted at the start of a rewritten node (outer nodes open first).
    Open,
    /// Original text replaced by new text.
    Replace,
}

#[derive(Debug)]
struct Edit {
    start: u32,
    end: u32,
    text: String,
    class: EditClass,
    depth: i64,
    /// The original range that an atom segment maps the inserted text to.
    origin: Option<(u32, u32)>,
}

impl Edit {
    fn sort_key(&self) -> (u32, EditClass, i64) {
        let depth = match self.class {
            EditClass::Close => -self.depth,
            EditClass::Open | EditClass::Replace => self.depth,
        };
        (self.start, self.class, depth)
    }
}

/// The `__nmath` helper that a binary operator lowers to.
fn binary_helper(operator: Kind) -> Option<&'static str> {
    Some(match operator {
        Kind::PlusToken => "add",
        Kind::MinusToken => "sub",
        Kind::AsteriskToken => "mul",
        Kind::SlashToken => "div",
        Kind::PercentToken => "mod",
        Kind::AsteriskAsteriskToken => "pow",
        _ => return None,
    })
}

/// The `__nmath` helper that a prefix unary operator lowers to.
fn unary_helper(operator: Kind) -> Option<&'static str> {
    Some(match operator {
        Kind::PlusToken => "pos",
        Kind::MinusToken => "neg",
        _ => return None,
    })
}

/// Whether an expression is built only from numeric or string literals, so
/// arithmetic on it keeps ordinary JS semantics and is not rewritten.
fn is_literal_only(node: P<Node>) -> bool {
    match node.kind() {
        Kind::NumericLiteral
        | Kind::BigIntLiteral
        | Kind::StringLiteral
        | Kind::NoSubstitutionTemplateLiteral => true,
        Kind::ParenthesizedExpression => {
            is_literal_only(node.as_parenthesized_expression().expression.get())
        }
        Kind::PrefixUnaryExpression => {
            let unary = node.as_prefix_unary_expression();
            unary_helper(unary.operator).is_some() && is_literal_only(unary.operand)
        }
        Kind::BinaryExpression => {
            let binary = node.as_binary_expression();
            binary_helper(binary.operator_token.kind()).is_some()
                && is_literal_only(binary.left)
                && is_literal_only(binary.right.get())
        }
        _ => false,
    }
}

fn is_string_literal_like(node: P<Node>) -> bool {
    matches!(
        node.kind(),
        Kind::StringLiteral | Kind::NoSubstitutionTemplateLiteral
    )
}

struct Rewriter<'a> {
    text: &'a str,
    options: TransformOptions,
    edits: Vec<Edit>,
    diagnostics: Vec<Diagnostic>,
}

impl Rewriter<'_> {
    fn token_start(&self, node: P<Node>) -> u32 {
        tsrs_scanner::skip_trivia(self.text, pos(node))
    }

    fn open(&mut self, at: u32, text: String, depth: i64, origin: Option<(u32, u32)>) {
        self.edits.push(Edit {
            start: at,
            end: at,
            text,
            class: EditClass::Open,
            depth,
            origin,
        });
    }

    fn close(&mut self, at: u32, text: &str, depth: i64) {
        self.edits.push(Edit {
            start: at,
            end: at,
            text: text.to_string(),
            class: EditClass::Close,
            depth,
            origin: Some((at, at)),
        });
    }

    fn replace(
        &mut self,
        start: u32,
        end: u32,
        text: &str,
        depth: i64,
        origin: Option<(u32, u32)>,
    ) {
        self.edits.push(Edit {
            start,
            end,
            text: text.to_string(),
            class: EditClass::Replace,
            depth,
            origin,
        });
    }

    /// The argument separator that replaces an operator ending at `at`: the
    /// following trivia supplies the space.
    fn separator_after(&self, at: u32) -> &'static str {
        match self.text.as_bytes().get(at as usize) {
            Some(b' ' | b'\t' | b'\n' | b'\r') => ",",
            _ => ", ",
        }
    }

    /// The offset just past the `[` that follows `after` (skipping trivia).
    fn open_bracket_end(&self, after: u32) -> u32 {
        let bracket = tsrs_scanner::skip_trivia(self.text, after);
        debug_assert_eq!(self.text.as_bytes().get(bracket as usize), Some(&b'['));
        bracket + 1
    }

    /// Parenthesizes a comma-expression index (`a[f(), g()]`) so it stays one
    /// helper argument instead of splitting into two.
    fn keep_single_argument(&mut self, argument: P<Node>, depth: i64) {
        if argument.kind() != Kind::BinaryExpression
            || argument.as_binary_expression().operator_token.kind() != Kind::CommaToken
        {
            return;
        }
        let start = self.token_start(argument);
        self.open(start, "(".to_string(), depth, Some((start, end(argument))));
        self.close(end(argument), ")", depth);
    }

    fn visit_children(&mut self, node: P<Node>, async_context: bool, depth: i64) {
        for child in node.iter_children() {
            self.visit(child, async_context, depth + 1);
        }
    }

    fn visit(&mut self, node: P<Node>, async_context: bool, depth: i64) {
        match node.kind() {
            // Type-level and constant contexts never contain runtime NMath operations.
            Kind::InterfaceDeclaration | Kind::TypeAliasDeclaration | Kind::EnumDeclaration => {
                return;
            }
            _ if tsrs_ast::is_type_node(node) => return,
            Kind::FunctionDeclaration
            | Kind::FunctionExpression
            | Kind::ArrowFunction
            | Kind::MethodDeclaration
            | Kind::Constructor
            | Kind::GetAccessor
            | Kind::SetAccessor => {
                // Only the body is an await context, because `await` is a
                // syntax error in parameter initializers. Computed names and
                // decorators evaluate in the enclosing context.
                let is_async =
                    tsrs_ast::get_function_flags(Some(node)).contains(FunctionFlags::Async);
                let body = node.body();
                for child in node.iter_children() {
                    let child_context = if Some(child) == body {
                        is_async
                    } else if child.kind() == Kind::Parameter {
                        false
                    } else {
                        async_context
                    };
                    self.visit(child, child_context, depth + 1);
                }
                return;
            }
            Kind::ClassStaticBlockDeclaration
            | Kind::PropertyDeclaration
            | Kind::ModuleDeclaration => {
                return self.visit_children(node, false, depth);
            }
            Kind::BinaryExpression => self.visit_binary(node, async_context, depth),
            Kind::PrefixUnaryExpression => self.visit_prefix_unary(node, depth),
            Kind::ElementAccessExpression => self.visit_element_access(node, depth),
            Kind::ImportDeclaration => {
                self.rewrite_module_specifier(Some(node.as_import_declaration().module_specifier));
            }
            Kind::ExportDeclaration => {
                self.rewrite_module_specifier(node.as_export_declaration().module_specifier);
            }
            Kind::CallExpression => {
                let call = node.as_call_expression();
                if call.expression.kind() == Kind::ImportKeyword {
                    self.rewrite_module_specifier(call.arguments.nodes().first().copied());
                }
            }
            _ => {}
        }
        self.visit_children(node, async_context, depth);
    }

    fn visit_binary(&mut self, node: P<Node>, async_context: bool, depth: i64) {
        let binary = node.as_binary_expression();
        let operator = binary.operator_token;
        let right = binary.right.get();
        if let Some(helper) = binary_helper(operator.kind()) {
            if is_literal_only(binary.left) && is_literal_only(right) {
                return;
            }
            let start = self.token_start(node);
            let operator_range = (self.token_start(operator), end(operator));
            self.open(
                start,
                format!("__nmath.{helper}("),
                depth,
                Some((start, end(node))),
            );
            let separator = self.separator_after(end(operator));
            self.replace(
                end(binary.left),
                end(operator),
                separator,
                depth,
                Some(operator_range),
            );
            self.close(end(node), ")", depth);
            return;
        }
        if operator.kind() != Kind::EqualsToken || !is_rewritable_element_access(binary.left) {
            return;
        }
        let target = binary.left;
        let element = target.as_element_access_expression();
        let target_start = self.token_start(target);
        if !async_context {
            let message = "Bracket assignment on an NMath value lowers to `await \
                           __nmath.set(...)`, which requires an async function body or the module \
                           top level."
                .to_string();
            self.diagnostics.push(Diagnostic::new(
                self.text,
                DiagnosticCode::AssignRequiresAsync,
                target_start,
                end(node) - target_start,
                message,
            ));
            return;
        }
        let bracket_end = self.open_bracket_end(end(element.expression));
        let prefix = if self.needs_statement_separator(node, target_start) {
            ";"
        } else {
            ""
        };
        self.open(
            target_start,
            format!("{prefix}(await __nmath.set("),
            depth,
            Some((target_start, end(node))),
        );
        self.replace(end(element.expression), bracket_end, ", ", depth, None);
        self.keep_single_argument(element.argument_expression, depth);
        let separator = self.separator_after(end(operator));
        self.replace(
            end(element.argument_expression),
            end(operator),
            separator,
            depth,
            None,
        );
        self.close(end(node), "))", depth);
    }

    /// Whether a lowered assignment that starts with `(` needs a leading `;` to
    /// keep it a separate statement. In semicolonless code, `x = 1` followed by
    /// `(await __nmath.set(...))` on the next line would otherwise parse as the
    /// call `1(...)`. Only an expression statement that begins with the
    /// assignment, in a statement list, after a statement that does not end in
    /// `;`, is at risk; assignments nested in other expressions never are.
    fn needs_statement_separator(&self, node: P<Node>, start: u32) -> bool {
        let mut current = node;
        let statement = loop {
            let Some(parent) = current.parent() else {
                return false;
            };
            if parent.kind() == Kind::ExpressionStatement {
                break parent;
            }
            if self.token_start(parent) != start {
                return false;
            }
            current = parent;
        };
        let Some(list) = statement.parent().filter(|list| {
            matches!(
                list.kind(),
                Kind::SourceFile
                    | Kind::Block
                    | Kind::ModuleBlock
                    | Kind::CaseClause
                    | Kind::DefaultClause
            )
        }) else {
            return false;
        };
        let mut previous = None;
        for child in list.iter_children() {
            if child == statement {
                break;
            }
            previous = Some(child);
        }
        previous.is_some_and(|previous| {
            self.text.as_bytes().get(end(previous) as usize - 1) != Some(&b';')
        })
    }

    fn visit_prefix_unary(&mut self, node: P<Node>, depth: i64) {
        let unary = node.as_prefix_unary_expression();
        let Some(helper) = unary_helper(unary.operator) else {
            return;
        };
        if is_literal_only(unary.operand) {
            return;
        }
        let start = self.token_start(node);
        self.open(
            start,
            format!("__nmath.{helper}("),
            depth,
            Some((start, end(node))),
        );
        self.replace(start, start + 1, "", depth, None);
        self.close(end(node), ")", depth);
    }

    fn visit_element_access(&mut self, node: P<Node>, depth: i64) {
        if !is_rewritable_element_access(node) || is_reference_position(node) {
            return;
        }
        let element = node.as_element_access_expression();
        let start = self.token_start(node);
        let bracket_end = self.open_bracket_end(end(element.expression));
        self.open(
            start,
            "__nmath.get(".to_string(),
            depth,
            Some((start, end(node))),
        );
        self.replace(end(element.expression), bracket_end, ", ", depth, None);
        self.keep_single_argument(element.argument_expression, depth);
        self.replace(
            end(element.argument_expression),
            end(node),
            ")",
            depth,
            None,
        );
    }

    /// Rewrites a source-relative `.nts` module specifier to `.js`.
    fn rewrite_module_specifier(&mut self, specifier: Option<P<Node>>) {
        if !self.options.rewrite_import_extensions {
            return;
        }
        let Some(specifier) = specifier.filter(|s| s.kind() == Kind::StringLiteral) else {
            return;
        };
        let start = self.token_start(specifier) as usize;
        let end = end(specifier) as usize;
        let Some(literal) = self.text.get(start..end) else {
            return;
        };
        if literal.len() < 2 {
            return;
        }
        let value = &literal[1..literal.len() - 1];
        let relative = value.starts_with("./") || value.starts_with("../");
        if relative && value.ends_with(crate::NTS_EXTENSION) {
            let extension_end = (end - 1) as u32;
            let extension_start = extension_end - crate::NTS_EXTENSION.len() as u32;
            self.replace(
                extension_start,
                extension_end,
                ".js",
                0,
                Some((extension_start, extension_end)),
            );
        }
    }
}

fn pos(node: P<Node>) -> u32 {
    node.pos().expect_source(".nts nodes come from source text")
}

fn end(node: P<Node>) -> u32 {
    node.end().expect_source(".nts nodes come from source text")
}

/// Whether an element access may lower to `__nmath.get` or `__nmath.set`: not
/// an optional chain, not `super[...]`, and not a string-literal key
/// (`obj["key"]` stays ordinary JS property access).
fn is_rewritable_element_access(node: P<Node>) -> bool {
    if node.kind() != Kind::ElementAccessExpression
        || node.flags().contains(NodeFlags::OptionalChain)
    {
        return false;
    }
    let element = node.as_element_access_expression();
    element.expression.kind() != Kind::SuperKeyword
        && !is_string_literal_like(element.argument_expression)
}

/// Whether an element access is used as a reference rather than read as a
/// value: an assignment target (plain, compound, `++`/`--`, destructuring,
/// `for-in`/`for-of`), a `delete` operand, or a call or tagged-template callee
/// (rewriting a callee would drop its `this` binding). Parentheses and
/// type-only wrappers (`!`, `as`, `satisfies`, `<T>`) preserve the reference,
/// so `(a[i])()` and `delete (a[i])` are reference positions too.
fn is_reference_position(node: P<Node>) -> bool {
    if tsrs_ast::is_assignment_target(node) {
        return true;
    }
    let mut node = node;
    let Some(mut parent) = node.parent() else {
        return false;
    };
    while matches!(
        parent.kind(),
        Kind::ParenthesizedExpression
            | Kind::NonNullExpression
            | Kind::AsExpression
            | Kind::SatisfiesExpression
            | Kind::TypeAssertionExpression
    ) {
        node = parent;
        let Some(next) = parent.parent() else {
            return false;
        };
        parent = next;
    }
    match parent.kind() {
        Kind::DeleteExpression => true,
        Kind::CallExpression => parent.as_call_expression().expression == node,
        Kind::TaggedTemplateExpression => parent.as_tagged_template_expression().tag == node,
        _ => false,
    }
}

pub(crate) fn run(file_name: &str, source: &str, options: TransformOptions) -> TransformOutput {
    // The parser allocates the tree in an arena; a scratch region frees it when
    // the transform returns so that long-running callers (the language
    // server and the content mapper) do not accumulate trees.
    let region = tsrs_core::arena::Region::new((source.len() * 16).clamp(64 << 10, 64 << 20));
    let output = {
        let _scope = region.enter();
        run_in_region(file_name, source, options)
    };
    drop(region);
    output
}

fn run_in_region(file_name: &str, source: &str, options: TransformOptions) -> TransformOutput {
    // The parser requires a normalized absolute name and only uses it for
    // declaration-file detection, so a fixed `.ts` name stands in for the
    // caller's file name (which may be relative or end in `.nts`).
    let _ = file_name;
    let parse_name = "/__nmath__/input.ts".to_string();
    let parse_options = SourceFileParseOptions {
        path: tspath::to_path(&parse_name, "", true),
        file_name: parse_name,
        external_module_indicator_options: Default::default(),
    };
    let source_file = tsrs_parser::parse_source_file(parse_options, source, ScriptKind::TS);

    let parse_diagnostics = source_file.diagnostics();
    if !parse_diagnostics.is_empty() {
        // A file with syntax errors is passed through unchanged, so downstream
        // syntax errors land at the same original positions as these
        // diagnostics.
        let diagnostics = parse_diagnostics
            .iter()
            .map(|d| {
                let start = d.pos().unwrap_or(0);
                let message = format!("{} (TS{})", d.localize(), d.code());
                Diagnostic::new(
                    source,
                    DiagnosticCode::Parse,
                    start,
                    d.len().unwrap_or(0),
                    message,
                )
            })
            .collect();
        return TransformOutput {
            text: source.to_string(),
            span_map: SpanMap::identity(source.len() as u32),
            diagnostics,
        };
    }

    let mut rewriter = Rewriter {
        text: source,
        options,
        edits: Vec::new(),
        diagnostics: Vec::new(),
    };
    for statement in source_file.statements().nodes() {
        rewriter.visit(*statement, true, 1);
    }

    let import_at = if source.starts_with("#!") {
        source
            .find('\n')
            .map_or(source.len(), |newline| newline + 1) as u32
    } else {
        0
    };
    let import_text = if import_at as usize == source.len() && import_at != 0 {
        format!("\n{RUNTIME_IMPORT}\n")
    } else {
        format!("{RUNTIME_IMPORT}\n")
    };
    rewriter.open(import_at, import_text, 0, None);

    let Rewriter {
        mut edits,
        diagnostics,
        ..
    } = rewriter;
    edits.sort_by_key(Edit::sort_key);
    let (text, span_map) = apply(source, &edits);
    TransformOutput {
        text,
        span_map,
        diagnostics,
    }
}

fn apply(source: &str, edits: &[Edit]) -> (String, SpanMap) {
    let mut text =
        String::with_capacity(source.len() + edits.iter().map(|e| e.text.len()).sum::<usize>());
    let mut map = SpanMap::default();
    let mut cursor = 0u32;
    let copy = |text: &mut String, map: &mut SpanMap, from: u32, to: u32| {
        let virtual_start = text.len() as u32;
        text.push_str(&source[from as usize..to as usize]);
        map.push(Segment {
            virtual_start,
            virtual_end: text.len() as u32,
            original_start: from,
            original_end: to,
            kind: SegmentKind::Verbatim,
        });
    };
    for edit in edits {
        if edit.start < cursor {
            // Overlapping edits would corrupt the output; the walker never
            // produces them.
            debug_assert!(false, "overlapping .nts edits: {edit:?} before {cursor}");
            tracing::warn!(start = edit.start, cursor, "skipping overlapping .nts edit");
            continue;
        }
        if edit.start > cursor {
            copy(&mut text, &mut map, cursor, edit.start);
        }
        let virtual_start = text.len() as u32;
        text.push_str(&edit.text);
        if let Some((original_start, original_end)) = edit.origin
            && !edit.text.is_empty()
        {
            map.push(Segment {
                virtual_start,
                virtual_end: text.len() as u32,
                original_start,
                original_end,
                kind: SegmentKind::Atom,
            });
        }
        cursor = edit.end;
    }
    if (cursor as usize) < source.len() {
        copy(&mut text, &mut map, cursor, source.len() as u32);
    }
    (text, map)
}
