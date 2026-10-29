//! Checker-directed decisions. No decision depends on an identifier's spelling
//! at its use site: aliases, imports and flow narrowing are resolved by tsrs.

use std::collections::BTreeSet;

use tsrs_ast::{Kind, Node, NodeFlags, SourceFile};
use tsrs_checker::{Checker, Type, TypeFlags};
use tsrs_core::P;

use crate::{Diagnostic, DiagnosticCode};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoweringPlan {
    pub operations: BTreeSet<(u32, u32)>,
    pub diagnostics: Vec<Diagnostic>,
}

impl LoweringPlan {
    pub fn contains(&self, start: u32, end: u32) -> bool {
        self.operations.contains(&(start, end))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueKind {
    Ordinary,
    NMath,
    Ambiguous,
}

fn classify(checker: &mut Checker, ty: P<Type>, depth: usize) -> ValueKind {
    if depth > 64 {
        return ValueKind::Ambiguous;
    }
    let flags = ty.flags();
    if flags.intersects(TypeFlags::Any | TypeFlags::Unknown) {
        return ValueKind::Ambiguous;
    }
    if flags.intersects(TypeFlags::Never) {
        return ValueKind::Ordinary;
    }
    if flags.intersects(TypeFlags::Union) {
        let mut kinds = ty
            .types()
            .iter()
            .filter(|t| !t.flags().intersects(TypeFlags::Never))
            .map(|t| classify(checker, *t, depth + 1));
        let first = kinds.next().unwrap_or(ValueKind::Ordinary);
        return if kinds.all(|k| k == first) {
            first
        } else {
            ValueKind::Ambiguous
        };
    }
    if flags.intersects(TypeFlags::Intersection) {
        let kinds: Vec<_> = ty
            .types()
            .iter()
            .map(|t| classify(checker, *t, depth + 1))
            .collect();
        return if kinds.contains(&ValueKind::NMath) {
            ValueKind::NMath
        } else if kinds.contains(&ValueKind::Ambiguous) {
            ValueKind::Ambiguous
        } else {
            ValueKind::Ordinary
        };
    }
    if flags.intersects(TypeFlags::InstantiableNonPrimitive) {
        return checker
            .get_base_constraint_of_type_exported(ty)
            .filter(|constraint| *constraint != ty)
            .map_or(ValueKind::Ambiguous, |constraint| {
                classify(checker, constraint, depth + 1)
            });
    }
    if let Some(symbol) = ty.symbol()
        && matches!(symbol.name(), "NMathArray" | "Expr" | "Lazy" | "NMathValue")
        && symbol.declarations().iter().any(|declaration| {
            let Some(file) = tsrs_ast::get_source_file_of_node(*declaration) else {
                return false;
            };
            // Resolve ownership from the declaring package, never from a local
            // alias or a class name alone. The package may be
            // installed or symlinked.
            let path = std::path::Path::new(file.file_name());
            if !matches!(
                path.file_name().and_then(|n| n.to_str()),
                Some("core.ts" | "core.d.ts")
            ) {
                return false;
            }
            for ancestor in path.ancestors().skip(1) {
                let manifest = ancestor.join("package.json");
                if let Ok(text) = std::fs::read_to_string(manifest) {
                    return serde_json::from_str::<serde_json::Value>(&text)
                        .ok()
                        .and_then(|v| v.get("name").and_then(|n| n.as_str()).map(str::to_owned))
                        .as_deref()
                        == Some("@delinoio/nmath");
                }
            }
            false
        })
    {
        return ValueKind::NMath;
    }
    // The checker resolves interface and class bases to their declarations.
    // Follow those identities so a user type inherits NMath semantics only
    // when its ancestry reaches a declaration owned by the NMath package.
    let mut inherited = ValueKind::Ordinary;
    for &base in checker.get_base_types_exported(ty) {
        match classify(checker, base, depth + 1) {
            ValueKind::NMath => return ValueKind::NMath,
            ValueKind::Ambiguous => inherited = ValueKind::Ambiguous,
            ValueKind::Ordinary => {}
        }
    }
    if inherited == ValueKind::Ambiguous {
        return inherited;
    }
    ValueKind::Ordinary
}

/// Adds newly proven operations to an existing plan. Rebuild the project with
/// this plan until it stops changing: lowered expressions then supply their
/// actual lazy result types to downstream inference (also across files).
pub fn classify_file(
    checker: &mut Checker,
    file: P<SourceFile>,
    previous: &LoweringPlan,
) -> LoweringPlan {
    let mut plan = previous.clone();
    plan.diagnostics.clear();
    visit(checker, file, file.as_node(), &mut plan);
    plan
}

fn visit(checker: &mut Checker, file: P<SourceFile>, node: P<Node>, plan: &mut LoweringPlan) {
    if tsrs_ast::is_type_node(node)
        || matches!(
            node.kind(),
            Kind::InterfaceDeclaration | Kind::TypeAliasDeclaration
        )
    {
        return;
    }
    let (operands, supported): (Vec<P<Node>>, bool) = match node.kind() {
        Kind::BinaryExpression => {
            let b = node.as_binary_expression();
            let operator = b.operator_token.kind();
            if operator == Kind::EqualsToken {
                if b.left.kind() == Kind::ElementAccessExpression
                    && !matches!(
                        b.left
                            .as_element_access_expression()
                            .argument_expression
                            .kind(),
                        Kind::StringLiteral | Kind::NoSubstitutionTemplateLiteral
                    )
                {
                    (vec![b.left.as_element_access_expression().expression], true)
                } else {
                    (vec![], false)
                }
            } else if matches!(
                operator,
                Kind::CommaToken
                    | Kind::AmpersandAmpersandToken
                    | Kind::BarBarToken
                    | Kind::QuestionQuestionToken
                    | Kind::InstanceOfKeyword
                    | Kind::InKeyword
                    | Kind::EqualsEqualsEqualsToken
                    | Kind::ExclamationEqualsEqualsToken
                    | Kind::EqualsEqualsToken
                    | Kind::ExclamationEqualsToken
            ) {
                (vec![], false)
            } else {
                (
                    vec![b.left, b.right.get()],
                    matches!(
                        operator,
                        Kind::PlusToken
                            | Kind::MinusToken
                            | Kind::AsteriskToken
                            | Kind::SlashToken
                            | Kind::PercentToken
                            | Kind::AsteriskAsteriskToken
                    ),
                )
            }
        }
        Kind::PrefixUnaryExpression => {
            let u = node.as_prefix_unary_expression();
            (
                vec![u.operand],
                matches!(u.operator, Kind::PlusToken | Kind::MinusToken),
            )
        }
        Kind::PostfixUnaryExpression => (vec![node.as_postfix_unary_expression().operand], false),
        Kind::ElementAccessExpression => {
            let e = node.as_element_access_expression();
            let reference = tsrs_ast::is_assignment_target(node);
            if matches!(
                e.argument_expression.kind(),
                Kind::StringLiteral | Kind::NoSubstitutionTemplateLiteral
            ) {
                // Named properties keep JS property semantics, including
                // members of NMath instances. Selector syntax
                // uses numbers or wrappers.
                for child in node.iter_children() {
                    visit(checker, file, child, plan);
                }
                return;
            }
            // A simple write is classified at its BinaryExpression instead.
            if node.parent().is_some_and(|p| {
                p.kind() == Kind::BinaryExpression
                    && p.as_binary_expression().left == node
                    && p.as_binary_expression().operator_token.kind() == Kind::EqualsToken
            }) {
                (vec![], false)
            } else {
                (
                    vec![e.expression],
                    !reference
                        && !node.flags().contains(NodeFlags::OptionalChain)
                        && !unsupported_reference(node),
                )
            }
        }
        _ => (vec![], false),
    };
    if !operands.is_empty() {
        let kinds: Vec<_> = operands
            .iter()
            .map(|operand| {
                let ty = checker.get_type_at_location(*operand);
                classify(checker, ty, 0)
            })
            .collect();
        let start = tsrs_scanner::skip_trivia(file.text(), node.pos().expect_source("source node"));
        let end = node.end().expect_source("source node");
        let range = tsrs_core::TextRange::new(start, end);
        let mapped = file
            .span_map()
            .map_or(range, |map| map.virtual_to_original_span(range).0);
        let start = mapped.pos().source_offset().unwrap_or(0);
        let end = mapped.end().source_offset().unwrap_or(start);
        if kinds.contains(&ValueKind::Ambiguous) {
            plan.diagnostics.push(Diagnostic::new(
                file.original_text(),
                DiagnosticCode::AmbiguousType,
                start,
                end - start,
                "The operand type does not establish ordinary JavaScript or NMath semantics. \
                 Narrow the value or use an explicit API."
                    .into(),
            ));
        } else if kinds.contains(&ValueKind::NMath) {
            if supported {
                plan.operations.insert((start, end));
            } else {
                plan.diagnostics.push(Diagnostic::new(
                    file.original_text(),
                    DiagnosticCode::UnsupportedSyntax,
                    start,
                    end - start,
                    "This syntax is not supported for NMath values. Use an explicit NMath API."
                        .into(),
                ));
            }
        }
    }
    for child in node.iter_children() {
        visit(checker, file, child, plan);
    }
}

fn unsupported_reference(mut node: P<Node>) -> bool {
    while let Some(parent) = node.parent() {
        match parent.kind() {
            Kind::ParenthesizedExpression
            | Kind::AsExpression
            | Kind::NonNullExpression
            | Kind::TypeAssertionExpression
            | Kind::SatisfiesExpression => node = parent,
            Kind::CallExpression => return parent.as_call_expression().expression == node,
            Kind::NewExpression => return parent.as_new_expression().expression == node,
            Kind::TaggedTemplateExpression => {
                return parent.as_tagged_template_expression().tag == node;
            }
            Kind::DeleteExpression => return true,
            _ => return false,
        }
    }
    false
}
