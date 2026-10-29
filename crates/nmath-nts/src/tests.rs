use super::*;

const PREFIX: &str = "import * as __nmath from \"@delinoio/nmath/nts-runtime\";\n";

fn lower(source: &str) -> String {
    let output = transform("test.nts", source);
    assert!(
        output.diagnostics.is_empty(),
        "unexpected diagnostics: {:?}",
        output.diagnostics
    );
    assert_valid_span_map(source, &output);
    output
        .text
        .strip_prefix(PREFIX)
        .expect("runtime import first")
        .to_string()
}

/// Checks the span map against the tsrs span-map contract and checks that every
/// verbatim segment copies text.
fn assert_valid_span_map(source: &str, output: &TransformOutput) {
    let tuples: Vec<String> = output
        .span_map
        .to_tuples()
        .iter()
        .map(|t| format!("[{},{},{},{},{}]", t[0], t[1], t[2], t[3], t[4]))
        .collect();
    let json = format!("[{}]", tuples.join(","));
    let map = tsrs_spanmap::unmarshal(json.as_bytes()).expect("span map decodes");
    assert!(
        map.validate(&output.text, source).is_none(),
        "invalid span map {json} for {:?}",
        output.text
    );
}

#[test]
fn inserts_runtime_import_at_top() {
    let output = transform("a.nts", "const x = 1;\n");
    assert_eq!(output.text, format!("{PREFIX}const x = 1;\n"));
}

#[test]
fn keeps_shebang_first() {
    let output = transform("a.nts", "#!/usr/bin/env node\nconst y = a + b;\n");
    assert_eq!(
        output.text,
        format!("#!/usr/bin/env node\n{PREFIX}const y = __nmath.add(a, b);\n")
    );
}

#[test]
fn rewrites_every_binary_operator() {
    for (op, helper) in [
        ("+", "add"),
        ("-", "sub"),
        ("*", "mul"),
        ("/", "div"),
        ("%", "mod"),
        ("**", "pow"),
    ] {
        assert_eq!(
            lower(&format!("x = a {op} b;")),
            format!("x = __nmath.{helper}(a, b);"),
            "operator {op}"
        );
    }
}

#[test]
fn rewrites_unary_plus_and_minus() {
    assert_eq!(lower("x = -a;"), "x = __nmath.neg(a);");
    assert_eq!(lower("x = +a;"), "x = __nmath.pos(a);");
    assert_eq!(lower("x = -(a);"), "x = __nmath.neg((a));");
}

#[test]
fn rewrites_element_reads() {
    assert_eq!(lower("x = a[i];"), "x = __nmath.get(a, i);");
    assert_eq!(lower("x = a[0];"), "x = __nmath.get(a, 0);");
    assert_eq!(lower("x = a[[i, j]];"), "x = __nmath.get(a, [i, j]);");
    assert_eq!(
        lower("x = a[slice(0, 2)];"),
        "x = __nmath.get(a, slice(0, 2));"
    );
    assert_eq!(
        lower("x = a[[slice(1), indices([0, 2])]];"),
        "x = __nmath.get(a, [slice(1), indices([0, 2])]);"
    );
    assert_eq!(lower("x = a[mask(m)];"), "x = __nmath.get(a, mask(m));");
    assert_eq!(lower("x = (a[i]);"), "x = (__nmath.get(a, i));");
    assert_eq!(lower("x = a[f(), g()];"), "x = __nmath.get(a, (f(), g()));");
    assert_eq!(
        lower("x = a[ i + 1, j ];"),
        "x = __nmath.get(a,  (__nmath.add(i, 1), j));"
    );
    assert_eq!(lower("x = f((a[i]));"), "x = f((__nmath.get(a, i)));");
    assert_eq!(
        lower("x = a[i][j];"),
        "x = __nmath.get(__nmath.get(a, i), j);"
    );
}

#[test]
fn rewrites_element_assignment_at_top_level_and_in_async_functions() {
    assert_eq!(lower("a[i] = v;"), "(await __nmath.set(a, i, v));");
    assert_eq!(
        lower("a[[i, j]] = v;"),
        "(await __nmath.set(a, [i, j], v));"
    );
    assert_eq!(
        lower("async function f() { a[i] = v + 1; }"),
        "async function f() { (await __nmath.set(a, i, __nmath.add(v, 1))); }"
    );
    assert_eq!(
        lower("const f = async () => { a[i] = v; };"),
        "const f = async () => { (await __nmath.set(a, i, v)); };"
    );
    assert_eq!(
        lower("class C { async m() { a[i] = v; } }"),
        "class C { async m() { (await __nmath.set(a, i, v)); } }"
    );
    assert_eq!(lower("a[i]=v;"), "(await __nmath.set(a, i, v));");
    assert_eq!(
        lower("const f = async (x) => a[i] = x;"),
        "const f = async (x) => (await __nmath.set(a, i, x));"
    );
    assert_eq!(
        lower("class C { [a[i] = k]() {} }"),
        "class C { [(await __nmath.set(a, i, k))]() {} }"
    );
    assert_eq!(
        lower("a[f(), g()] = v;"),
        "(await __nmath.set(a, (f(), g()), v));"
    );
    assert_eq!(
        lower("a[b[k]] = c[i];"),
        "(await __nmath.set(a, __nmath.get(b, k), __nmath.get(c, i)));"
    );
}

#[test]
fn reports_element_assignment_outside_async_context() {
    let source = "function f() {\n  a[i] = v;\n}\n";
    let output = transform("a.nts", source);
    assert_eq!(output.diagnostics.len(), 1);
    let diagnostic = &output.diagnostics[0];
    assert_eq!(diagnostic.code, DiagnosticCode::AssignRequiresAsync);
    assert_eq!(diagnostic.code.as_str(), "NMATH_NTS_ASSIGN_REQUIRES_ASYNC");
    assert_eq!((diagnostic.line, diagnostic.column), (2, 3));
    assert_eq!(
        &source[diagnostic.start as usize..(diagnostic.start + diagnostic.length) as usize],
        "a[i] = v"
    );
    assert!(!diagnostic.message.is_empty());
    assert!(!diagnostic.recovery.is_empty());
    // The assignment is left unchanged.
    assert!(output.text.contains("a[i] = v;"));

    for source in [
        "const f = () => { a[i] = v; };",
        "class C { m() { a[i] = v; } }",
        "class C { static { a[i] = v; } }",
        "async function f() { function g() { a[i] = v; } }",
        "async function f() { [1].forEach((x) => { a[x] = v; }); }",
        "async function f(x = (a[i] = v)) {}",
        "const f = async (x = (a[i] = v)) => x;",
        "class C { async m(x = (a[i] = v)) {} }",
    ] {
        let output = transform("a.nts", source);
        assert_eq!(output.diagnostics.len(), 1, "{source}");
        assert_eq!(
            output.diagnostics[0].code,
            DiagnosticCode::AssignRequiresAsync
        );
    }
}

#[test]
fn preserves_literal_only_arithmetic() {
    assert_eq!(lower("x = 1 + 2;"), "x = 1 + 2;");
    assert_eq!(lower("x = 'a' + \"b\";"), "x = 'a' + \"b\";");
    assert_eq!(lower("x = 2 * 3 + 1;"), "x = 2 * 3 + 1;");
    assert_eq!(lower("x = -1;"), "x = -1;");
    assert_eq!(lower("x = (1 + 2) ** -3;"), "x = (1 + 2) ** -3;");
    assert_eq!(lower("x = a + 1;"), "x = __nmath.add(a, 1);");
}

#[test]
fn preserves_ordinary_javascript() {
    for source in [
        "a += b;",
        "a[i] += b;",
        "a[i] -= 1;",
        "a[i]++;",
        "--a[i];",
        "x++;",
        "x = a < b;",
        "x = a === b;",
        "x = a & b;",
        "x = a | b;",
        "x = a ^ b;",
        "x = a << b;",
        "x = ~a;",
        "x = !a;",
        "x = a && b;",
        "x = a ?? b;",
        "x = a.b;",
        "x = obj[\"key\"];",
        "obj[\"key\"] = 1;",
        "x = obj[`key`];",
        "x = a?.[i];",
        "x = a?.b[i];",
        "a[i](x);",
        "delete a[i];",
        "(a[i])(x);",
        "(a[i]!)(x);",
        "((a[i] as F))(x);",
        "(a[i])`t`;",
        "delete (a[i]);",
        "delete ((a[i]));",
        "[a[0], a[1]] = pair;",
        "for (a[i] of xs) {}",
        "class C extends B { m() { return super[k]; } }",
        "type T = -1;",
        "let t: [number, string][0];",
        "enum E { A = 1, B = A + 1 }",
    ] {
        assert_eq!(lower(source), source, "{source}");
    }
}

#[test]
fn rewrites_nested_expressions() {
    assert_eq!(
        lower("x = a * b + c[i] ** -d;"),
        "x = __nmath.add(__nmath.mul(a, b), __nmath.pow(__nmath.get(c, i), __nmath.neg(d)));"
    );
    assert_eq!(lower("x = -a[i];"), "x = __nmath.neg(__nmath.get(a, i));");
    assert_eq!(lower("x = -a + b;"), "x = __nmath.add(__nmath.neg(a), b);");
    assert_eq!(
        lower("x = a[i + 1];"),
        "x = __nmath.get(a, __nmath.add(i, 1));"
    );
    assert_eq!(
        lower("x = f(a - b, c / d);"),
        "x = f(__nmath.sub(a, b), __nmath.div(c, d));"
    );
    assert_eq!(
        lower("x = (a + b) * c;"),
        "x = __nmath.mul((__nmath.add(a, b)), c);"
    );
    assert_eq!(lower("x = a+b;"), "x = __nmath.add(a, b);");
    assert_eq!(lower("x = a /* c */ + b;"), "x = __nmath.add(a, b);");
}

#[test]
fn rewrites_relative_nts_specifiers() {
    assert_eq!(
        lower("import { a } from './a.nts';"),
        "import { a } from './a.js';"
    );
    assert_eq!(
        lower("import b from \"../lib/b.nts\";"),
        "import b from \"../lib/b.js\";"
    );
    assert_eq!(lower("export * from './c.nts';"), "export * from './c.js';");
    assert_eq!(
        lower("export { d } from './d.nts';"),
        "export { d } from './d.js';"
    );
    assert_eq!(
        lower("const m = await import('./m.nts');"),
        "const m = await import('./m.js');"
    );
    // Bare and non-.nts specifiers are untouched.
    assert_eq!(
        lower("import x from 'pkg/x.nts';"),
        "import x from 'pkg/x.nts';"
    );
    assert_eq!(lower("import y from './y.ts';"), "import y from './y.ts';");
    assert_eq!(lower("import z from './z.js';"), "import z from './z.js';");
}

#[test]
fn import_rewriting_can_be_disabled() {
    let output = transform_with_options(
        "a.nts",
        "import { a } from './a.nts';",
        TransformOptions {
            rewrite_import_extensions: false,
        },
    );
    assert!(output.text.ends_with("import { a } from './a.nts';"));
}

#[test]
fn reports_parse_errors_and_passes_text_through() {
    let source = "const x = ;\n";
    let output = transform("a.nts", source);
    assert_eq!(output.text, source);
    assert!(!output.diagnostics.is_empty());
    let diagnostic = &output.diagnostics[0];
    assert_eq!(diagnostic.code, DiagnosticCode::Parse);
    assert_eq!(diagnostic.code.as_str(), "NMATH_NTS_PARSE");
    assert_eq!(diagnostic.line, 1);
    assert_eq!(diagnostic.column, 11);
    assert!(diagnostic.message.contains("TS"), "{}", diagnostic.message);
    assert_valid_span_map(source, &output);
}

#[test]
fn span_map_round_trips_operand_positions() {
    let source = "const total = alpha * beta + gamma[index] ** -delta;\n";
    let output = transform("a.nts", source);
    assert_valid_span_map(source, &output);
    for name in ["alpha", "beta", "gamma", "index", "delta", "total"] {
        let original = source.find(name).unwrap() as u32;
        for offset in 0..name.len() as u32 {
            let virtual_pos = output
                .span_map
                .original_to_virtual(original + offset)
                .expect("operand is verbatim");
            assert_eq!(
                &output.text[virtual_pos as usize..][..(name.len() - offset as usize)],
                &name[offset as usize..]
            );
            assert_eq!(
                output.span_map.virtual_to_original(virtual_pos),
                Some(original + offset),
                "{name}+{offset}"
            );
        }
    }
}

#[test]
fn span_map_maps_helper_calls_to_the_rewritten_expression() {
    let source = "x = a + b;";
    let output = transform("a.nts", source);
    let helper = output.text.find("__nmath.add").unwrap() as u32;
    // The helper name maps to the start of the original `a + b` expression.
    assert_eq!(output.span_map.virtual_to_original(helper + 8), Some(4));
    // The runtime import is synthesized.
    assert_eq!(output.span_map.virtual_to_original(0), None);
}

#[test]
fn line_column_counts_utf16_units() {
    let text = "a\r\nb\u{1F600}c\nd";
    assert_eq!(line_column(text, 0), (1, 1));
    assert_eq!(line_column(text, 3), (2, 1));
    assert_eq!(line_column(text, 4), (2, 2));
    assert_eq!(line_column(text, 8), (2, 4));
    assert_eq!(line_column(text, 10), (3, 1));
    assert_eq!(offset_of_line_character(text, 1, 3), 8);
    assert_eq!(offset_of_line_character(text, 2, 0), 10);
}

#[test]
fn diagnostic_codes_round_trip_through_mapper_codes() {
    for code in DiagnosticCode::ALL {
        assert_eq!(
            DiagnosticCode::from_mapper_code(code.mapper_code()),
            Some(code)
        );
    }
    assert_eq!(DiagnosticCode::from_mapper_code(99), None);
}
