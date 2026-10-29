//! The library API with the in-process content mapper (no child process).

mod common;

use common::Fixture;
use nmath_compiler::{CompilerCode, ExitCode, MapperLaunch, ProjectOptions};

#[test]
fn check_and_build_with_the_in_process_mapper() {
    let fixture = Fixture::new("in-process");
    fixture.write(
        "src/main.nts",
        "const xs = [1, 2];\nexport const total: number = xs[0] * xs[1] + 1;\n",
    );
    let options = ProjectOptions {
        tsconfig: fixture.tsconfig(),
        mapper: MapperLaunch::InProcess,
    };

    let checked = nmath_compiler::check(&options);
    assert!(checked.ok, "{checked:#?}");
    assert_eq!(checked.exit_code(), ExitCode::Success);

    let built = nmath_compiler::build(&options);
    assert!(built.ok, "{built:#?}");
    assert!(
        fixture
            .read("dist/main.js")
            .contains("__nmath.add(__nmath.mul(__nmath.get(xs, 0), __nmath.get(xs, 1)), 1)")
    );

    fixture.write("src/main.nts", "export const total: string = [1][0] * 2;\n");
    let failed = nmath_compiler::check(&options);
    assert_eq!(failed.exit_code(), ExitCode::Diagnostics);
    assert_eq!(failed.diagnostics[0].code, CompilerCode::TypeError);
    assert_eq!(
        (failed.diagnostics[0].line, failed.diagnostics[0].column),
        (1, 14)
    );
}

#[test]
fn emitted_maps_locate_original_sources_and_preserve_outputs_on_error() {
    use serde_json::Value;
    use tsrs_sourcemap::decode_mappings;

    let fixture = Fixture::new("original-maps");
    let source = "import { factor } from './factor.nts';\r\nconst emoji = '😀'; export const \
                  answer = factor * (factor + 1);\r\n";
    fixture.write("src/main.nts", source);
    fixture.write("src/factor.nts", "export const factor = 3;\n");
    let options = ProjectOptions {
        tsconfig: fixture.tsconfig(),
        mapper: MapperLaunch::InProcess,
    };
    let result = nmath_compiler::build(&options);
    assert!(result.ok, "{result:#?}");
    let js = fixture.read("dist/main.js");
    assert!(js.contains("./factor.js"));
    let map_text = fixture.read("dist/main.js.map");
    let map: Value = serde_json::from_str(&map_text).unwrap();
    assert_eq!(map["sources"], serde_json::json!(["../src/main.nts"]));
    assert_eq!(map["sourcesContent"], serde_json::json!([source]));
    let generated = js.find("factor, 1").unwrap() as u32;
    let original = source.find("factor + 1").unwrap() as u32;
    let (generated_line, generated_column) = nmath_nts::line_column(&js, generated);
    let (original_line, original_column) = nmath_nts::line_column(source, original);
    let mut decoder = decode_mappings(map["mappings"].as_str().unwrap());
    let mapping = decoder
        .values()
        .find(|m| {
            m.generated_line == generated_line - 1 && m.generated_character == generated_column - 1
        })
        .expect("nested operand has a mapping");
    assert_eq!(
        (mapping.source_line, mapping.source_character),
        (Some(original_line - 1), Some(original_column - 1))
    );
    assert!(decoder.error().is_none());

    let declaration = fixture.read("dist/main.d.ts");
    let declaration_map: Value = serde_json::from_str(&fixture.read("dist/main.d.ts.map")).unwrap();
    assert_eq!(
        declaration_map["sources"],
        serde_json::json!(["../src/main.nts"])
    );
    let generated = declaration.find("answer").unwrap() as u32;
    let original = source.find("answer").unwrap() as u32;
    let (gl, gc) = nmath_nts::line_column(&declaration, generated);
    let (ol, oc) = nmath_nts::line_column(source, original);
    let mut decoder = decode_mappings(declaration_map["mappings"].as_str().unwrap());
    let mapping = decoder
        .values()
        .find(|m| m.generated_line == gl - 1 && m.generated_character == gc - 1)
        .expect("declaration name has a mapping");
    assert_eq!(
        (mapping.source_line, mapping.source_character),
        (Some(ol - 1), Some(oc - 1))
    );

    let before: Vec<_> = result
        .output_files
        .iter()
        .map(|path| (path.clone(), std::fs::read(path).unwrap()))
        .collect();
    fixture.write("src/factor.nts", "export const factor: number = 'bad';\n");
    let failed = nmath_compiler::build(&options);
    assert!(!failed.ok);
    assert!(failed.output_files.is_empty());
    assert!(
        failed
            .diagnostics
            .iter()
            .any(|d| d.file.ends_with("factor.nts") && d.line == 1 && d.column == 14)
    );
    for (path, bytes) in &before {
        assert_eq!(std::fs::read(path).unwrap(), *bytes);
    }
    fixture.write("src/factor.nts", "export const factor = ;\n");
    assert!(!nmath_compiler::build(&options).ok);
    for (path, bytes) in before {
        assert_eq!(std::fs::read(path).unwrap(), bytes);
    }
}
