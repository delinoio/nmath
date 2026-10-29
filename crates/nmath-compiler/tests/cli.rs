mod common;

use std::process::{Command, Output};

use common::Fixture;
use serde_json::Value;

const BIN: &str = env!("CARGO_BIN_EXE_nmath-compiler");

fn run(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .env_remove("NMATH_LOG")
        .output()
        .expect("nmath-compiler runs")
}

fn run_json(command: &str, fixture: &Fixture) -> (i32, Value) {
    let tsconfig = fixture.tsconfig();
    let output = run(&[command, "-p", tsconfig.to_str().unwrap(), "--json"]);
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).unwrap_or_else(|err| {
        panic!(
            "invalid JSON ({err}): {stdout}\nstderr: {}",
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output.status.code().unwrap(), value)
}

fn codes(result: &Value) -> Vec<String> {
    result["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["code"].as_str().unwrap().to_string())
        .collect()
}

const LIB: &str =
    "export function scale(xs: number[], k: number): number {\n  return xs[0] * k + -k;\n}\n";
const MAIN: &str = "import { scale } from './lib.nts';\nconst xs = [1, 2, 3];\nxs[1] = scale(xs, \
                    2);\nexport const first: number = xs[0] + 1;\n";

#[test]
fn check_succeeds_on_a_valid_project() {
    let fixture = Fixture::new("check-ok");
    fixture.write("src/lib.nts", LIB);
    fixture.write("src/main.nts", MAIN);
    fixture.write("src/plain.ts", "export const plain = 1;\n");
    let (code, result) = run_json("check", &fixture);
    assert_eq!(code, 0, "{result:#}");
    assert_eq!(result["ok"], Value::Bool(true));
    assert_eq!(result["diagnostics"], Value::Array(Vec::new()));
    assert_eq!(result["outputFiles"], Value::Array(Vec::new()));
    assert!(
        !fixture.path("dist").exists(),
        "check must not write outputs"
    );
}

#[test]
fn check_reports_type_errors_at_original_positions() {
    let fixture = Fixture::new("check-type");
    fixture.write("src/lib.nts", LIB);
    fixture.write(
        "src/main.nts",
        "import { scale } from './lib.nts';\nconst k = 2;\nconst wrong: string = scale([1], k * \
         k);\n",
    );
    let (code, result) = run_json("check", &fixture);
    assert_eq!(code, 1, "{result:#}");
    assert_eq!(codes(&result), ["NMATH_COMPILER_TYPE_ERROR"]);
    let diagnostic = &result["diagnostics"][0];
    assert!(
        diagnostic["file"]
            .as_str()
            .unwrap()
            .ends_with("/src/main.nts"),
        "{diagnostic}"
    );
    assert_eq!(
        (diagnostic["line"].as_u64(), diagnostic["column"].as_u64()),
        (Some(3), Some(7))
    );
    assert!(diagnostic["message"].as_str().unwrap().contains("TS2322"));
    assert!(!diagnostic["recovery"].as_str().unwrap().is_empty());
}

#[test]
fn check_reports_transform_diagnostics() {
    let fixture = Fixture::new("check-async");
    fixture.write(
        "src/main.nts",
        "export function f(a: number[]) {\n  a[0] = 1;\n}\n",
    );
    let (code, result) = run_json("check", &fixture);
    assert_eq!(code, 0, "{result:#}");
    assert_eq!(codes(&result), Vec::<String>::new());
    let fixture = Fixture::new("check-parse");
    fixture.write("src/main.nts", "export const x = ;\n");
    let (code, result) = run_json("check", &fixture);
    assert_eq!(code, 1, "{result:#}");
    assert_eq!(
        codes(&result),
        ["NMATH_NTS_PARSE"],
        "duplicate compiler syntax errors are dropped"
    );
    assert_eq!(
        (
            result["diagnostics"][0]["line"].as_u64(),
            result["diagnostics"][0]["column"].as_u64()
        ),
        (Some(1), Some(18))
    );
}

#[test]
fn missing_tsconfig_is_a_configuration_error() {
    let fixture = Fixture::new("missing-config");
    let missing = fixture.path("nope/tsconfig.json");
    let output = run(&["check", "-p", missing.to_str().unwrap(), "--json"]);
    assert_eq!(output.status.code(), Some(2));
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["ok"], Value::Bool(false));
    assert_eq!(codes(&result), ["NMATH_COMPILER_CONFIG"]);
}

#[test]
fn usage_errors_exit_with_two() {
    assert_eq!(run(&[]).status.code(), Some(2));
    assert_eq!(run(&["check"]).status.code(), Some(2));
    assert_eq!(run(&["frobnicate"]).status.code(), Some(2));
    assert_eq!(run(&["--help"]).status.code(), Some(0));
}

#[test]
fn human_output_lists_diagnostics() {
    let fixture = Fixture::new("human");
    fixture.write(
        "src/main.nts",
        "export function f(a: number[]) {\n  a[0] = \"bad\";\n}\n",
    );
    let tsconfig = fixture.tsconfig();
    let output = run(&["check", "-p", tsconfig.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("main.nts:2:3 - error NMATH_COMPILER_TYPE_ERROR"),
        "{stdout}"
    );
    assert!(stdout.contains("recovery:"), "{stdout}");
}

#[test]
fn build_writes_javascript_declarations_and_source_maps() {
    let fixture = Fixture::new("build-ok");
    fixture.write("src/lib.nts", LIB);
    fixture.write("src/main.nts", MAIN);
    fixture.write("src/plain.ts", "export const plain = 1;\n");
    let (code, result) = run_json("build", &fixture);
    assert_eq!(code, 0, "{result:#}");
    let mut files: Vec<String> = result["outputFiles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f.as_str().unwrap().rsplit('/').next().unwrap().to_string())
        .collect();
    files.sort();
    assert_eq!(
        files,
        [
            "lib.d.ts",
            "lib.d.ts.map",
            "lib.js",
            "lib.js.map",
            "main.d.ts",
            "main.d.ts.map",
            "main.js",
            "main.js.map",
            "plain.d.ts",
            "plain.d.ts.map",
            "plain.js",
            "plain.js.map"
        ]
    );
    let lib = fixture.read("dist/lib.js");
    assert!(!lib.contains("__nmath"), "{lib}");
    assert!(lib.contains("xs[0] * k + -k"), "{lib}");
    let main = fixture.read("dist/main.js");
    assert!(main.contains("from './lib.js'"), "{main}");
    assert!(main.contains("xs[1] = scale(xs, 2)"), "{main}");
    assert!(
        fixture
            .read("dist/lib.d.ts")
            .contains("export declare function scale")
    );
    let leftovers: Vec<_> = std::fs::read_dir(fixture.path("dist"))
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "temporary files left behind");
}

fn inherited_nmath_fixture(name: &str, no_implicit_any: bool) -> Fixture {
    let fixture = Fixture::new(name);
    fixture.write(
        "node_modules/@delinoio/nmath/package.json",
        r#"{"name":"@delinoio/nmath","type":"module","exports":{".":"./core.d.ts","./nts-runtime":"./nts-runtime.d.ts"}}"#,
    );
    fixture.write(
        "node_modules/@delinoio/nmath/core.d.ts",
        "export declare const lazyBrand: unique symbol;\nexport declare class NMathArray<D, S> { \
         readonly [lazyBrand]: true; compute(): number[]; }\nexport declare class Expr { readonly \
         [lazyBrand]: true; compute(): number; }\n",
    );
    fixture.write(
        "node_modules/@delinoio/nmath/nts-runtime.d.ts",
        "import { NMathArray, Expr } from './core.js';\nexport declare function add(a: \
         NMathArray<any, any> | Expr, b: number): Expr;\nexport declare function neg(a: \
         NMathArray<any, any> | Expr): Expr;\nexport declare function get(a: NMathArray<any, any> \
         | Expr, i: number): Expr;\n",
    );
    if no_implicit_any {
        fixture.write(
            "tsconfig.json",
            r#"{"compilerOptions":{"strict":true,"noImplicitAny":false,"module":"esnext","target":"es2022","moduleResolution":"bundler","outDir":"dist","rootDir":"src","types":[]},"include":["src"]}"#,
        );
    }
    fixture
}

#[test]
fn inherited_nmath_types_lower_arithmetic_and_indexing() {
    let fixture = inherited_nmath_fixture("inherited-nmath", true);
    fixture.write(
        "src/main.nts",
        "import { NMathArray } from '@delinoio/nmath';\ninterface Vector extends \
         NMathArray<'float64', [2]> {}\ninterface Transitive extends Vector {}\ndeclare const \
         direct: Vector;\ndeclare const transitive: Transitive;\nclass Subclass extends \
         NMathArray<'float64', [2]> {}\ndeclare const subclass: Subclass;\nfunction constrained<T \
         extends Vector>(value: T) { return value + 1; }\nexport const a = direct + 1;\nexport \
         const b = -transitive;\nexport const c = subclass[0];\nexport const d = \
         direct[0];\nexport const e = constrained(direct);\n",
    );
    let (code, result) = run_json("check", &fixture);
    assert_eq!(code, 0, "{result:#}");
    let (code, result) = run_json("build", &fixture);
    assert_eq!(code, 0, "{result:#}");
    let output = fixture.read("dist/main.js");
    assert!(output.contains("__nmath.add(value, 1)"), "{output}");
    assert!(output.contains("__nmath.add(direct, 1)"), "{output}");
    assert!(output.contains("__nmath.neg(transitive)"), "{output}");
    assert!(output.contains("__nmath.get(subclass, 0)"), "{output}");
    assert!(output.contains("__nmath.get(direct, 0)"), "{output}");
}

#[test]
fn inherited_nmath_does_not_classify_lookalikes_or_mixed_unions() {
    let fixture = inherited_nmath_fixture("inherited-nmath-negative", false);
    fixture.write(
        "src/main.nts",
        "import { NMathArray as PackageArray } from '@delinoio/nmath';\nclass NMathArray { value \
         = 1; }\ninterface Ordinary { value: number; }\ndeclare const local: NMathArray;\ndeclare \
         const ordinary: Ordinary;\nexport const a = local.value + 1;\nexport const b = \
         ordinary.value + 1;\ndeclare const mixed: PackageArray<'float64', [2]> | \
         number[];\nexport const c = mixed[0];\n",
    );
    let (code, result) = run_json("check", &fixture);
    assert_eq!(code, 1, "{result:#}");
    assert!(
        codes(&result).contains(&"NMATH_NTS_AMBIGUOUS_TYPE".to_string()),
        "{result:#}"
    );
    assert!(
        !codes(&result).contains(&"NMATH_NTS_UNSUPPORTED_SYNTAX".to_string()),
        "{result:#}"
    );
}

#[test]
fn inherited_numeric_index_executes_when_no_implicit_any_is_disabled() {
    let fixture = inherited_nmath_fixture("inherited-nmath-runtime", true);
    fixture.write(
        "node_modules/@delinoio/nmath/package.json",
        r#"{"name":"@delinoio/nmath","type":"module","exports":{".":{"types":"./core.d.ts","default":"./core.js"},"./nts-runtime":{"types":"./nts-runtime.d.ts","default":"./nts-runtime.js"}}}"#,
    );
    fixture.write(
        "node_modules/@delinoio/nmath/core.d.ts",
        "export declare const lazyBrand: unique symbol;\nexport declare class NMathArray<D, S> { \
         readonly [lazyBrand]: true; constructor(values: number[]); values: number[]; }\nexport \
         declare class Expr { readonly [lazyBrand]: true; compute(): number; }\n",
    );
    fixture.write(
        "node_modules/@delinoio/nmath/core.js",
        "export class NMathArray { constructor(values) { this.values = values; } }\n",
    );
    fixture.write(
        "node_modules/@delinoio/nmath/nts-runtime.js",
        "export function get(array, index) { return { compute: () => array.values[index] }; }\n",
    );
    fixture.write(
        "src/main.nts",
        "import { NMathArray } from '@delinoio/nmath';\ninterface Vector extends \
         NMathArray<'float64', [2]> {}\nconst a: Vector = new NMathArray<'float64', [2]>([3, \
         4]);\nexport const first = a[0].compute();\n",
    );
    let (code, result) = run_json("build", &fixture);
    assert_eq!(code, 0, "{result:#}");
    let output = fixture.read("dist/main.js");
    assert!(output.contains("__nmath.get(a, 0)"), "{output}");
    let runtime = Command::new("node")
        .arg("--input-type=module")
        .arg("-e")
        .arg("import(process.argv[1]).then(m => console.log(m.first))")
        .arg(fixture.path("dist/main.js"))
        .output()
        .expect("Node.js runs fixture");
    assert!(
        runtime.status.success(),
        "{}",
        String::from_utf8_lossy(&runtime.stderr)
    );
    assert_eq!(String::from_utf8(runtime.stdout).unwrap().trim(), "3");
}

#[test]
fn build_rewrites_mixed_module_imports_for_every_user_setting() {
    for setting in [None, Some(false), Some(true)] {
        let fixture = Fixture::new("mixed-imports");
        let rewrite_setting = setting.map_or(String::new(), |enabled| {
            format!(", \"rewriteRelativeImportExtensions\": {enabled}")
        });
        let config = format!(
            "{{\"compilerOptions\": {{\"strict\": true, \"module\": \"NodeNext\", \
             \"moduleResolution\": \"NodeNext\", \"target\": \"ES2022\", \"outDir\": \"dist\", \
             \"rootDir\": \"src\", \"types\": []{rewrite_setting}}}, \"include\": [\"src\"]}}\n"
        );
        fixture.write("tsconfig.json", &config);
        fixture.write("package.json", r#"{"type":"module"}"#);
        fixture.write("src/helper.ts", "export const value = 40;\n");
        fixture.write("src/extra.nts", "export const extra = 1;\n");
        fixture.write(
            "src/main.nts",
            "import { value } from './helper.ts';\nimport { extra } from './extra.nts';\nimport { \
             marker } from 'fixture-bare';\nexport const answer = value + extra + marker;\n",
        );
        fixture.write(
            "node_modules/fixture-bare/package.json",
            r#"{"name":"fixture-bare","type":"module","exports":{".":{"types":"./index.d.ts","default":"./index.js"}}}"#,
        );
        fixture.write(
            "node_modules/fixture-bare/index.d.ts",
            "export const marker: number;\n",
        );
        fixture.write(
            "node_modules/fixture-bare/index.js",
            "export const marker = 1;\n",
        );

        let (code, result) = run_json("build", &fixture);
        assert_eq!(code, 0, "setting={setting:?}: {result:#}");
        assert_eq!(fixture.read("tsconfig.json"), config);
        assert!(fixture.path("dist/helper.js").exists());
        assert!(fixture.path("dist/extra.js").exists());
        let output = fixture.read("dist/main.js");
        assert!(output.contains("'./helper.js'"), "{output}");
        assert!(output.contains("'./extra.js'"), "{output}");
        assert!(output.contains("'fixture-bare'"), "{output}");
        assert!(!output.contains(".ts'"), "{output}");

        std::fs::remove_dir_all(fixture.path("src")).unwrap();
        let loaded = Command::new("node")
            .args([
                "--input-type=module",
                "-e",
                "import { pathToFileURL } from 'node:url'; const module = await \
                 import(pathToFileURL(process.argv[1]).href); if (module.answer !== 42) \
                 process.exit(1);",
                fixture.path("dist/main.js").to_str().unwrap(),
            ])
            .output()
            .expect("Node.js runs the emitted ESM graph");
        assert!(
            loaded.status.success(),
            "setting={setting:?}: {}",
            String::from_utf8_lossy(&loaded.stderr)
        );
    }
}

#[test]
fn failed_mixed_module_build_preserves_output_and_config() {
    for source in [
        "import { value } from './helper.ts'; export const answer: string = value;\n",
        "import { value } from './helper.ts'; export const answer = ;\n",
    ] {
        let fixture = Fixture::new("mixed-import-fail");
        fixture.write("src/helper.ts", "export const value = 42;\n");
        fixture.write("src/main.nts", source);
        fixture.write("dist/main.js", "// previous output\n");
        let config = fixture.read("tsconfig.json");

        let (code, result) = run_json("build", &fixture);
        assert_eq!(code, 1, "{result:#}");
        assert_eq!(result["outputFiles"], Value::Array(Vec::new()));
        assert_eq!(fixture.read("dist/main.js"), "// previous output\n");
        assert!(!fixture.path("dist/helper.js").exists());
        assert_eq!(std::fs::read_dir(fixture.path("dist")).unwrap().count(), 1);
        assert_eq!(fixture.read("tsconfig.json"), config);
    }
}

#[test]
fn failed_build_keeps_existing_outputs() {
    let fixture = Fixture::new("build-fail");
    fixture.write("src/lib.nts", LIB);
    fixture.write(
        "src/main.nts",
        "import { scale } from './lib.nts';\nexport const wrong: string = scale([1], 2);\n",
    );
    fixture.write("dist/main.js", "// previous output\n");
    let (code, result) = run_json("build", &fixture);
    assert_eq!(code, 1, "{result:#}");
    assert_eq!(result["outputFiles"], Value::Array(Vec::new()));
    assert_eq!(fixture.read("dist/main.js"), "// previous output\n");
    assert!(
        !fixture.path("dist/lib.js").exists(),
        "no output may be written on error"
    );
    let entries: Vec<_> = std::fs::read_dir(fixture.path("dist")).unwrap().collect();
    assert_eq!(entries.len(), 1);
}
