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
