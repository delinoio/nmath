fn main() {
    nmath_compiler::logging::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(nmath_compiler::cli::main(&args));
}
