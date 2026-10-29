//! `check` and `build`: drive the pinned tsrs compiler in-process with the
//! `.nts` content mapper.
//!
//! The user's `tsconfig.json` is not modified. The compiler reads the project
//! through an overlay file system that adds two virtual files next to it:
//!
//! - a wrapper config that `extends` the user's config and registers the `.nts`
//!   content mapper, and
//! - a virtual `node_modules` package whose `typescript.contentMapper` manifest
//!   runs the mapper.
//!
//! tsrs spawns the mapper through `System::spawn` with `--runExternalCode`
//! semantics, so `.nts` files are lowered, type-checked and emitted by tsrs,
//! and diagnostics are reported against the original `.nts` positions through
//! the mapper's span map. Emitted files are buffered in memory and written only
//! when the whole build succeeds.

use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime},
};

use tsrs_ast::Diagnostic;
use tsrs_core::P;
use tsrs_diagnostics::Category;
use tsrs_execute::tsc::{ExitStatus, ReadWriteCloser, System};
use tsrs_vfs::{Entries, FS, FileInfo, FileMode, bundled, osvfs};

use crate::result::{CompilerCode, CompilerDiagnostic, CompilerResult};

/// The project operation to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectCommand {
    Check,
    Build,
}

/// How the compiler starts the `.nts` content mapper.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MapperLaunch {
    /// Run `<current executable> content-mapper` as a child process (the CLI).
    CurrentExecutable,
    /// Run this command as a child process.
    Command(Vec<String>),
    /// Serve the mapper on a thread over an in-memory pipe (library callers and
    /// tests).
    InProcess,
}

#[derive(Clone, Debug)]
pub struct ProjectOptions {
    pub tsconfig: PathBuf,
    pub mapper: MapperLaunch,
}

/// The virtual npm package name of the content mapper.
const MAPPER_PACKAGE: &str = "@delinoio/nmath-nts-content-mapper";
/// The virtual wrapper config's file name, next to the user's config.
const WRAPPER_CONFIG: &str = "__nmath_compiler__.tsconfig.json";
/// The virtual config of the `.nts` JavaScript emit pass.
const EMIT_WRAPPER_CONFIG: &str = "__nmath_compiler_emit__.tsconfig.json";

pub fn run(command: ProjectCommand, options: &ProjectOptions) -> CompilerResult {
    let started = Instant::now();
    let result = run_inner(command, options);
    tracing::info!(
        ?command,
        ok = result.ok,
        diagnostics = result.diagnostics.len(),
        output_files = result.output_files.len(),
        elapsed_ms = started.elapsed().as_millis() as u64,
        "project command finished"
    );
    result
}

fn config_error(message: String) -> CompilerResult {
    CompilerResult {
        ok: false,
        diagnostics: vec![CompilerDiagnostic::without_location(
            CompilerCode::Config,
            message,
        )],
        output_files: Vec::new(),
    }
}

fn run_inner(command: ProjectCommand, options: &ProjectOptions) -> CompilerResult {
    let tsconfig = match std::fs::canonicalize(&options.tsconfig) {
        Ok(path) if path.is_file() => path,
        Ok(path) => {
            return config_error(format!(
                "The tsconfig path '{}' is not a file.",
                path.display()
            ));
        }
        Err(err) => {
            return config_error(format!(
                "The tsconfig path '{}' cannot be read: {err}.",
                options.tsconfig.display()
            ));
        }
    };
    let Some(config_dir) = tsconfig.parent().map(normalize) else {
        return config_error(format!(
            "The tsconfig path '{}' has no parent directory.",
            tsconfig.display()
        ));
    };
    let config_path = normalize(&tsconfig);
    let config_name = tsconfig
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    let mapper_command = match &options.mapper {
        MapperLaunch::CurrentExecutable => match std::env::current_exe() {
            Ok(exe) => vec![
                exe.to_string_lossy().into_owned(),
                "content-mapper".to_string(),
            ],
            Err(err) => {
                return config_error(format!(
                    "The nmath-compiler executable path is unavailable: {err}."
                ));
            }
        },
        MapperLaunch::Command(command) => command.clone(),
        MapperLaunch::InProcess => vec!["nmath-compiler".to_string(), "content-mapper".to_string()],
    };

    let wrapper_path = format!("{config_dir}/{WRAPPER_CONFIG}");
    let package_dir = format!("{config_dir}/node_modules/{MAPPER_PACKAGE}");
    let mut wrapper = serde_json::json!({
        "extends": format!("./{config_name}"),
        "contentMappers": [{ "package": MAPPER_PACKAGE, "extensions": [nmath_nts::NTS_EXTENSION] }],
    });
    let manifest = serde_json::json!({
        "name": MAPPER_PACKAGE,
        "version": env!("CARGO_PKG_VERSION"),
        "typescript": {
            "contentMapper": { "exec": mapper_command, "outputExtensions": { ".nts": ".js" } }
        },
    });
    let mut virtual_files = BTreeMap::new();
    virtual_files.insert(wrapper_path.clone(), wrapper.to_string());
    virtual_files.insert(format!("{package_dir}/package.json"), manifest.to_string());

    let plans = match analyze_project(&virtual_files, &wrapper_path, &config_dir) {
        Ok(plans) => plans,
        Err(message) => return config_error(message),
    };
    let mut transforms = serde_json::Map::new();
    for (path, plan) in &plans {
        let source = match std::fs::read_to_string(path) {
            Ok(source) => source,
            Err(error) => return config_error(format!("Cannot read '{path}': {error}")),
        };
        let output = nmath_nts::transform_with_plan(
            path,
            &source,
            nmath_nts::TransformOptions {
                rewrite_import_extensions: false,
            },
            plan,
        );
        transforms.insert(
            path.clone(),
            serde_json::json!({
                "source": source,
                "result": crate::content_mapper::output_result(&output),
            }),
        );
    }
    wrapper["contentMappers"][0]["options"] = serde_json::json!({ "transforms": transforms });
    virtual_files.insert(wrapper_path.clone(), wrapper.to_string());

    let mut args = vec![
        "--runExternalCode".to_string(),
        // The mapper's `.nts` -> `.js` output extension requires the relative import rewrite
        // (TS18120). It also turns `./x.nts` specifiers into `./x.js` in emitted
        // declarations.
        "--rewriteRelativeImportExtensions".to_string(),
    ];
    match command {
        ProjectCommand::Check => args.push("--noEmit".to_string()),
        ProjectCommand::Build => {
            args.extend(
                [
                    "--declaration",
                    "--declarationMap",
                    "--sourceMap",
                    "--noEmit",
                    "false",
                ]
                .map(String::from),
            );
        }
    }
    let names = FileNames {
        wrapper: wrapper_path,
        config: config_path.clone(),
    };
    let pass = run_pass(virtual_files, &names, &config_dir, &options.mapper, args);
    let mut diagnostics = pass.diagnostics;
    dedupe_parse_diagnostics(&mut diagnostics);

    let mut result = CompilerResult {
        ok: diagnostics.is_empty(),
        diagnostics,
        output_files: Vec::new(),
    };
    if command != ProjectCommand::Build || !result.ok {
        return result;
    }
    let mut writes = pass.writes;
    match emit_nts_javascript(&pass.nts_files, &names, &config_dir, &config_name, &plans) {
        Ok(nts_writes) => writes.extend(nts_writes),
        Err(diagnostics) => {
            result.ok = false;
            result.diagnostics = diagnostics;
            return result;
        }
    }
    match commit_outputs(&writes) {
        Ok(files) => result.output_files = files,
        Err(diagnostic) => {
            result.ok = false;
            result.diagnostics.push(diagnostic);
        }
    }
    result
}

/// What one in-process tsrs run produced.
struct PassOutput {
    diagnostics: Vec<CompilerDiagnostic>,
    /// Buffered output files by absolute path.
    writes: BTreeMap<String, String>,
    /// The `.nts` files the compiler read (outside `node_modules`).
    nts_files: BTreeSet<String>,
}

/// Runs tsrs once on `names.wrapper` (a virtual file in `virtual_files`) with
/// extra command-line arguments.
fn run_pass(
    virtual_files: BTreeMap<String, String>,
    names: &FileNames,
    cwd: &str,
    launch: &MapperLaunch,
    extra_args: Vec<String>,
) -> PassOutput {
    let fs = Arc::new(OverlayFs::new(virtual_files));
    let diagnostics = Arc::new(Mutex::new(Vec::new()));
    let sink_diagnostics = Arc::clone(&diagnostics);
    let sink_names = FileNames {
        wrapper: names.wrapper.clone(),
        config: names.config.clone(),
    };
    // tsrs keeps `&'static` references to its system and config host for the
    // life of the parsed command line, so the system is leaked. A CLI
    // process runs at most two passes.
    let sys: &'static ProjectSystem = Box::leak(Box::new(ProjectSystem {
        fs: Arc::clone(&fs),
        cwd: cwd.to_string(),
        start: Instant::now(),
        launch: launch.clone(),
        output: Mutex::new(String::new()),
        sink: Box::new(move |diagnostic| {
            if let Some(converted) = convert_diagnostic(diagnostic, &sink_names) {
                sink_diagnostics.lock().unwrap().push(converted);
            }
        }),
    }));
    let mut args = vec![
        "-p".to_string(),
        names.wrapper.clone(),
        "--pretty".to_string(),
        "false".to_string(),
    ];
    args.extend(extra_args);
    tracing::debug!(config = %names.config, ?args, "running tsrs");
    let status = tsrs_execute::execute::command_line(sys, args).status;

    let mut diagnostics = std::mem::take(&mut *diagnostics.lock().unwrap());
    let output = std::mem::take(&mut *sys.output.lock().unwrap());
    let status_ok = matches!(
        status,
        ExitStatus::Success | ExitStatus::DiagnosticsPresentOutputsGenerated
    );
    tracing::debug!(?status, diagnostics = diagnostics.len(), "tsrs finished");
    if diagnostics.is_empty() && !status_ok {
        // tsrs stopped without a reportable diagnostic (for example an
        // unsupported option); surface its text.
        let message = if output.trim().is_empty() {
            format!("tsrs exited with {status:?}.")
        } else {
            output.trim().to_string()
        };
        diagnostics.push(CompilerDiagnostic::without_location(
            CompilerCode::Config,
            message,
        ));
    }
    PassOutput {
        diagnostics,
        writes: fs.take_writes(),
        nts_files: std::mem::take(&mut *fs.nts_reads.lock().unwrap()),
    }
}

/// Emits JavaScript and source maps for `.nts` files.
///
/// Workaround: tsrs (like TypeScript 7.1) never emits runtime JavaScript for
/// content-mapped files; it leaves that to the mapper's build tool
/// (`tsrs_compiler` emitter.rs, "Runtime output for content-mapped files is
/// owned by the external content mapper or build tool"). After the checked pass
/// succeeds, a second tsrs pass compiles the lowered TypeScript of each `.nts`
/// file as a virtual `.ts` file next to it with `--noCheck`, so the output
/// layout and compiler options match the project. Compose its source maps with
/// the transform maps before publishing any output, so debugging uses `.nts`.
/// Remove this pass if tsrs gains a mapper-owned JavaScript emit hook at a
/// pinned commit.
fn emit_nts_javascript(
    nts_files: &BTreeSet<String>,
    names: &FileNames,
    config_dir: &str,
    config_name: &str,
    plans: &BTreeMap<String, nmath_nts::semantic::LoweringPlan>,
) -> Result<BTreeMap<String, String>, Vec<CompilerDiagnostic>> {
    if nts_files.is_empty() {
        return Ok(BTreeMap::new());
    }
    let mut virtual_files = BTreeMap::new();
    let mut virtual_to_nts = BTreeMap::new();
    let mut mapped_sources = BTreeMap::new();
    let mut errors = Vec::new();
    for nts in nts_files {
        let lowered_path = format!(
            "{}.ts",
            nts.strip_suffix(nmath_nts::NTS_EXTENSION).unwrap_or(nts)
        );
        if Path::new(&lowered_path).exists() {
            errors.push(CompilerDiagnostic::new(
                CompilerCode::Emit,
                nts.clone(),
                0,
                0,
                format!("'{nts}' and '{lowered_path}' would both emit the same JavaScript file."),
            ));
            continue;
        }
        let source = match std::fs::read_to_string(nts) {
            Ok(source) => source,
            Err(err) => {
                errors.push(CompilerDiagnostic::new(
                    CompilerCode::Emit,
                    nts.clone(),
                    0,
                    0,
                    format!("The file cannot be read: {err}."),
                ));
                continue;
            }
        };
        let transformed = nmath_nts::transform_with_plan(
            nts,
            &source,
            nmath_nts::TransformOptions::default(),
            &plans.get(nts).cloned().unwrap_or_default(),
        );
        virtual_files.insert(lowered_path.clone(), transformed.text.clone());
        mapped_sources.insert(lowered_path.clone(), (nts.clone(), source, transformed));
        virtual_to_nts.insert(lowered_path, nts.clone());
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    let wrapper_path = format!("{config_dir}/{EMIT_WRAPPER_CONFIG}");
    let wrapper = serde_json::json!({
        "extends": format!("./{config_name}"),
        "files": virtual_to_nts.keys().collect::<Vec<_>>(),
        "include": [],
    });
    virtual_files.insert(wrapper_path.clone(), wrapper.to_string());
    let emit_names = FileNames {
        wrapper: wrapper_path,
        config: names.config.clone(),
    };
    let args = [
        "--noCheck",
        "--declaration",
        "false",
        "--declarationMap",
        "false",
        "--sourceMap",
        "--inlineSources",
        "--inlineSourceMap",
        "false",
        "--sourceRoot",
        "",
        "--mapRoot",
        "",
        "--noEmit",
        "false",
    ]
    .map(String::from)
    .to_vec();
    let pass = run_pass(
        virtual_files,
        &emit_names,
        config_dir,
        &MapperLaunch::InProcess,
        args,
    );
    if !pass.diagnostics.is_empty() {
        return Err(pass
            .diagnostics
            .into_iter()
            .map(|d| {
                let file = virtual_to_nts.get(&d.file).cloned().unwrap_or(d.file);
                CompilerDiagnostic::new(
                    CompilerCode::Emit,
                    file,
                    0,
                    0,
                    format!("Emitting lowered .nts output failed: {}", d.message),
                )
            })
            .collect());
    }
    let mut writes = pass.writes;
    for (path, text) in &mut writes {
        if path.ends_with(".js.map") {
            *text =
                crate::source_maps::compose(path, text, &mapped_sources).map_err(|message| {
                    vec![CompilerDiagnostic::without_location(
                        CompilerCode::Emit,
                        message,
                    )]
                })?;
        }
    }
    Ok(writes)
}

/// Drops compiler syntax errors that repeat an `NMATH_NTS_PARSE` diagnostic at
/// the same location: a `.nts` file with syntax errors is passed through
/// unchanged, so tsrs reports the same errors again.
fn dedupe_parse_diagnostics(diagnostics: &mut Vec<CompilerDiagnostic>) {
    let parse_locations: BTreeSet<(String, u32, u32)> = diagnostics
        .iter()
        .filter(|d| d.code == CompilerCode::NtsParse)
        .map(|d| (d.file.clone(), d.line, d.column))
        .collect();
    diagnostics.retain(|d| {
        d.code == CompilerCode::NtsParse
            || !parse_locations.contains(&(d.file.clone(), d.line, d.column))
    });
}

/// Writes every buffered output to a temporary file, then renames them all into
/// place. On any failure the temporary files are removed and no existing output
/// is replaced.
fn commit_outputs(writes: &BTreeMap<String, String>) -> Result<Vec<String>, CompilerDiagnostic> {
    let mut staged: Vec<(PathBuf, PathBuf)> = Vec::new();
    let cleanup = |staged: &[(PathBuf, PathBuf)]| {
        for (temp, _) in staged {
            let _ = std::fs::remove_file(temp);
        }
    };
    for (path, content) in writes {
        let target = PathBuf::from(path);
        let temp = temp_path(&target);
        let written = target
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::File::create(&temp))
            .and_then(|mut file| {
                file.write_all(content.as_bytes())
                    .and_then(|()| file.sync_all())
            });
        if let Err(err) = written {
            let _ = std::fs::remove_file(&temp);
            cleanup(&staged);
            tracing::error!(file = %path, error = %err, "staging output failed");
            return Err(CompilerDiagnostic::new(
                CompilerCode::Emit,
                path,
                0,
                0,
                format!("Output could not be written: {err}."),
            ));
        }
        staged.push((temp, target));
    }
    // Each committed output remembers the backup of the file it replaced, so a
    // later failure can restore every earlier target.
    let mut committed: Vec<(&PathBuf, Option<PathBuf>)> = Vec::new();
    for (index, (temp, target)) in staged.iter().enumerate() {
        let result = replace_with_backup(temp, target);
        match result {
            Ok(backup) => committed.push((target, backup)),
            Err(err) => {
                cleanup(&staged[index..]);
                rollback(&committed);
                tracing::error!(file = %target.display(), error = %err, "renaming output failed");
                return Err(CompilerDiagnostic::new(
                    CompilerCode::Emit,
                    target.to_string_lossy(),
                    0,
                    0,
                    format!("Output could not be moved into place: {err}."),
                ));
            }
        }
    }
    for (_, backup) in &committed {
        if let Some(backup) = backup {
            let _ = std::fs::remove_file(backup);
        }
    }
    Ok(committed
        .iter()
        .map(|(target, _)| target.to_string_lossy().into_owned())
        .collect())
}

/// Moves an existing target file aside, then renames `temp` into place. On
/// failure the target is left as it was.
fn replace_with_backup(temp: &Path, target: &Path) -> std::io::Result<Option<PathBuf>> {
    let backup = match std::fs::symlink_metadata(target) {
        Ok(meta) if meta.is_file() => {
            let backup = backup_path(target);
            std::fs::rename(target, &backup)?;
            Some(backup)
        }
        // A directory or other non-file target makes the rename below fail and
        // is never moved.
        Ok(_) => None,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(err) => return Err(err),
    };
    if let Err(err) = std::fs::rename(temp, target) {
        if let Some(backup) = &backup {
            let _ = std::fs::rename(backup, target);
        }
        return Err(err);
    }
    Ok(backup)
}

/// Restores earlier outputs in reverse order after a failed commit.
fn rollback(committed: &[(&PathBuf, Option<PathBuf>)]) {
    for (target, backup) in committed.iter().rev() {
        let restored = match backup {
            Some(backup) => std::fs::rename(backup, target),
            None => std::fs::remove_file(target),
        };
        if let Err(err) = restored {
            tracing::error!(file = %target.display(), error = %err, "restoring output failed");
        }
    }
}

fn backup_path(target: &Path) -> PathBuf {
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    target.with_file_name(format!(".{name}.nmath-{}.bak", std::process::id()))
}

fn temp_path(target: &Path) -> PathBuf {
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    target.with_file_name(format!(".{name}.nmath-{}.tmp", std::process::id()))
}

/// A tsrs-style path: absolute, forward slashes.
fn normalize(path: &Path) -> String {
    tsrs_core::tspath::normalize_path(&path.to_string_lossy())
}

struct FileNames {
    wrapper: String,
    config: String,
}

struct ProjectSystem {
    fs: Arc<OverlayFs>,
    cwd: String,
    start: Instant,
    launch: MapperLaunch,
    output: Mutex<String>,
    /// Receives every reported diagnostic (from checker threads) instead of
    /// tsrs's text reporters.
    sink: Box<dyn Fn(P<Diagnostic>) + Sync + Send>,
}

impl System for ProjectSystem {
    fn fs(&self) -> Arc<dyn FS> {
        Arc::clone(&self.fs) as Arc<dyn FS>
    }

    fn default_library_path(&self) -> &str {
        static LIB_PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        LIB_PATH.get_or_init(bundled::default_lib_path)
    }

    fn get_current_directory(&self) -> &str {
        &self.cwd
    }

    fn write(&self, text: &str) {
        self.output.lock().unwrap().push_str(text);
    }

    fn flush(&self) {}

    fn write_output_is_tty(&self) -> bool {
        false
    }

    fn get_environment_variable(&self, name: &str) -> Option<String> {
        std::env::var(name).ok()
    }

    fn spawn(
        &self,
        command: &[String],
        dir: &str,
        stderr: Option<Box<dyn Write + Send>>,
    ) -> Result<ReadWriteCloser, String> {
        tracing::debug!(launch = ?self.launch, "starting the .nts content mapper");
        match self.launch {
            MapperLaunch::InProcess => {
                let (host, mapper) = tsrs_ipc::pipe().map_err(|err| err.to_string())?;
                std::thread::Builder::new()
                    .name("nmath-content-mapper".to_string())
                    .spawn(move || {
                        if let Err(err) = crate::content_mapper::serve(mapper.reader, mapper.writer)
                        {
                            tracing::debug!(error = %err, "in-process content mapper stopped");
                        }
                    })
                    .map_err(|err| err.to_string())?;
                Ok(host)
            }
            MapperLaunch::CurrentExecutable | MapperLaunch::Command(_) => {
                // tsrs runs a mapper in its package directory, which is virtual
                // here; run it in the project instead.
                let dir = if self.fs.inner.directory_exists(dir) {
                    dir
                } else {
                    self.cwd.as_str()
                };
                tsrs_contentmapper::spawn_process(command, dir, stderr)
            }
        }
    }

    fn now(&self) -> Instant {
        Instant::now()
    }

    fn since_start(&self) -> Duration {
        self.start.elapsed()
    }

    fn diagnostic_sink(&self) -> Option<&(dyn Fn(P<Diagnostic>) + Sync)> {
        Some(&*self.sink)
    }
}

/// Converts a reported tsrs diagnostic to a compiler diagnostic at its original
/// location. Non-errors are dropped.
fn convert_diagnostic(d: P<Diagnostic>, names: &FileNames) -> Option<CompilerDiagnostic> {
    if d.category() != Category::Error {
        return None;
    }
    let mapper_code = crate::content_mapper::diagnostic_code(d.source(), d.code());
    let mut message = d.localize();
    append_chain(&mut message, d.message_chain(), 1);
    // Messages name the config tsrs was given; show the user's config instead
    // of the virtual wrapper.
    let message = message
        .replace(&names.wrapper, &names.config)
        .replace(basename(&names.wrapper), basename(&names.config));

    let Some(file) = d.file() else {
        let code = if (5000..7000).contains(&d.code()) {
            CompilerCode::Config
        } else {
            CompilerCode::TypeError
        };
        return Some(CompilerDiagnostic::without_location(
            code,
            format!("TS{}: {message}", d.code()),
        ));
    };
    let loc = d.loc().unwrap_or_else(tsrs_core::TextRange::synthetic);
    let start = loc.pos().source_offset().unwrap_or(0);
    // Mirror tsrs's diagnostic writer: mapper-authored diagnostics are already
    // in original coordinates; others map through the span map, and a
    // location entirely in synthesized code stays in the virtual text.
    let (text, position) = if !d.source().is_empty() {
        (file.original_text(), start)
    } else if let Some(span_map) = file.span_map() {
        let (mapped, fidelity) = span_map.virtual_to_original_span(loc);
        if fidelity == tsrs_spanmap::Fidelity::None {
            (file.text(), start)
        } else {
            (
                file.original_text(),
                mapped.pos().source_offset().unwrap_or(0),
            )
        }
    } else {
        (file.text(), start)
    };
    let (line, column) = nmath_nts::line_column(text, position);
    let mut file_name = file.original_file_name().to_string();
    let (mut line, mut column) = (line, column);
    if file_name == names.wrapper {
        // The location points into the virtual wrapper config, which the user
        // never sees.
        file_name = names.config.clone();
        (line, column) = (0, 0);
    }
    let (code, message) = match mapper_code {
        Some(code) => (CompilerCode::from(code), message),
        None if file_name.ends_with(".json") => {
            (CompilerCode::Config, format!("TS{}: {message}", d.code()))
        }
        None => (
            CompilerCode::TypeError,
            format!("TS{}: {message}", d.code()),
        ),
    };
    Some(CompilerDiagnostic::new(
        code, file_name, line, column, message,
    ))
}

fn basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn append_chain(message: &mut String, chain: &[P<Diagnostic>], level: usize) {
    for link in chain {
        message.push('\n');
        message.push_str(&"  ".repeat(level));
        message.push_str(&link.localize());
        append_chain(message, link.message_chain(), level + 1);
    }
}

/// The OS file system with virtual read-only files and buffered writes.
struct OverlayFs {
    inner: Box<dyn FS>,
    files: BTreeMap<String, String>,
    directories: BTreeSet<String>,
    writes: Mutex<BTreeMap<String, String>>,
    nts_reads: Mutex<BTreeSet<String>>,
}

impl OverlayFs {
    fn new(files: BTreeMap<String, String>) -> OverlayFs {
        let mut directories = BTreeSet::new();
        for path in files.keys() {
            let mut dir = parent(path);
            while let Some(d) = dir {
                directories.insert(d.to_string());
                dir = parent(d);
            }
        }
        OverlayFs {
            inner: Box::new(bundled::wrap_fs(osvfs::fs())),
            files,
            directories,
            writes: Mutex::new(BTreeMap::new()),
            nts_reads: Mutex::new(BTreeSet::new()),
        }
    }

    fn take_writes(&self) -> BTreeMap<String, String> {
        std::mem::take(&mut *self.writes.lock().unwrap())
    }

    fn virtual_file(&self, path: &str) -> Option<String> {
        if let Some(text) = self.writes.lock().unwrap().get(path) {
            return Some(text.clone());
        }
        self.files.get(path).cloned()
    }
}

fn parent(path: &str) -> Option<&str> {
    let index = path.rfind('/')?;
    (index > 0).then(|| &path[..index])
}

impl FS for OverlayFs {
    fn use_case_sensitive_file_names(&self) -> bool {
        self.inner.use_case_sensitive_file_names()
    }

    fn file_exists(&self, path: &str) -> bool {
        self.virtual_file(path).is_some() || self.inner.file_exists(path)
    }

    fn read_file(&self, path: &str) -> Option<String> {
        if nmath_nts::is_nts_file(path)
            && !path.contains("/node_modules/")
            && !self.files.contains_key(path)
        {
            self.nts_reads.lock().unwrap().insert(path.to_string());
        }
        self.virtual_file(path)
            .or_else(|| self.inner.read_file(path))
    }

    fn write_file(&self, path: &str, data: &str) -> Result<(), String> {
        tracing::debug!(file = path, bytes = data.len(), "buffering output");
        self.writes
            .lock()
            .unwrap()
            .insert(path.to_string(), data.to_string());
        Ok(())
    }

    fn append_file(&self, path: &str, data: &str) -> Result<(), String> {
        let existing = self.read_file(path).unwrap_or_default();
        self.write_file(path, &(existing + data))
    }

    fn remove(&self, path: &str) -> Result<(), String> {
        // Builds never delete user files; a removal only drops a buffered
        // output.
        self.writes.lock().unwrap().remove(path);
        Ok(())
    }

    fn chtimes(&self, _path: &str, _a_time: SystemTime, _m_time: SystemTime) -> Result<(), String> {
        Ok(())
    }

    fn directory_exists(&self, path: &str) -> bool {
        self.directories.contains(path) || self.inner.directory_exists(path)
    }

    fn get_accessible_entries(&self, path: &str) -> Entries {
        let mut entries = if self.inner.directory_exists(path) {
            self.inner.get_accessible_entries(path)
        } else {
            Entries {
                files: Vec::new(),
                directories: Vec::new(),
                symlinks: None,
            }
        };
        let prefix = format!("{path}/");
        for dir in &self.directories {
            if let Some(name) = dir.strip_prefix(&prefix)
                && !name.contains('/')
                && !entries.directories.iter().any(|d| d == name)
            {
                entries.directories.push(name.to_string());
            }
        }
        for file in self.files.keys() {
            // The wrapper config is read by name only; listing it would add it
            // to globbed `.json` includes.
            if let Some(name) = file.strip_prefix(&prefix)
                && !name.contains('/')
                && name != WRAPPER_CONFIG
                && name != EMIT_WRAPPER_CONFIG
                && !entries.files.iter().any(|f| f == name)
            {
                entries.files.push(name.to_string());
            }
        }
        entries
    }

    fn stat(&self, path: &str) -> Option<FileInfo> {
        let name = path.rsplit('/').next().unwrap_or(path).to_string();
        if let Some(text) = self.virtual_file(path) {
            return Some(FileInfo {
                name,
                size: text.len() as i64,
                mode: FileMode::None,
                mod_time: None,
            });
        }
        if self.directories.contains(path) && !self.inner.directory_exists(path) {
            return Some(FileInfo {
                name,
                size: 0,
                mode: FileMode::Dir,
                mod_time: None,
            });
        }
        self.inner.stat(path)
    }

    fn realpath(&self, path: &str) -> String {
        if self.files.contains_key(path)
            || (self.directories.contains(path) && !self.inner.directory_exists(path))
        {
            return path.to_string();
        }
        self.inner.realpath(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedupes_compiler_errors_at_parse_diagnostic_locations() {
        let mut diagnostics = vec![
            CompilerDiagnostic::new(CompilerCode::NtsParse, "a.nts", 1, 5, "parse"),
            CompilerDiagnostic::new(CompilerCode::TypeError, "a.nts", 1, 5, "TS1109"),
            CompilerDiagnostic::new(CompilerCode::TypeError, "a.nts", 2, 1, "TS2322"),
        ];
        dedupe_parse_diagnostics(&mut diagnostics);
        assert_eq!(diagnostics.len(), 2);
        assert_eq!(diagnostics[1].message, "TS2322");
    }

    #[test]
    fn overlay_lists_virtual_package_directories() {
        let mut files = BTreeMap::new();
        files.insert(
            "/nonexistent-nmath-root/node_modules/@scope/pkg/package.json".to_string(),
            "{}".to_string(),
        );
        let fs = OverlayFs::new(files);
        assert!(fs.directory_exists("/nonexistent-nmath-root/node_modules/@scope"));
        let entries = fs.get_accessible_entries("/nonexistent-nmath-root/node_modules/@scope/pkg");
        assert_eq!(entries.files, vec!["package.json".to_string()]);
        assert!(fs.write_file("/nonexistent-nmath-root/out.js", "x").is_ok());
        assert_eq!(
            fs.read_file("/nonexistent-nmath-root/out.js").as_deref(),
            Some("x")
        );
        assert_eq!(fs.take_writes().len(), 1);
    }

    #[test]
    fn failed_commit_restores_earlier_outputs() {
        let dir = std::env::temp_dir().join(format!(
            "nmath-commit-rollback-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(dir.join("b.js")).unwrap();
        std::fs::write(dir.join("a.js"), "old").unwrap();
        let mut writes = BTreeMap::new();
        writes.insert(normalize(&dir.join("a.js")), "new a".to_string());
        writes.insert(normalize(&dir.join("a.d.ts")), "new decl".to_string());
        // `b.js` is a directory, so its rename fails after `a.d.ts` and `a.js`
        // commit.
        writes.insert(normalize(&dir.join("b.js")), "new b".to_string());
        assert!(commit_outputs(&writes).is_err());
        assert_eq!(std::fs::read_to_string(dir.join("a.js")).unwrap(), "old");
        assert!(!dir.join("a.d.ts").exists());
        let mut leftovers: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        leftovers.sort();
        assert_eq!(leftovers, vec!["a.js".to_string(), "b.js".to_string()]);

        writes.remove(&normalize(&dir.join("b.js")));
        let files = commit_outputs(&writes).unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(std::fs::read_to_string(dir.join("a.js")).unwrap(), "new a");
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            3,
            "no backups remain"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// Build standard tsrs programs until all proven operations have been lowered.
/// Each pass adds at least one source operation or terminates; there is no
/// arbitrary iteration limit that would truncate long inference chains.
/// `virtual_files` may overlay unsaved sources. The config must register the
/// NMath content mapper; no external mapper is executed during this analysis.
pub fn analyze_project(
    virtual_files: &BTreeMap<String, String>,
    config_path: &str,
    cwd: &str,
) -> Result<BTreeMap<String, nmath_nts::semantic::LoweringPlan>, String> {
    let mut plans = BTreeMap::new();
    loop {
        let fs = Arc::new(OverlayFs::new(virtual_files.clone()));
        let parse_host = SemanticConfigHost {
            fs: Arc::clone(&fs),
            cwd: cwd.into(),
        };
        let compiler_options = tsrs_core::CompilerOptions {
            run_external_code: tsrs_core::Tristate::True,
            rewrite_relative_import_extensions: tsrs_core::Tristate::True,
            ..Default::default()
        };
        let (config, errors) = tsrs_tsoptions::get_parsed_command_line_of_config_file(
            config_path,
            Some(&compiler_options),
            None,
            &parse_host,
            None,
        );
        let Some(config) = config else {
            return Err(errors
                .iter()
                .map(|d| d.localize())
                .collect::<Vec<_>>()
                .join("\n"));
        };
        let mapper = Arc::new(SemanticMapper {
            plans: plans.clone(),
        });
        let host = tsrs_compiler::new_compiler_host(
            cwd,
            fs,
            &bundled::default_lib_path(),
            None,
            None,
            Some(mapper),
        );
        let mut options = tsrs_compiler::ProgramOptions::new(P::new(config), host);
        options.single_threaded = tsrs_core::Tristate::True;
        options.regions = Some(Arc::new(tsrs_compiler::ProgramRegions::default()));
        options.checker_regions = true;
        options.checker_diagnostic_regions = true;
        let program = tsrs_compiler::new_program(options);
        let context = tsrs_core::Context::default();
        let mut next = BTreeMap::new();
        for file in program.source_files() {
            if !nmath_nts::is_nts_file(file.file_name()) {
                continue;
            }
            let mut checker = program.get_type_checker_for_file(&context, *file);
            let previous = plans.get(file.file_name()).cloned().unwrap_or_default();
            let plan = nmath_nts::semantic::classify_file(&mut checker, *file, &previous);
            next.insert(file.file_name().to_string(), plan);
        }
        // SAFETY: all checker handles are dropped and the result owns only
        // source offsets and diagnostic strings. No AST or type pointer
        // escapes.
        unsafe {
            tsrs_compiler::free_unshared_program(program);
        }
        let changed = next.iter().any(|(path, plan)| {
            plans
                .get(path)
                .is_none_or(|old| old.operations != plan.operations)
        });
        plans = next;
        if !changed {
            return Ok(plans);
        }
    }
}

struct SemanticConfigHost {
    fs: Arc<OverlayFs>,
    cwd: String,
}
impl tsrs_tsoptions::ParseConfigHost for SemanticConfigHost {
    fn fs(&self) -> &dyn FS {
        &*self.fs
    }

    fn get_current_directory(&self) -> &str {
        &self.cwd
    }
}
struct SemanticMapper {
    plans: BTreeMap<String, nmath_nts::semantic::LoweringPlan>,
}
impl tsrs_contentmapper::Project for SemanticMapper {
    fn refresh(&self) -> Result<(), tsrs_contentmapper::Error> {
        Ok(())
    }

    fn identities(&self) -> Result<Vec<String>, tsrs_contentmapper::Error> {
        Ok(vec!["nmath-semantic".into()])
    }

    fn identity(
        &self,
        _: &tsrs_contentmapper::Mapper,
    ) -> Result<String, tsrs_contentmapper::Error> {
        Ok("nmath-semantic".into())
    }

    fn watched_files(&self) -> Result<Vec<String>, tsrs_contentmapper::Error> {
        Ok(vec![])
    }

    fn diagnostics(&self) -> Vec<tsrs_contentmapper::OptionDiagnostic> {
        vec![]
    }

    fn close(&self) -> Result<(), tsrs_contentmapper::Error> {
        Ok(())
    }

    fn transform(
        &self,
        _: &tsrs_contentmapper::Mapper,
        request: tsrs_contentmapper::Request<'_>,
    ) -> Result<tsrs_contentmapper::TransformResultFiles, tsrs_contentmapper::Error> {
        let mut plan = self
            .plans
            .get(request.file_name)
            .cloned()
            .unwrap_or_default();
        // Intermediate ambiguity can disappear after inference sees a lowered
        // producer. Only the converged pass reports semantic diagnostics.
        plan.diagnostics.clear();
        let output = nmath_nts::transform_with_plan(
            request.file_name,
            request.content,
            nmath_nts::TransformOptions {
                rewrite_import_extensions: false,
            },
            &plan,
        );
        let segments: Vec<_> = output
            .span_map
            .segments()
            .iter()
            .map(|s| tsrs_spanmap::Segment {
                virtual_start: s.virtual_start,
                virtual_end: s.virtual_end,
                original_start: s.original_start,
                original_end: s.original_end,
                kind: match s.kind {
                    nmath_nts::SegmentKind::Verbatim => tsrs_spanmap::Kind::VERBATIM,
                    nmath_nts::SegmentKind::Atom => tsrs_spanmap::Kind::ATOM,
                },
                features: tsrs_spanmap::Feature::all(),
            })
            .collect();
        Ok(tsrs_contentmapper::TransformResultFiles {
            text: output.text,
            virtual_extension: ".ts".into(),
            mappings: Some(tsrs_spanmap::new(&segments)),
            ..Default::default()
        })
    }
}
