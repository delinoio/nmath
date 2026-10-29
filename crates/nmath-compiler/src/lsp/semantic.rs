//! Semantic requests use an immutable editor snapshot and the pinned tsrs
//! language service. The direct mapper avoids the tsrs LSP process-spawn gap;
//! remove this adapter when that server can host the NMath mapper itself.

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
    sync::Arc,
};

use serde_json::{Value, json};
use tsrs_compiler::{ProgramOptions, new_compiler_host, new_program};
use tsrs_contentmapper as mapper;
use tsrs_core::{P, TextRange, Tristate, arena::Region, context::Context};
use tsrs_ls::{
    Host,
    autoimport::{ProjectID, Registry},
    lsconv::{self, Converters},
    lsutil::{UserPreferences, new_default_user_preferences},
    sourcemap::ECMALineInfo,
};
use tsrs_lsproto::{self as proto, Json};
use tsrs_tsoptions::ParseConfigHost;
use tsrs_vfs::{FS, bundled, vfsmatch};

use super::snapshot::Snapshot;

pub enum Query {
    Diagnostics,
    Completion(proto::Position),
    Definition(proto::Position),
}

pub fn query(
    documents: &HashMap<String, String>,
    uri: &str,
    query: Query,
) -> Result<Value, String> {
    let uri = proto::DocumentUri(uri.to_string());
    if uri.check().is_err() {
        return Err("Invalid document URI".into());
    }
    let file = uri.file_name();
    // All tsrs arena data is request-local. Program/checker regions release
    // worker allocations too, so repeated edits do not retain old programs.
    let region = Region::new(1 << 20);
    let _scope = region.enter();
    let mut files = HashMap::new();
    for (uri, text) in documents {
        let uri = proto::DocumentUri(uri.clone());
        if uri.check().is_ok() {
            files.insert(uri.file_name(), text.clone());
        }
    }
    let directory = Path::new(&file).parent().unwrap_or(Path::new("/"));
    let config = directory
        .ancestors()
        .map(|p| p.join("tsconfig.json"))
        .find(|p| p.is_file());
    let cwd = config
        .as_ref()
        .and_then(|p| p.parent())
        .unwrap_or(directory)
        .to_string_lossy()
        .into_owned();
    let wrapper = format!("{cwd}/__nmath_lsp__.tsconfig.json");
    let package = "@delinoio/nmath-lsp-mapper";
    let mut settings = json!({
        "contentMappers": [{"package":package,"extensions":[".nts"]}],
        "compilerOptions": {"rewriteRelativeImportExtensions":true}
    });
    if let Some(config) = &config {
        settings["extends"] = json!(config.to_string_lossy());
    } else {
        settings["files"] = json!([file]);
        settings["compilerOptions"]["strict"] = json!(true);
        settings["compilerOptions"]["target"] = json!("esnext");
        settings["compilerOptions"]["module"] = json!("nodenext");
    }
    files.insert(wrapper.clone(), settings.to_string());
    files.insert(format!("{cwd}/node_modules/{package}/package.json"), json!({
        "name":package,"version":"0.0.0",
        "typescript":{"contentMapper":{"exec":["nmath-in-process"],"outputExtensions":{".nts":".js"}}}
    }).to_string());
    if config.is_some() {
        let fs: Arc<dyn FS> = Arc::new(Snapshot::new(files.clone()));
        let host = ConfigHost {
            fs,
            cwd: cwd.clone(),
        };
        let (parsed, errors) = tsrs_tsoptions::get_parsed_command_line_of_config_file(
            &wrapper, None, None, &host, None,
        );
        let parsed = parsed.ok_or_else(|| {
            errors
                .iter()
                .map(|d| d.localize())
                .collect::<Vec<_>>()
                .join("\n")
        })?;
        if !parsed.file_names().contains(&file) {
            // Keep the effective roots, whose paths tsrs resolved against their
            // owning configs. The query may be outside the configured project.
            let mut roots = parsed.file_names().to_vec();
            roots.push(file.clone());
            settings["files"] = json!(roots);
            files.insert(wrapper.clone(), settings.to_string());
        }
    }
    let plans =
        crate::project::analyze_project(&files.clone().into_iter().collect(), &wrapper, &cwd)?;
    let original_diagnostics = files
        .get(&file)
        .map(|text| {
            nmath_nts::transform_with_plan(
                &file,
                text,
                nmath_nts::TransformOptions {
                    rewrite_import_extensions: false,
                },
                &plans.get(&file).cloned().unwrap_or_default(),
            )
            .diagnostics
        })
        .unwrap_or_default();
    let fs: Arc<dyn FS> = Arc::new(Snapshot::new(files));
    let host = ConfigHost {
        fs: fs.clone(),
        cwd: cwd.clone(),
    };
    // The only registered mapper is the in-process adapter above. This
    // compiler flag registers its extension; no project command is executed.
    let overrides = tsrs_core::CompilerOptions {
        run_external_code: Tristate::True,
        ..Default::default()
    };
    let (config, errors) = tsrs_tsoptions::get_parsed_command_line_of_config_file(
        &wrapper,
        Some(&overrides),
        None,
        &host,
        None,
    );
    if !errors.is_empty() {
        return Err(errors
            .iter()
            .map(|d| d.localize())
            .collect::<Vec<_>>()
            .join("\n"));
    }
    let mut config = config.ok_or("Cannot read project configuration")?;
    let errors: Vec<_> = config
        .errors
        .iter()
        .filter(|d| !matches!(d.code(), 18002 | 18003))
        .collect();
    if !errors.is_empty() {
        return Err(errors
            .iter()
            .map(|d| d.localize())
            .collect::<Vec<_>>()
            .join("\n"));
    }
    if !config.file_names().contains(&file) {
        config.parsed_config.file_names.push(file.clone());
    }
    let config = P::new(config);
    let compiler_host = new_compiler_host(
        &cwd,
        fs.clone(),
        &bundled::lib_path(),
        None,
        None,
        Some(Arc::new(Mapper { plans })),
    );
    let mut options = ProgramOptions::new(config, compiler_host);
    options.single_threaded = Tristate::True;
    options.checker_regions = true;
    options.checker_diagnostic_regions = true;
    options.regions = Some(Arc::new(Default::default()));
    let program = new_program(options);
    let owner = ProgramOwner(program);
    if program.get_source_file(&file).is_none() {
        return Err("Document is not part of the TypeScript program".into());
    }
    let fs_lines = fs.clone();
    let converters = lsconv::new_converters(proto::PositionEncodingKind::UTF16, move |name| {
        program
            .get_source_file(name)
            .map(|file| lsconv::compute_lsp_line_starts(file.text()))
            .or_else(|| {
                fs_lines
                    .read_file(name)
                    .map(|text| lsconv::compute_lsp_line_starts(&text))
            })
    });
    let ls = tsrs_ls::new_language_service(
        ProjectID(wrapper),
        program,
        Arc::new(ServiceHost { fs, converters }),
        &file,
    );
    let mut capabilities = proto::ResolvedClientCapabilities::default();
    // Request explicit edits internally, then return ordinary TextEdit values
    // that also work with clients without InsertReplaceEdit support.
    capabilities
        .text_document
        .completion
        .completion_item
        .insert_replace_support = true;
    let ctx = proto::with_client_capabilities(&Context::background(), Arc::new(capabilities));
    let query = match query {
        Query::Completion(position) => {
            let Some(position) =
                to_virtual(program, &file, position, tsrs_spanmap::Feature::Completion)
            else {
                return Ok(json!({"isIncomplete":false,"items":[]}));
            };
            Query::Completion(position)
        }
        Query::Definition(position) => {
            let Some(position) =
                to_virtual(program, &file, position, tsrs_spanmap::Feature::Definition)
            else {
                return Ok(Value::Null);
            };
            Query::Definition(position)
        }
        query => query,
    };
    let strict_ranges = !matches!(query, Query::Diagnostics);
    let mut result = match query {
        Query::Diagnostics => ls.provide_diagnostics(&ctx, &uri).map(|r| encode(&r)),
        Query::Completion(position) => ls
            .provide_completion(&ctx, &uri, position, None)
            .map(|r| encode(&r)),
        Query::Definition(position) => ls
            .provide_definition(&ctx, &uri, position)
            .map(|r| encode(&r)),
    }
    .map_err(|error| format!("{error:?}"));
    if let Ok(value) = &mut result {
        map_response(program, &file, value, strict_ranges);
        if !strict_ranges {
            if let Some(items) = value.get_mut("items").and_then(Value::as_array_mut) {
                // tsrs converts external byte offsets using the generated text.
                // Rebuild mapper diagnostics from original source coordinates
                // until its language service supports original external ranges.
                items.retain(|d| d["source"] != "nmath");
                let text = program
                    .get_source_file(&file)
                    .map(|f| f.original_text())
                    .unwrap_or("");
                items.extend(original_diagnostics.iter().map(|d| {
                    json!({
                        "range": super::range(text, d.start, d.start + d.length),
                        "severity": 1, "code": d.code.as_str(), "source": "nmath",
                        "message": format!("{}\n{}", d.message, d.recovery),
                    })
                }));
            }
        }
        if let Some(items) = value.get_mut("items").and_then(Value::as_array_mut) {
            items.retain(|item| {
                item.get("label")
                    .and_then(Value::as_str)
                    .is_none_or(|label| {
                        !label.starts_with("__nmath")
                            || program
                                .get_source_file(&file)
                                .is_some_and(|file| file.original_text().contains(label))
                    })
            });
            for item in items {
                // Resolve is not advertised. Do not expose private virtual
                // offsets in completion data that the client cannot use.
                if let Some(item) = item.as_object_mut() {
                    item.remove("data");
                }
                if let Some(edit) = item.get_mut("textEdit").and_then(Value::as_object_mut)
                    && let Some(range) = edit.remove("replace")
                {
                    edit.remove("insert");
                    edit.insert("range".into(), range);
                }
            }
        }
    }
    drop(ls);
    drop(owner);
    result
}

// At this pin, lsconv::Script for SourceFile deliberately returns no span map.
// Supply virtual line maps to tsrs and adapt the protocol boundary with the
// existing compiler span maps. Remove this bridge when that implementation
// exposes SourceFile::span_map; otherwise locations would be mapped twice.
fn to_virtual(
    program: &tsrs_compiler::Program,
    name: &str,
    position: proto::Position,
    feature: tsrs_spanmap::Feature,
) -> Option<proto::Position> {
    let file = program.get_source_file(name)?;
    let Some(map) = file.span_map() else {
        return Some(position);
    };
    let offset = nmath_nts::offset_of_line_character(
        file.original_text(),
        position.line,
        position.character,
    );
    let mapped = map.original_to_virtual_positions(offset, feature);
    let mapped = mapped.iter().find(|m| m.fidelity.is_exact())?;
    let (line, column) = nmath_nts::line_column(file.text(), mapped.position);
    Some(proto::Position {
        line: line - 1,
        character: column - 1,
    })
}

fn map_response(
    program: &tsrs_compiler::Program,
    name: &str,
    value: &mut Value,
    strict: bool,
) -> bool {
    match value {
        Value::Array(values) => {
            values.retain_mut(|value| map_response(program, name, value, strict));
        }
        Value::Object(object) => {
            // Mapper-authored diagnostics already use original coordinates.
            if object.get("source").and_then(Value::as_str) == Some("nmath") {
                if let Some(code) = object
                    .get("code")
                    .and_then(Value::as_i64)
                    .and_then(|code| nmath_nts::DiagnosticCode::from_mapper_code(code as i32))
                {
                    object.insert("code".into(), json!(code.as_str()));
                    if let Some(message) = object.get("message").and_then(Value::as_str) {
                        object.insert(
                            "message".into(),
                            json!(format!("{message}\n{}", code.recovery())),
                        );
                    }
                }
                return true;
            }
            let target = object
                .get("uri")
                .or_else(|| object.get("targetUri"))
                .and_then(Value::as_str)
                .map(|uri| proto::DocumentUri(uri.into()).file_name())
                .unwrap_or_else(|| name.into());
            for (key, value) in object {
                let file_name = if key == "originSelectionRange" {
                    name
                } else {
                    &target
                };
                if matches!(
                    key.as_str(),
                    "range"
                        | "insert"
                        | "replace"
                        | "targetRange"
                        | "targetSelectionRange"
                        | "originSelectionRange"
                ) {
                    if let Some(file) = program.get_source_file(file_name)
                        && !map_range(file, value, strict)
                    {
                        return false;
                    }
                } else if !map_response(program, &target, value, strict) {
                    return false;
                }
            }
        }
        _ => {}
    }
    true
}

fn map_range(file: P<tsrs_ast::SourceFile>, value: &mut Value, strict: bool) -> bool {
    let Some(map) = file.span_map() else {
        return true;
    };
    let offset = |position: &Value| {
        nmath_nts::offset_of_line_character(
            file.text(),
            position["line"].as_u64().unwrap_or(0) as u32,
            position["character"].as_u64().unwrap_or(0) as u32,
        )
    };
    let span = TextRange::new(offset(&value["start"]), offset(&value["end"]));
    let (mapped, fidelity) = map.virtual_to_original_span(span);
    if strict && !fidelity.is_exact() {
        return false;
    }
    // Generated imports have no original span. Anchor their diagnostics at
    // the start of the document rather than exposing virtual line numbers.
    let (start, end) = if fidelity.is_none() {
        (0, 0)
    } else {
        (
            mapped.pos().source_offset().unwrap_or(0),
            mapped.end().source_offset().unwrap_or(0),
        )
    };
    *value = super::range(file.original_text(), start, end);
    true
}

fn encode(value: &impl Json) -> Value {
    serde_json::from_str(&tsrs_core::json::marshal(&value.to_json()).expect("LSP JSON"))
        .expect("LSP JSON")
}

struct ProgramOwner(&'static tsrs_compiler::Program);
impl Drop for ProgramOwner {
    fn drop(&mut self) {
        // SAFETY: this request creates one unshared program. The language
        // service and all checker handles are dropped before this guard.
        unsafe { tsrs_compiler::free_unshared_program(self.0) };
    }
}

struct ConfigHost {
    fs: Arc<dyn FS>,
    cwd: String,
}
impl ParseConfigHost for ConfigHost {
    fn fs(&self) -> &dyn FS {
        &*self.fs
    }

    fn get_current_directory(&self) -> &str {
        &self.cwd
    }
}

struct ServiceHost {
    fs: Arc<dyn FS>,
    converters: Arc<Converters>,
}
impl Host for ServiceHost {
    fn use_case_sensitive_file_names(&self) -> bool {
        self.fs.use_case_sensitive_file_names()
    }

    fn read_file(&self, path: &str) -> Option<String> {
        self.fs.read_file(path)
    }

    fn converters(&self) -> Arc<Converters> {
        self.converters.clone()
    }

    fn get_preferences(&self, _: &str) -> UserPreferences {
        let mut preferences = new_default_user_preferences();
        // Auto-import discovery needs a persistent registry. Existing imports
        // and their typed members still use the full semantic completion API.
        preferences.include_completions_for_module_exports = Tristate::False;
        preferences
    }

    fn get_ecma_line_info(&self, _: &str) -> Option<Arc<ECMALineInfo>> {
        None
    }

    fn auto_import_registry(&self) -> Option<Arc<Registry>> {
        None
    }

    fn read_directory(
        &self,
        cwd: &str,
        path: &str,
        extensions: &[String],
        excludes: &[String],
        includes: &[String],
        depth: usize,
    ) -> Vec<String> {
        vfsmatch::read_directory(&*self.fs, cwd, path, extensions, excludes, includes, depth)
    }

    fn get_directories(&self, path: &str) -> Vec<String> {
        self.fs.get_accessible_entries(path).directories
    }

    fn directory_exists(&self, path: &str) -> bool {
        self.fs.directory_exists(path)
    }

    fn file_exists(&self, path: &str) -> bool {
        self.fs.file_exists(path)
    }
}

struct Mapper {
    plans: BTreeMap<String, nmath_nts::semantic::LoweringPlan>,
}
impl mapper::Project for Mapper {
    fn refresh(&self) -> Result<(), mapper::Error> {
        Ok(())
    }

    fn identities(&self) -> Result<Vec<String>, mapper::Error> {
        Ok(vec!["nmath".into()])
    }

    fn identity(&self, _: &mapper::Mapper) -> Result<String, mapper::Error> {
        Ok("nmath".into())
    }

    fn watched_files(&self) -> Result<Vec<String>, mapper::Error> {
        Ok(vec![])
    }

    fn diagnostics(&self) -> Vec<mapper::OptionDiagnostic> {
        vec![]
    }

    fn close(&self) -> Result<(), mapper::Error> {
        Ok(())
    }

    fn transform(
        &self,
        _: &mapper::Mapper,
        request: mapper::Request<'_>,
    ) -> Result<mapper::TransformResultFiles, mapper::Error> {
        let plan = self
            .plans
            .get(request.file_name)
            .cloned()
            .unwrap_or_default();
        let output = nmath_nts::transform_with_plan(
            request.file_name,
            request.content,
            nmath_nts::TransformOptions {
                rewrite_import_extensions: false,
            },
            &plan,
        );
        let mappings =
            tsrs_spanmap::unmarshal(&serde_json::to_vec(&output.span_map.to_tuples()).unwrap())
                .map_err(mapper::Error::Other)?;
        let diagnostics = output
            .diagnostics
            .iter()
            .map(|d| {
                tsrs_ast::new_external_diagnostic(
                    None,
                    TextRange::new(d.start, d.start + d.length),
                    "nmath",
                    tsrs_diagnostics::Category::Error,
                    d.code.mapper_code(),
                    &d.message,
                )
            })
            .collect();
        Ok(mapper::TransformResultFiles {
            text: output.text,
            virtual_extension: ".ts".into(),
            mappings: Some(mappings),
            diagnostics,
            ..Default::default()
        })
    }
}
