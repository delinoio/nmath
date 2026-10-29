//! The internal tsrs content-mapper process (`nmath-compiler content-mapper`).
//!
//! It speaks the tsrs content-mapper protocol (JSON-RPC 2.0 over stdio with
//! `Content-Length` framing; methods `initialize`, `openProject`, `transform`
//! and `closeProject`) and lowers `.nts` files with `nmath-nts`. Positions
//! are UTF-8 byte offsets (`positionEncoding: "utf-8"`).

use std::io::{self, BufRead, Write};

use nmath_nts::{DiagnosticCode, TransformOptions};
use serde_json::{Value, json};

use crate::framing::{RpcErrorCode, error_response, read_message, response, write_message};

/// The diagnostic source prefix for mapper-authored diagnostics (`nmath1`,
/// `nmath2`, ...).
pub const DIAGNOSTIC_SOURCE: &str = "nmath";

/// The protocol methods this mapper implements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Method {
    Initialize,
    OpenProject,
    Transform,
    CloseProject,
}

impl Method {
    fn parse(name: &str) -> Option<Method> {
        Some(match name {
            "initialize" => Method::Initialize,
            "openProject" => Method::OpenProject,
            "transform" => Method::Transform,
            "closeProject" => Method::CloseProject,
            _ => return None,
        })
    }
}

/// Serves the protocol until the input ends.
pub fn serve(reader: impl io::Read, mut writer: impl Write) -> io::Result<()> {
    let mut reader = io::BufReader::new(reader);
    serve_buffered(&mut reader, &mut writer)
}

fn serve_buffered(reader: &mut impl BufRead, writer: &mut impl Write) -> io::Result<()> {
    tracing::debug!("content mapper started");
    let mut projects = std::collections::BTreeMap::<String, Value>::new();
    while let Some(message) = read_message(reader)? {
        let Some(id) = message.get("id").cloned() else {
            // Notifications need no answer.
            continue;
        };
        let name = message
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let reply = match Method::parse(name) {
            Some(method) => handle(
                method,
                message.get("params").unwrap_or(&Value::Null),
                &mut projects,
            )
            .map_or_else(
                |err| error_response(&id, RpcErrorCode::InvalidParams, &err),
                |r| response(&id, r),
            ),
            None => error_response(
                &id,
                RpcErrorCode::MethodNotFound,
                &format!("Unknown method: {name}"),
            ),
        };
        write_message(writer, &reply)?;
    }
    tracing::debug!("content mapper input ended");
    Ok(())
}

fn handle(
    method: Method,
    params: &Value,
    projects: &mut std::collections::BTreeMap<String, Value>,
) -> Result<Value, String> {
    Ok(match method {
        Method::Initialize => {
            json!({ "positionEncoding": "utf-8", "diagnosticSource": DIAGNOSTIC_SOURCE })
        }
        Method::OpenProject => {
            let handle = params
                .get("projectHandle")
                .and_then(Value::as_str)
                .unwrap_or_default();
            projects.insert(
                handle.into(),
                params.get("options").cloned().unwrap_or(Value::Null),
            );
            json!({})
        }
        Method::CloseProject => {
            projects.remove(
                params
                    .get("projectHandle")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            );
            Value::Null
        }
        Method::Transform => {
            let file_name = params
                .get("fileName")
                .and_then(Value::as_str)
                .ok_or("transform requires fileName")?;
            let content = params
                .get("content")
                .and_then(Value::as_str)
                .ok_or("transform requires content")?;
            let handle = params
                .get("projectHandle")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if let Some(transforms) = projects
                .get(handle)
                .and_then(|options| options.get("transforms"))
            {
                let prepared = transforms.get(file_name).ok_or(
                    "File was not included in semantic analysis; rerun the project command",
                )?;
                if prepared.get("source").and_then(Value::as_str) != Some(content) {
                    return Err("Source changed after semantic analysis; rerun the project \
                                command"
                        .into());
                }
                prepared["result"].clone()
            } else {
                transform_result(file_name, content)
            }
        }
    })
}

/// The `transform` result for one `.nts` file.
pub fn transform_result(file_name: &str, content: &str) -> Value {
    // The compiler resolves `.nts` imports itself and rewrites them through the
    // mapper's output extensions on emit, so the virtual text keeps the
    // `.nts` specifiers.
    let output = nmath_nts::transform_with_options(
        file_name,
        content,
        TransformOptions {
            rewrite_import_extensions: false,
        },
    );
    tracing::debug!(
        file = file_name,
        diagnostics = output.diagnostics.len(),
        "transformed .nts file"
    );
    output_result(&output)
}

pub fn output_result(output: &nmath_nts::TransformOutput) -> Value {
    let diagnostics: Vec<Value> = output
        .diagnostics
        .iter()
        .map(|d| {
            json!({
                "messageText": d.message,
                "start": d.start,
                "length": d.length,
                "code": d.code.mapper_code(),
            })
        })
        .collect();
    json!({
        "text": output.text,
        "extension": ".ts",
        "mappings": output.span_map.to_tuples(),
        "diagnostics": diagnostics,
    })
}

/// Maps a mapper-authored diagnostic (`source`, numeric `code`) back to its
/// stable transform code.
pub fn diagnostic_code(source: &str, code: i32) -> Option<DiagnosticCode> {
    (source == DIAGNOSTIC_SOURCE)
        .then(|| DiagnosticCode::from_mapper_code(code))
        .flatten()
}
