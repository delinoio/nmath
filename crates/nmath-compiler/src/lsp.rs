//! `nmath-compiler lsp`: a minimal Language Server Protocol server over stdio
//! for `.nts` files.
//!
//! Workaround: tsrs at the pinned commit does not run content mappers in its
//! language server (its "phase 2"), so this server runs the NMath transform
//! directly instead of hosting the tsrs language server. It reports transform
//! diagnostics (syntax and async-assignment errors), completes the NMath
//! indexing helpers and resolves definitions of declarations in the same
//! document by name. It does not report type errors; `nmath-compiler check`
//! does. Remove this workaround when tsrs ships language-server content-mapper
//! support at a pinned commit.

use std::{
    collections::HashMap,
    io::{self, BufRead, Write},
};

use serde_json::{Value, json};

use crate::framing::{
    RpcErrorCode, error_response, notification, read_message, response, write_message,
};

/// NMath names offered by completion.
const COMPLETIONS: [(&str, &str); 4] = [
    (
        "slice",
        "slice(start?, stop?, step?): select a strided range along one axis",
    ),
    (
        "indices",
        "indices(list): select explicit positions along one axis",
    ),
    (
        "mask",
        "mask(booleans): select positions where a boolean array is true",
    ),
    (
        "compute",
        "expr.compute(options?): run a lazy expression and await its result",
    ),
];

/// LSP methods this server handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Method {
    Initialize,
    Initialized,
    Shutdown,
    Exit,
    DidOpen,
    DidChange,
    DidClose,
    Completion,
    Definition,
}

impl Method {
    fn parse(name: &str) -> Option<Method> {
        Some(match name {
            "initialize" => Method::Initialize,
            "initialized" => Method::Initialized,
            "shutdown" => Method::Shutdown,
            "exit" => Method::Exit,
            "textDocument/didOpen" => Method::DidOpen,
            "textDocument/didChange" => Method::DidChange,
            "textDocument/didClose" => Method::DidClose,
            "textDocument/completion" => Method::Completion,
            "textDocument/definition" => Method::Definition,
            _ => return None,
        })
    }
}

#[derive(Default)]
struct Server {
    documents: HashMap<String, String>,
    shutdown_requested: bool,
}

/// Serves until `exit` or the end of input. Returns the process exit code (0
/// after `shutdown`, else 1).
pub fn serve(reader: impl io::Read, writer: impl Write) -> io::Result<i32> {
    let mut reader = io::BufReader::new(reader);
    let mut writer = writer;
    serve_buffered(&mut reader, &mut writer)
}

fn serve_buffered(reader: &mut impl BufRead, writer: &mut impl Write) -> io::Result<i32> {
    let mut server = Server::default();
    while let Some(message) = read_message(reader)? {
        let name = message
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let id = message.get("id").cloned();
        let params = message.get("params").cloned().unwrap_or(Value::Null);
        tracing::debug!(method = %name, request = id.is_some(), "lsp message");
        let Some(method) = Method::parse(&name) else {
            if let Some(id) = id {
                write_message(
                    writer,
                    &error_response(
                        &id,
                        RpcErrorCode::MethodNotFound,
                        &format!("Unhandled method: {name}"),
                    ),
                )?;
            }
            continue;
        };
        if method == Method::Exit {
            return Ok(if server.shutdown_requested { 0 } else { 1 });
        }
        for outgoing in server.handle(method, id.as_ref(), &params) {
            write_message(writer, &outgoing)?;
        }
    }
    Ok(if server.shutdown_requested { 0 } else { 1 })
}

impl Server {
    fn handle(&mut self, method: Method, id: Option<&Value>, params: &Value) -> Vec<Value> {
        let uri = params
            .pointer("/textDocument/uri")
            .and_then(Value::as_str)
            .map(str::to_string);
        match method {
            Method::Initialize => reply(
                id,
                json!({
                    "capabilities": {
                        "textDocumentSync": 1,
                        "completionProvider": { "triggerCharacters": [] },
                        "definitionProvider": true,
                    },
                    "serverInfo": { "name": "nmath-compiler", "version": env!("CARGO_PKG_VERSION") },
                }),
            ),
            Method::Initialized | Method::Exit => Vec::new(),
            Method::Shutdown => {
                self.shutdown_requested = true;
                reply(id, Value::Null)
            }
            Method::DidOpen => {
                let (Some(uri), Some(text)) = (
                    uri,
                    params.pointer("/textDocument/text").and_then(Value::as_str),
                ) else {
                    return Vec::new();
                };
                self.documents.insert(uri.clone(), text.to_string());
                vec![self.publish_diagnostics(&uri)]
            }
            Method::DidChange => {
                // Full document sync: the last change carries the whole text.
                let text = params
                    .get("contentChanges")
                    .and_then(Value::as_array)
                    .and_then(|changes| changes.last())
                    .and_then(|change| change.get("text"))
                    .and_then(Value::as_str);
                let (Some(uri), Some(text)) = (uri, text) else {
                    return Vec::new();
                };
                self.documents.insert(uri.clone(), text.to_string());
                vec![self.publish_diagnostics(&uri)]
            }
            Method::DidClose => {
                let Some(uri) = uri else {
                    return Vec::new();
                };
                self.documents.remove(&uri);
                vec![notification(
                    "textDocument/publishDiagnostics",
                    json!({ "uri": uri, "diagnostics": [] }),
                )]
            }
            Method::Completion => {
                let items: Vec<Value> = COMPLETIONS
                    .iter()
                    .map(|(label, detail)| json!({ "label": label, "kind": 3, "detail": detail }))
                    .collect();
                reply(id, json!({ "isIncomplete": false, "items": items }))
            }
            Method::Definition => {
                let result = uri
                    .as_deref()
                    .and_then(|uri| Some((uri, self.documents.get(uri)?)))
                    .and_then(|(uri, text)| definition(uri, text, params.get("position")?))
                    .unwrap_or(Value::Null);
                reply(id, result)
            }
        }
    }

    fn publish_diagnostics(&self, uri: &str) -> Value {
        let text = self
            .documents
            .get(uri)
            .map(String::as_str)
            .unwrap_or_default();
        let output = nmath_nts::transform(uri, text);
        let diagnostics: Vec<Value> = output
            .diagnostics
            .iter()
            .map(|d| {
                json!({
                    "range": range(text, d.start, d.start + d.length),
                    "severity": 1,
                    "code": d.code.as_str(),
                    "source": "nmath",
                    "message": format!("{}\n{}", d.message, d.recovery),
                })
            })
            .collect();
        notification(
            "textDocument/publishDiagnostics",
            json!({ "uri": uri, "diagnostics": diagnostics }),
        )
    }
}

fn reply(id: Option<&Value>, result: Value) -> Vec<Value> {
    id.map(|id| vec![response(id, result)]).unwrap_or_default()
}

/// A 0-based LSP position for a UTF-8 byte offset.
fn position(text: &str, offset: u32) -> Value {
    let (line, column) = nmath_nts::line_column(text, offset);
    json!({ "line": line - 1, "character": column - 1 })
}

fn range(text: &str, start: u32, end: u32) -> Value {
    json!({ "start": position(text, start), "end": position(text, end) })
}

fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'$'
}

/// Best-effort go-to-definition: the identifier under the cursor resolves to a
/// `function`, `class`, `const`, `let` or `var` declaration of the same name in
/// the same document.
fn definition(uri: &str, text: &str, position_param: &Value) -> Option<Value> {
    let line = position_param.get("line")?.as_u64()? as u32;
    let character = position_param.get("character")?.as_u64()? as u32;
    let offset = nmath_nts::offset_of_line_character(text, line, character) as usize;
    let bytes = text.as_bytes();
    let mut start = offset.min(bytes.len());
    while start > 0 && is_identifier_byte(bytes[start - 1]) {
        start -= 1;
    }
    let mut end = offset.min(bytes.len());
    while end < bytes.len() && is_identifier_byte(bytes[end]) {
        end += 1;
    }
    let name = text.get(start..end).filter(|name| !name.is_empty())?;
    for keyword in ["function", "class", "const", "let", "var"] {
        let pattern = format!("{keyword} {name}");
        let mut from = 0;
        while let Some(found) = text[from..].find(&pattern) {
            let at = from + found;
            let name_start = at + keyword.len() + 1;
            let name_end = name_start + name.len();
            let boundary_before = at == 0 || !is_identifier_byte(bytes[at - 1]);
            let boundary_after = name_end >= bytes.len() || !is_identifier_byte(bytes[name_end]);
            if boundary_before && boundary_after {
                return Some(
                    json!({ "uri": uri, "range": range(text, name_start as u32, name_end as u32) }),
                );
            }
            from = at + pattern.len();
        }
    }
    None
}
