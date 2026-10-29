//! Generic LSP over stdio, backed by the pinned tsrs semantic language service.

mod semantic;
mod snapshot;

use std::{
    collections::HashMap,
    io::{self, BufRead, Write},
};

use serde_json::{Value, json};

use crate::framing::{
    RpcErrorCode, error_response, notification, read_message, response, write_message,
};

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
    versions: HashMap<String, i64>,
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
                        "positionEncoding": "utf-16",
                        "completionProvider": { "triggerCharacters": ["."] },
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
                if tsrs_lsproto::DocumentUri(uri.clone()).check().is_err() {
                    return Vec::new();
                }
                let version = params
                    .pointer("/textDocument/version")
                    .and_then(Value::as_i64)
                    .unwrap_or(0);
                if self.versions.get(&uri).is_some_and(|old| version <= *old) {
                    return Vec::new();
                }
                self.versions.insert(uri.clone(), version);
                self.documents.insert(uri.clone(), text.to_string());
                self.publish_all()
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
                if tsrs_lsproto::DocumentUri(uri.clone()).check().is_err() {
                    return Vec::new();
                }
                let version = params
                    .pointer("/textDocument/version")
                    .and_then(Value::as_i64)
                    .unwrap_or(0);
                if self.versions.get(&uri).is_some_and(|old| version <= *old) {
                    return Vec::new();
                }
                self.versions.insert(uri.clone(), version);
                self.documents.insert(uri.clone(), text.to_string());
                self.publish_all()
            }
            Method::DidClose => {
                let Some(uri) = uri else {
                    return Vec::new();
                };
                self.documents.remove(&uri);
                self.versions.remove(&uri);
                let mut messages = vec![notification(
                    "textDocument/publishDiagnostics",
                    json!({ "uri": uri, "diagnostics": [] }),
                )];
                messages.extend(self.publish_all());
                messages
            }
            Method::Completion | Method::Definition => {
                let Some(uri) = uri else {
                    return reply(id, Value::Null);
                };
                let position = tsrs_lsproto::Position {
                    line: params
                        .pointer("/position/line")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as u32,
                    character: params
                        .pointer("/position/character")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as u32,
                };
                let query = if method == Method::Completion {
                    semantic::Query::Completion(position)
                } else {
                    semantic::Query::Definition(position)
                };
                match semantic::query(&self.documents, &uri, query) {
                    Ok(result) => reply(id, result),
                    Err(message) => id
                        .map(|id| vec![error_response(id, RpcErrorCode::InvalidRequest, &message)])
                        .unwrap_or_default(),
                }
            }
        }
    }

    fn publish_all(&self) -> Vec<Value> {
        let mut uris: Vec<_> = self.documents.keys().collect();
        uris.sort();
        uris.into_iter()
            .map(|uri| self.publish_diagnostics(uri))
            .collect()
    }

    fn publish_diagnostics(&self, uri: &str) -> Value {
        let text = self
            .documents
            .get(uri)
            .map(String::as_str)
            .unwrap_or_default();
        let mut diagnostics: Vec<Value> = Vec::new();
        match semantic::query(&self.documents, uri, semantic::Query::Diagnostics) {
            Ok(report) => {
                if let Some(items) = report.get("items").and_then(Value::as_array) {
                    diagnostics.extend(items.iter().cloned());
                }
            }
            Err(message) => diagnostics.push(json!({
                "range": range(text, 0, 0), "severity": 1,
                "code": "NMATH_COMPILER_CONFIG", "source": "nmath", "message": message,
            })),
        }
        notification(
            "textDocument/publishDiagnostics",
            json!({ "uri": uri, "version": self.versions.get(uri), "diagnostics": diagnostics }),
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
