//! `Content-Length` framed JSON-RPC 2.0 messages, shared by the content mapper
//! and the language server.

use std::io::{self, BufRead, Write};

use serde_json::{Value, json};

/// JSON-RPC error codes used by the servers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RpcErrorCode {
    InvalidParams,
    MethodNotFound,
    InvalidRequest,
}

impl RpcErrorCode {
    pub fn code(self) -> i64 {
        match self {
            RpcErrorCode::InvalidRequest => -32600,
            RpcErrorCode::MethodNotFound => -32601,
            RpcErrorCode::InvalidParams => -32602,
        }
    }
}

/// Reads one framed message. `Ok(None)` is a clean end of input.
pub fn read_message(reader: &mut impl BufRead) -> io::Result<Option<Value>> {
    let mut content_length: Option<usize> = None;
    let mut saw_header = false;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            if saw_header {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "end of input inside message headers",
                ));
            }
            return Ok(None);
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            if !saw_header {
                continue;
            }
            break;
        }
        saw_header = true;
        if let Some((name, value)) = line.split_once(':')
            && name.trim().eq_ignore_ascii_case("content-length")
        {
            content_length = value.trim().parse().ok();
        }
    }
    let length = content_length.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "message without a valid Content-Length header",
        )
    })?;
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    serde_json::from_slice(&body)
        .map(Some)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}

/// Writes one framed message and flushes.
pub fn write_message(writer: &mut impl Write, message: &Value) -> io::Result<()> {
    let body = serde_json::to_vec(message)?;
    write!(writer, "Content-Length: {}\r\n\r\n", body.len())?;
    writer.write_all(&body)?;
    writer.flush()
}

pub fn response(id: &Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

pub fn error_response(id: &Value, code: RpcErrorCode, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code.code(), "message": message } })
}

pub fn notification(method: &str, params: Value) -> Value {
    json!({ "jsonrpc": "2.0", "method": method, "params": params })
}

pub fn request(id: i64, method: &str, params: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
}
