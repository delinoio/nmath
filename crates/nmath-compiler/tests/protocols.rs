//! Content-mapper and language-server protocol round trips against the built
//! executable.

use std::{
    io::{BufReader, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};

use nmath_compiler::framing::{notification, read_message, request, write_message};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_nmath-compiler");

struct Session {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl Session {
    fn start(subcommand: &str) -> Session {
        let mut child = Command::new(BIN)
            .arg(subcommand)
            .env_remove("NMATH_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("nmath-compiler starts");
        let stdin = child.stdin.take();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Session {
            child,
            stdin,
            stdout,
        }
    }

    fn send(&mut self, message: &Value) {
        write_message(self.stdin.as_mut().unwrap(), message).unwrap();
    }

    fn receive(&mut self) -> Value {
        read_message(&mut self.stdout).unwrap().expect("a message")
    }

    fn call(&mut self, id: i64, method: &str, params: Value) -> Value {
        self.send(&request(id, method, params));
        let reply = self.receive();
        assert_eq!(reply["id"], json!(id), "{reply}");
        reply
    }

    fn finish(mut self) -> i32 {
        drop(self.stdin.take());
        self.child.wait().unwrap().code().unwrap()
    }
}

#[test]
fn content_mapper_round_trip() {
    let mut mapper = Session::start("content-mapper");
    let init = mapper.call(
        1,
        "initialize",
        json!({ "positionEncodings": ["utf-8", "utf-16"] }),
    );
    assert_eq!(
        init["result"],
        json!({ "positionEncoding": "utf-8", "diagnosticSource": "nmath" })
    );

    let open = mapper.call(
        2,
        "openProject",
        json!({ "configFileName": "/p/tsconfig.json", "projectHandle": "1" }),
    );
    assert!(open["result"].is_object());

    // Notifications get no reply.
    mapper.send(&notification("$/progress", json!({})));

    let source = "const y = a * b;\n";
    let transformed = mapper.call(
        3,
        "transform",
        json!({ "fileName": "/p/a.nts", "content": source, "projectHandle": "1" }),
    );
    let result = &transformed["result"];
    assert_eq!(result["extension"], json!(".ts"));
    let text = result["text"].as_str().unwrap();
    assert!(text.contains("const y = __nmath.mul(a, b);"), "{text}");
    // A position inside the operand `b` maps back to the original position.
    let mappings = result["mappings"].as_array().unwrap();
    let virtual_b = text.find("b)").unwrap() as i64;
    let original_b = source.find('b').unwrap() as i64;
    let segment = mappings
        .iter()
        .map(|m| {
            m.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect::<Vec<_>>()
        })
        .find(|m| m[4] == 0 && m[0] <= virtual_b && virtual_b < m[0] + m[1])
        .expect("operand is in a verbatim segment");
    assert_eq!(segment[2] + (virtual_b - segment[0]), original_b);
    assert_eq!(result["diagnostics"], json!([]));

    // Relative `.nts` imports stay for the compiler to resolve.
    let imports = mapper.call(
        4,
        "transform",
        json!({ "fileName": "/p/b.nts", "content": "import { y } from './a.nts';\n" }),
    );
    assert!(
        imports["result"]["text"]
            .as_str()
            .unwrap()
            .contains("'./a.nts'")
    );

    let bad = mapper.call(
        5,
        "transform",
        json!({ "fileName": "/p/c.nts", "content": "function f() { a[i] = 1; }\n" }),
    );
    let diagnostic = &bad["result"]["diagnostics"][0];
    assert_eq!(diagnostic["code"], json!(2));
    assert_eq!(diagnostic["start"], json!(15));

    let unknown = mapper.call(6, "frobnicate", json!({}));
    assert_eq!(unknown["error"]["code"], json!(-32601));

    let close = mapper.call(7, "closeProject", json!({ "projectHandle": "1" }));
    assert_eq!(close["result"], Value::Null);
    assert_eq!(mapper.finish(), 0);
}

#[test]
fn language_server_round_trip() {
    let mut lsp = Session::start("lsp");
    let init = lsp.call(1, "initialize", json!({ "capabilities": {} }));
    let capabilities = &init["result"]["capabilities"];
    assert_eq!(capabilities["textDocumentSync"], json!(1));
    assert_eq!(capabilities["definitionProvider"], json!(true));
    lsp.send(&notification("initialized", json!({})));

    let uri = "file:///p/main.nts";
    let text = "function scale(a: number[]) {\n  a[0] = 1;\n}\nscale([1]);\n";
    lsp.send(&notification(
        "textDocument/didOpen",
        json!({ "textDocument": { "uri": uri, "languageId": "nts", "version": 1, "text": text } }),
    ));
    let published = lsp.receive();
    assert_eq!(
        published["method"],
        json!("textDocument/publishDiagnostics")
    );
    let diagnostic = &published["params"]["diagnostics"][0];
    assert_eq!(diagnostic["code"], json!("NMATH_NTS_ASSIGN_REQUIRES_ASYNC"));
    assert_eq!(
        diagnostic["range"]["start"],
        json!({ "line": 1, "character": 2 })
    );
    assert_eq!(
        diagnostic["range"]["end"],
        json!({ "line": 1, "character": 10 })
    );

    lsp.send(&notification(
        "textDocument/didChange",
        json!({ "textDocument": { "uri": uri, "version": 2 }, "contentChanges": [{ "text": "export const x = a + b;\n" }] }),
    ));
    let republished = lsp.receive();
    assert_eq!(republished["params"]["diagnostics"], json!([]));

    let completion = lsp.call(
        2,
        "textDocument/completion",
        json!({ "textDocument": { "uri": uri }, "position": { "line": 0, "character": 0 } }),
    );
    let labels: Vec<&str> = completion["result"]["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["label"].as_str().unwrap())
        .collect();
    for helper in ["slice", "indices", "mask"] {
        assert!(labels.contains(&helper), "{labels:?}");
    }

    lsp.send(&notification(
        "textDocument/didChange",
        json!({ "textDocument": { "uri": uri, "version": 3 }, "contentChanges": [{ "text": text }] }),
    ));
    lsp.receive();
    let definition = lsp.call(
        3,
        "textDocument/definition",
        json!({ "textDocument": { "uri": uri }, "position": { "line": 3, "character": 2 } }),
    );
    assert_eq!(
        definition["result"]["range"]["start"],
        json!({ "line": 0, "character": 9 })
    );

    let unknown = lsp.call(4, "textDocument/hover", json!({}));
    assert_eq!(unknown["error"]["code"], json!(-32601));

    let shutdown = lsp.call(5, "shutdown", Value::Null);
    assert_eq!(shutdown["result"], Value::Null);
    lsp.send(&notification("exit", Value::Null));
    lsp.stdin.as_mut().unwrap().flush().unwrap();
    assert_eq!(lsp.finish(), 0);
}
