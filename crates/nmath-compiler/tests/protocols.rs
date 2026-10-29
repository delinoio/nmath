//! Content-mapper and language-server protocol round trips against the built
//! executable.

use std::{
    io::{BufReader, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};

use nmath_compiler::framing::{notification, read_message, request, write_message};
use serde_json::{Value, json};

mod common;

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

    let fixture = common::Fixture::new("lsp");
    let uri = format!("file://{}", fixture.path("src/main.nts").display());
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
        json!({ "textDocument": { "uri": uri, "version": 2 }, "contentChanges": [{ "text": "export const x = 1;\n" }] }),
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
    for helper in ["x"] {
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
        definition["result"][0]["range"]["start"],
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

fn open_document(lsp: &mut Session, uri: &str, text: &str) {
    lsp.send(&notification(
        "textDocument/didOpen",
        json!({
            "textDocument": {"uri":uri,"languageId":"nts","version":1,"text":text}
        }),
    ));
}

fn change_document(lsp: &mut Session, uri: &str, version: i64, text: &str) {
    lsp.send(&notification(
        "textDocument/didChange",
        json!({
            "textDocument": {"uri":uri,"version":version},"contentChanges":[{"text":text}]
        }),
    ));
}

fn at(text: &str, needle: &str) -> Value {
    let offset = text.find(needle).unwrap() as u32;
    let (line, column) = nmath_nts::line_column(text, offset);
    json!({"line":line-1,"character":column-1})
}

#[test]
fn semantic_lsp_maps_generic_errors_imports_and_completion_edits() {
    let fixture = common::Fixture::new("semantic-lsp");
    let library = "// 😀 library\nexport function identity<T>(value: T): T { return value; \
                   }\nexport const point = { horizontal: 1, vertical: 2 };\n";
    fixture.write("src/library.nts", library);
    fixture.write("src/main.nts", "export const saved = true;\n");
    let uri = format!("file://{}", fixture.path("src/main.nts").display());
    let library_uri = format!("file://{}", fixture.path("src/library.nts").display());
    let bad = "import { identity as copy, point } from './library.nts';\nexport const sum = \
               copy(1) + 2; /* 😀 */ export const wrong: number = copy('bad');\n/* 😀 */ \
               point.horizontal;\n";
    let mut lsp = Session::start("lsp");
    lsp.call(1, "initialize", json!({"capabilities":{}}));
    open_document(&mut lsp, &uri, bad);
    let diagnostics = lsp.receive();
    let items = diagnostics["params"]["diagnostics"].as_array().unwrap();
    let error = items
        .iter()
        .find(|d| d["code"] == 2322)
        .expect("generic return type is checked");
    assert_eq!(error["range"]["start"], at(bad, "wrong:"), "{diagnostics}");
    assert_eq!(diagnostics["params"]["version"], 1);

    let definition = lsp.call(
        2,
        "textDocument/definition",
        json!({
            "textDocument":{"uri":uri}, "position":at(bad, "copy('bad')")
        }),
    );
    assert_eq!(definition["result"][0]["uri"], library_uri, "{definition}");
    assert_eq!(
        definition["result"][0]["range"]["start"],
        at(library, "identity<T>"),
        "{definition}"
    );

    let good = bad.replace("copy('bad')", "copy(3)");
    change_document(&mut lsp, &uri, 2, &good);
    assert_eq!(lsp.receive()["params"]["diagnostics"], json!([]));
    let completion = lsp.call(
        3,
        "textDocument/completion",
        json!({
            "textDocument":{"uri":uri},"position":at(&good, "horizontal;")
        }),
    );
    let items = completion["result"]["items"]
        .as_array()
        .expect("semantic completion list");
    let horizontal = items
        .iter()
        .find(|i| i["label"] == "horizontal")
        .expect("imported object member");
    assert!(
        items.iter().any(|i| i["label"] == "vertical"),
        "{completion}"
    );
    assert!(
        !items.iter().any(|i| i["label"] == "slice"),
        "no canned helpers"
    );
    let edit = &horizontal["textEdit"];
    assert_eq!(
        edit["range"]["start"],
        at(&good, "horizontal;"),
        "{completion}"
    );

    let changed_library = library.replace("horizontal: 1, vertical: 2", "depth: 3");
    open_document(&mut lsp, &library_uri, &changed_library);
    lsp.receive();
    lsp.receive();
    let changed_completion = lsp.call(
        30,
        "textDocument/completion",
        json!({
            "textDocument":{"uri":uri},"position":at(&good, "horizontal;")
        }),
    );
    let changed_items = changed_completion["result"]["items"].as_array().unwrap();
    assert!(
        changed_items.iter().any(|i| i["label"] == "depth"),
        "{changed_completion}"
    );
    assert!(
        !changed_items.iter().any(|i| i["label"] == "horizontal"),
        "{changed_completion}"
    );

    // The source file on disk was never replaced by its open buffer.
    assert_eq!(fixture.read("src/main.nts"), "export const saved = true;\n");
    lsp.call(4, "shutdown", Value::Null);
    lsp.send(&notification("exit", Value::Null));
    assert_eq!(lsp.finish(), 0);
}

#[test]
fn semantic_lsp_refreshes_dependents_and_restores_disk_after_close() {
    let fixture = common::Fixture::new("semantic-lsp-edits");
    let disk = "export const value = 1;\n";
    fixture.write("src/library.nts", disk);
    let uri = format!("file://{}", fixture.path("src/main.nts").display());
    let library_uri = format!("file://{}", fixture.path("src/library.nts").display());
    let text = "import { value } from './library.nts';\nexport const result: number = value;\n";
    let mut lsp = Session::start("lsp");
    lsp.call(1, "initialize", json!({"capabilities":{}}));
    open_document(&mut lsp, &uri, text);
    assert_eq!(lsp.receive()["params"]["diagnostics"], json!([]));
    let edited = "/* 😀 unsaved */ export const value = 'changed';\n";
    open_document(&mut lsp, &library_uri, edited);
    let reports = [lsp.receive(), lsp.receive()];
    let dependent = reports.iter().find(|m| m["params"]["uri"] == uri).unwrap();
    assert!(
        dependent["params"]["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["code"] == 2322),
        "{dependent}"
    );
    let definition = lsp.call(
        2,
        "textDocument/definition",
        json!({
            "textDocument":{"uri":uri},"position":at(text,"value;\n")
        }),
    );
    assert_eq!(
        definition["result"][0]["range"]["start"],
        at(edited, "value ="),
        "{definition}"
    );
    change_document(&mut lsp, &library_uri, 2, disk);
    for _ in 0..2 {
        assert_eq!(lsp.receive()["params"]["diagnostics"], json!([]));
    }
    // An out-of-order edit must not replace the latest open buffer.
    change_document(&mut lsp, &library_uri, 1, edited);
    let definition = lsp.call(
        3,
        "textDocument/definition",
        json!({
            "textDocument":{"uri":uri},"position":at(text,"value;\n")
        }),
    );
    assert_eq!(
        definition["result"][0]["range"]["start"],
        at(disk, "value =")
    );
    change_document(&mut lsp, &library_uri, 3, edited);
    lsp.receive();
    lsp.receive();
    lsp.send(&notification(
        "textDocument/didClose",
        json!({"textDocument":{"uri":library_uri}}),
    ));
    assert_eq!(lsp.receive()["params"]["diagnostics"], json!([]));
    assert_eq!(lsp.receive()["params"]["diagnostics"], json!([]));
    assert_eq!(fixture.read("src/library.nts"), disk);
    lsp.call(4, "shutdown", Value::Null);
    lsp.send(&notification("exit", Value::Null));
    assert_eq!(lsp.finish(), 0);
}

#[test]
fn semantic_lsp_resolves_lexical_scope_and_plain_typescript_imports() {
    let fixture = common::Fixture::new("semantic-lsp-scope");
    let declaration =
        "export interface Point { x: number; y: number }\nexport declare const point: Point;\n";
    fixture.write("src/types.d.ts", declaration);
    let uri = format!("file://{}", fixture.path("src/main.nts").display());
    let text = "import { point } from './types.js';\nconst value = 1;\nexport function nested() { \
                const value = point; return value.x; }\nexport const outer = value;\n";
    let mut lsp = Session::start("lsp");
    lsp.call(1, "initialize", json!({"capabilities":{}}));
    open_document(&mut lsp, &uri, text);
    assert_eq!(lsp.receive()["params"]["diagnostics"], json!([]));
    let nested = lsp.call(
        2,
        "textDocument/definition",
        json!({
            "textDocument":{"uri":uri},"position":at(text,"value.x")
        }),
    );
    assert_eq!(
        nested["result"][0]["range"]["start"],
        at(text, "value = point"),
        "{nested}"
    );
    let outer = lsp.call(
        3,
        "textDocument/definition",
        json!({
            "textDocument":{"uri":uri},"position":at(text,"value;\n")
        }),
    );
    assert_eq!(
        outer["result"][0]["range"]["start"],
        at(text, "value = 1"),
        "{outer}"
    );
    let property = lsp.call(
        4,
        "textDocument/definition",
        json!({
            "textDocument":{"uri":uri},"position":at(text,"x; }")
        }),
    );
    assert_eq!(
        property["result"][0]["uri"],
        format!("file://{}", fixture.path("src/types.d.ts").display()),
        "{property}"
    );
    assert_eq!(
        property["result"][0]["range"]["start"],
        at(declaration, "x: number")
    );
    lsp.call(5, "shutdown", Value::Null);
    lsp.send(&notification("exit", Value::Null));
    assert_eq!(lsp.finish(), 0);
}
