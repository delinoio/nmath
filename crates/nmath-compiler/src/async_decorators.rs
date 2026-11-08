//! Repair the pinned emitter's modern field-decorator await placement.
//!
//! tsrs emits modern decorator evaluation inside a class static block, where
//! JavaScript forbids `await`. A computed method name runs after heritage and
//! preceding computed names but before decorator application and static field
//! initialization. Move only the affected decorator evaluation to that phase,
//! make the generated class wrapper async, and remove the temporary symbol
//! method before decorators run. Remove this pass when the pinned emitter can
//! suspend modern class evaluation while retaining that phase order.

#[derive(Clone, Debug)]
pub(crate) struct CopySpan {
    pub old_start: usize,
    pub old_end: usize,
    pub new_start: usize,
}

pub(crate) struct Rewrite {
    pub old: String,
    pub new: String,
    pub copies: Vec<CopySpan>,
}

enum Part {
    New(String),
    Copy(usize, usize),
}

struct Edit {
    start: usize,
    end: usize,
    parts: Vec<Part>,
}

fn line_start(text: &str, at: usize) -> usize {
    text[..at].rfind('\n').map_or(0, |index| index + 1)
}

fn line_end(text: &str, at: usize) -> usize {
    text[at..]
        .find('\n')
        .map_or(text.len(), |index| at + index + 1)
}

fn matching_brace(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut quote = None;
    let mut escaped = false;
    let mut line_comment = false;
    let mut block_comment = false;
    let bytes = text.as_bytes();
    let mut index = open;
    while index < bytes.len() {
        let byte = bytes[index];
        let next = bytes.get(index + 1).copied();
        if line_comment {
            if byte == b'\n' {
                line_comment = false;
            }
        } else if block_comment {
            if byte == b'*' && next == Some(b'/') {
                block_comment = false;
                index += 1;
            }
        } else if let Some(delimiter) = quote {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == delimiter {
                quote = None;
            }
        } else if byte == b'/' && next == Some(b'/') {
            line_comment = true;
            index += 1;
        } else if byte == b'/' && next == Some(b'*') {
            block_comment = true;
            index += 1;
        } else if matches!(byte, b'\'' | b'"' | b'`') {
            quote = Some(byte);
        } else if byte == b'{' {
            depth += 1;
        } else if byte == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some(index);
            }
        }
        index += 1;
    }
    None
}

fn apply(text: &str, mut edits: Vec<Edit>) -> Result<Rewrite, String> {
    edits.sort_by_key(|edit| (edit.start, edit.end));
    let mut output = String::new();
    let mut copies = Vec::new();
    let mut cursor = 0;
    let append_copy =
        |start: usize, end: usize, output: &mut String, copies: &mut Vec<CopySpan>| {
            if start < end {
                let new_start = output.len();
                output.push_str(&text[start..end]);
                copies.push(CopySpan {
                    old_start: start,
                    old_end: end,
                    new_start,
                });
            }
        };
    for edit in edits {
        if edit.start < cursor || edit.end > text.len() {
            return Err("Overlapping modern decorator emit edits".into());
        }
        append_copy(cursor, edit.start, &mut output, &mut copies);
        for part in edit.parts {
            match part {
                Part::New(value) => output.push_str(&value),
                Part::Copy(start, end) => append_copy(start, end, &mut output, &mut copies),
            }
        }
        cursor = edit.end;
    }
    append_copy(cursor, text.len(), &mut output, &mut copies);
    Ok(Rewrite {
        old: text.to_string(),
        new: output,
        copies,
    })
}

fn find_await_set(text: &str, from: usize) -> Option<usize> {
    let mut cursor = from;
    while let Some(offset) = text[cursor..].find("await __nmath") {
        let at = cursor + offset;
        let after_name = at + "await __nmath".len();
        let mut suffix = &text[after_name..];
        if let Some(rest) = suffix.strip_prefix('_') {
            let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
            suffix = if digits > 0 { &rest[digits..] } else { "" };
        }
        if suffix.starts_with(".set(") {
            return Some(at);
        }
        cursor = after_name;
    }
    None
}

pub(crate) fn rewrite(text: &str) -> Result<Option<Rewrite>, String> {
    if !text.contains("__esDecorate(") {
        return Ok(None);
    }
    let mut cursor = 0;
    let (await_at, class_at, class_open, class_close) = loop {
        let Some(at) = find_await_set(text, cursor) else {
            return Ok(None);
        };
        let return_class = text[..at].rfind("return class ");
        let assigned_class = text[..at].rfind("= class ");
        let Some(class_token) = return_class.into_iter().chain(assigned_class).max() else {
            cursor = at + "await __nmath".len();
            continue;
        };
        let class_at = line_start(text, class_token)
            + text[line_start(text, class_token)..class_token]
                .bytes()
                .take_while(|byte| *byte == b' ' || *byte == b'\t')
                .count();
        let class_open = text[class_token..]
            .find('{')
            .map(|offset| class_token + offset)
            .ok_or("Generated class body is missing")?;
        let class_close =
            matching_brace(text, class_open).ok_or("Generated class body is unbalanced")?;
        if at > class_close {
            cursor = at + "await __nmath".len();
            continue;
        }
        let prefix = &text[class_open..at];
        let Some(marker) = prefix.rfind("_decorators = [") else {
            cursor = at + "await __nmath".len();
            continue;
        };
        let end = class_open + marker + "_decorators".len();
        let start = text[..end]
            .rfind(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
            .map_or(0, |index| index + 1);
        let decorator_var = &text[start..end];
        let field_call = format!("{decorator_var}, {{ kind: \"field\"");
        if !text[class_open..class_close].contains(&field_call) {
            cursor = at + "await __nmath".len();
            continue;
        }
        let sync = text[..class_at].rfind("(() => {");
        let asynchronous = text[..class_at].rfind("(async () => {");
        if asynchronous.is_some_and(|async_at| sync.is_none_or(|sync_at| async_at > sync_at)) {
            cursor = at + "await __nmath".len();
            continue;
        }
        break (at, class_at, class_open, class_close);
    };
    let wrapper_at = text[..class_at]
        .rfind("(() => {")
        .ok_or("Generated class wrapper is missing")?;
    let wrapper_open = wrapper_at + "(() => ".len();
    let wrapper_close =
        matching_brace(text, wrapper_open).ok_or("Generated class wrapper is unbalanced")?;
    if wrapper_close < class_close {
        return Err("Generated class is outside its wrapper".into());
    }
    let mut edits = Vec::new();
    let mut found_decorator_block = false;
    let mut static_cursor = class_open;
    while let Some(static_at) = text[static_cursor..class_close]
        .find("static {")
        .map(|offset| static_cursor + offset)
    {
        let static_open = static_at + "static ".len();
        let static_close =
            matching_brace(text, static_open).ok_or("Generated decorator block is unbalanced")?;
        if await_at > static_open && await_at < static_close {
            let metadata_at = text[static_open..static_close]
                .find("const _metadata = ")
                .map(|offset| static_open + offset)
                .ok_or("Generated decorator metadata is missing")?;
            let metadata_start = line_start(text, metadata_at);
            let metadata_end = line_end(text, metadata_at);
            let metadata_value_start = metadata_at + "const ".len();
            let metadata_value_end = text[metadata_value_start..metadata_end]
                .rfind(';')
                .map(|offset| metadata_value_start + offset)
                .ok_or("Generated metadata terminator is missing")?;
            let mut assignments = Vec::new();
            let mut position = metadata_end;
            while position < static_close {
                let end = line_end(text, position).min(static_close);
                let line = text[position..end].trim();
                if line.contains("_decorators = [") && !line.contains("__esDecorate(") {
                    let trimmed_start = position + text[position..end].find(line).unwrap();
                    let trimmed_end = trimmed_start + line.len() - usize::from(line.ends_with(';'));
                    assignments.push((
                        position,
                        end,
                        trimmed_start,
                        trimmed_end,
                        find_await_set(line, 0).is_some(),
                    ));
                }
                position = end;
            }
            let Some(last) = assignments.iter().rposition(|assignment| assignment.4) else {
                return Ok(None);
            };
            found_decorator_block = true;
            let sentinel = (0..)
                .map(|index| format!("__nmath_decorator_sentinel_{index}"))
                .find(|candidate| !text.contains(candidate))
                .unwrap();
            edits.push(Edit {
                start: class_at,
                end: class_at,
                parts: vec![Part::New(format!(
                    "let _metadata; let {sentinel};\n        "
                ))],
            });
            edits.push(Edit {
                start: metadata_start,
                end: metadata_end,
                parts: vec![],
            });
            let mut parts = vec![
                Part::New("\n        [(".into()),
                Part::Copy(metadata_value_start, metadata_value_end),
            ];
            for &(start, end, value_start, value_end, _) in &assignments[..=last] {
                edits.push(Edit {
                    start,
                    end,
                    parts: vec![],
                });
                parts.push(Part::New(", ".into()));
                parts.push(Part::Copy(value_start, value_end));
            }
            parts.push(Part::New(format!(
                ", {sentinel} = Symbol())]() {{}}\n        "
            )));
            edits.push(Edit {
                start: class_close,
                end: class_close,
                parts,
            });
            edits.push(Edit {
                start: metadata_end,
                end: metadata_end,
                parts: vec![Part::New(format!(
                    "                delete this.prototype[{sentinel}];\n"
                ))],
            });
            break;
        }
        static_cursor = static_close + 1;
    }
    if !found_decorator_block {
        return Ok(None);
    }
    edits.push(Edit {
        start: wrapper_at,
        end: wrapper_at + "(() => {".len(),
        parts: vec![Part::New("await (async () => {".into())],
    });
    apply(text, edits).map(Some)
}
