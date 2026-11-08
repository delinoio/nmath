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
    pub new: String,
    pub copies: Vec<CopySpan>,
    /// Restart at this wrapper so nested repairs remain visible without
    /// scanning earlier classes.
    pub resume_at: usize,
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

fn is_identifier_char(ch: char) -> bool {
    unicode_ident::is_xid_continue(ch) || ch == '$'
}

fn identifier_start(text: &str, end: usize) -> usize {
    text[..end]
        .char_indices()
        .rev()
        .find_map(|(index, ch)| (!is_identifier_char(ch)).then_some(index + ch.len_utf8()))
        .unwrap_or(0)
}

fn skip_quoted(text: &str, start: usize, delimiter: u8) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut index = start + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index += 2,
            byte if byte == delimiter => return Some(index + 1),
            _ => index += 1,
        }
    }
    None
}

fn skip_regex(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut index = start + 1;
    let mut escaped = false;
    let mut character_class = false;
    while index < bytes.len() {
        let byte = bytes[index];
        if escaped {
            escaped = false;
        } else if byte == b'\\' {
            escaped = true;
        } else if character_class {
            if byte == b']' {
                character_class = false;
            }
        } else if byte == b'[' {
            character_class = true;
        } else if byte == b'/' {
            index += 1;
            while index < bytes.len() && bytes[index].is_ascii_alphabetic() {
                index += 1;
            }
            return Some(index);
        }
        index += 1;
    }
    None
}

fn skip_template_expression(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut index = start;
    let mut braces = 1usize;
    let mut can_start_regex = true;
    while index < bytes.len() {
        let byte = bytes[index];
        let next = bytes.get(index + 1).copied();
        if byte == b'/' && next == Some(b'/') {
            index += 2;
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            continue;
        }
        if byte == b'/' && next == Some(b'*') {
            index += 2;
            while index + 1 < bytes.len() && !(bytes[index] == b'*' && bytes[index + 1] == b'/') {
                index += 1;
            }
            index = (index + 2).min(bytes.len());
            continue;
        }
        if matches!(byte, b'\'' | b'"') {
            index = skip_quoted(text, index, byte)?;
            can_start_regex = false;
            continue;
        }
        if byte == b'`' {
            index = skip_template(text, index)?;
            can_start_regex = false;
            continue;
        }
        if byte == b'/' && can_start_regex {
            index = skip_regex(text, index)?;
            can_start_regex = false;
            continue;
        }
        if byte.is_ascii_alphabetic() || matches!(byte, b'_' | b'$') {
            let token_start = index;
            index += 1;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || matches!(bytes[index], b'_' | b'$'))
            {
                index += 1;
            }
            can_start_regex = matches!(
                &text[token_start..index],
                "await"
                    | "case"
                    | "delete"
                    | "do"
                    | "else"
                    | "in"
                    | "instanceof"
                    | "of"
                    | "return"
                    | "throw"
                    | "typeof"
                    | "void"
                    | "yield"
            );
            continue;
        }
        if byte == b'{' {
            braces += 1;
            can_start_regex = true;
        } else if byte == b'}' {
            braces = braces.checked_sub(1)?;
            if braces == 0 {
                return Some(index + 1);
            }
            can_start_regex = false;
        } else if matches!(byte, b'(' | b'[') {
            can_start_regex = true;
        } else if matches!(byte, b')' | b']' | b'.') {
            can_start_regex = false;
        } else if matches!(
            byte,
            b';' | b','
                | b':'
                | b'?'
                | b'='
                | b'!'
                | b'&'
                | b'|'
                | b'+'
                | b'-'
                | b'*'
                | b'%'
                | b'^'
                | b'~'
                | b'<'
                | b'>'
        ) {
            can_start_regex = true;
        }
        index += 1;
    }
    None
}

fn skip_template(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut index = start + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index += 2,
            b'`' => return Some(index + 1),
            b'$' if bytes.get(index + 1) == Some(&b'{') => {
                index = skip_template_expression(text, index + 2)?;
            }
            _ => index += 1,
        }
    }
    None
}

fn statement_end(text: &str, start: usize, limit: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut index = start;
    let mut parentheses = 0usize;
    let mut brackets = 0usize;
    let mut braces = 0usize;
    let mut quote = None;
    let mut escaped = false;
    let mut regex = false;
    let mut regex_class = false;
    let mut line_comment = false;
    let mut block_comment = false;
    let mut can_start_regex = true;
    while index < limit {
        let byte = bytes[index];
        let next = bytes.get(index + 1).copied();
        if regex {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if regex_class {
                if byte == b']' {
                    regex_class = false;
                }
            } else if byte == b'[' {
                regex_class = true;
            } else if byte == b'/' {
                regex = false;
                can_start_regex = false;
            }
        } else if line_comment {
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
        } else if byte == b'/' && can_start_regex {
            regex = true;
            regex_class = false;
        } else if byte == b'`' {
            index = skip_template(text, index)?;
            can_start_regex = false;
            continue;
        } else if matches!(byte, b'\'' | b'"') {
            quote = Some(byte);
            can_start_regex = false;
        } else if byte.is_ascii_alphabetic() || byte == b'_' || byte == b'$' {
            let token_start = index;
            index += 1;
            while index < limit
                && (bytes[index].is_ascii_alphanumeric() || matches!(bytes[index], b'_' | b'$'))
            {
                index += 1;
            }
            can_start_regex = matches!(
                &text[token_start..index],
                "await"
                    | "case"
                    | "delete"
                    | "do"
                    | "else"
                    | "in"
                    | "instanceof"
                    | "of"
                    | "return"
                    | "throw"
                    | "typeof"
                    | "void"
                    | "yield"
            );
            continue;
        } else if byte.is_ascii_digit() {
            index += 1;
            while index < limit && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'.') {
                index += 1;
            }
            can_start_regex = false;
            continue;
        } else if byte == b'(' {
            parentheses += 1;
            can_start_regex = true;
        } else if byte == b')' {
            parentheses = parentheses.checked_sub(1)?;
            can_start_regex = false;
        } else if byte == b'[' {
            brackets += 1;
            can_start_regex = true;
        } else if byte == b']' {
            brackets = brackets.checked_sub(1)?;
            can_start_regex = false;
        } else if byte == b'{' {
            braces += 1;
            can_start_regex = true;
        } else if byte == b'}' {
            braces = braces.checked_sub(1)?;
            can_start_regex = false;
        } else if matches!(byte, b'.' | b')' | b']') {
            can_start_regex = false;
        } else if matches!(
            byte,
            b';' | b','
                | b':'
                | b'?'
                | b'='
                | b'!'
                | b'&'
                | b'|'
                | b'+'
                | b'-'
                | b'*'
                | b'%'
                | b'^'
                | b'~'
                | b'<'
                | b'>'
        ) {
            if byte == b';' && parentheses == 0 && brackets == 0 && braces == 0 {
                return Some(line_end(text, index));
            }
            can_start_regex = true;
        }
        index += 1;
    }
    None
}

fn matching_delimiter(text: &str, open: usize, opening: u8, closing: u8) -> Option<usize> {
    let mut depth = 0usize;
    let mut quote = None;
    let mut escaped = false;
    let mut regex = false;
    let mut regex_class = false;
    let mut line_comment = false;
    let mut block_comment = false;
    let mut can_start_regex = true;
    let bytes = text.as_bytes();
    let mut index = open;
    while index < bytes.len() {
        let byte = bytes[index];
        let next = bytes.get(index + 1).copied();
        if regex {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if regex_class {
                if byte == b']' {
                    regex_class = false;
                }
            } else if byte == b'[' {
                regex_class = true;
            } else if byte == b'/' {
                regex = false;
                can_start_regex = false;
            }
        } else if line_comment {
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
        } else if byte == b'/' && can_start_regex {
            regex = true;
            regex_class = false;
        } else if byte == b'`' {
            index = skip_template(text, index)?;
            can_start_regex = false;
            continue;
        } else if matches!(byte, b'\'' | b'"') {
            quote = Some(byte);
            can_start_regex = false;
        } else if byte.is_ascii_alphabetic() || byte == b'_' || byte == b'$' {
            let start = index;
            index += 1;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || matches!(bytes[index], b'_' | b'$'))
            {
                index += 1;
            }
            let identifier = &text[start..index];
            can_start_regex = matches!(
                identifier,
                "await"
                    | "case"
                    | "delete"
                    | "do"
                    | "else"
                    | "in"
                    | "instanceof"
                    | "of"
                    | "return"
                    | "throw"
                    | "typeof"
                    | "void"
                    | "yield"
            );
            continue;
        } else if byte.is_ascii_digit() {
            index += 1;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'.')
            {
                index += 1;
            }
            can_start_regex = false;
            continue;
        } else if byte == opening {
            depth += 1;
            can_start_regex = true;
        } else if byte == closing {
            depth -= 1;
            if depth == 0 {
                return Some(index);
            }
            can_start_regex = false;
        } else if matches!(byte, b'(' | b'[') {
            can_start_regex = true;
        } else if matches!(byte, b')' | b']' | b'.') {
            can_start_regex = false;
        } else if matches!(
            byte,
            b';' | b','
                | b':'
                | b'?'
                | b'='
                | b'!'
                | b'&'
                | b'|'
                | b'+'
                | b'-'
                | b'*'
                | b'%'
                | b'^'
                | b'~'
                | b'<'
                | b'>'
        ) {
            can_start_regex = true;
        }
        index += 1;
    }
    None
}

fn matching_brace(text: &str, open: usize) -> Option<usize> {
    matching_delimiter(text, open, b'{', b'}')
}

fn enclosing_wrapper(text: &str, class_at: usize) -> Option<(usize, usize, usize, bool)> {
    let mut search = class_at;
    loop {
        let (wrapper_at, pattern, asynchronous) = [("(() => {", false), ("(async () => {", true)]
            .into_iter()
            .filter_map(|(pattern, asynchronous)| {
                text[..search]
                    .rfind(pattern)
                    .map(|at| (at, pattern, asynchronous))
            })
            .max_by_key(|(at, _, _)| *at)?;
        search = wrapper_at;
        let wrapper_open = wrapper_at + pattern.len() - 1;
        let Some(wrapper_close) = matching_brace(text, wrapper_open) else {
            continue;
        };
        if wrapper_open < class_at && class_at < wrapper_close {
            return Some((wrapper_at, wrapper_open, wrapper_close, asynchronous));
        }
    }
}

fn enclosing_downlevel_decorator_block(
    text: &str,
    class_close: usize,
    wrapper_close: usize,
    await_at: usize,
) -> Option<(usize, usize)> {
    let mut search = class_close;
    while let Some(offset) = text[search..wrapper_close].find("(() => {") {
        let block_at = search + offset;
        let block_open = block_at + "(() => {".len() - 1;
        let block_close = matching_brace(text, block_open)?;
        if block_open < await_at
            && await_at < block_close
            && text[block_open..block_close].contains("__esDecorate(")
        {
            return Some((block_at, block_close));
        }
        search = block_at + "(() => {".len();
    }
    None
}

struct DecoratorAssignment {
    name_start: usize,
    name_end: usize,
    array_open: usize,
}

fn next_decorator_assignment(text: &str, from: usize, until: usize) -> Option<DecoratorAssignment> {
    let mut search = from;
    while let Some(offset) = text[search..until].find("_decorators") {
        let marker = search + offset;
        let mut name_end = marker + "_decorators".len();
        search = name_end;
        // The pinned emitter appends a numeric suffix when an outer binding
        // already uses its generated name. Keep that suffix in every scan.
        if text[name_end..until].starts_with('_') {
            let digits = text[name_end + 1..until]
                .bytes()
                .take_while(u8::is_ascii_digit)
                .count();
            if digits == 0 {
                continue;
            }
            name_end += 1 + digits;
        }
        if text[name_end..until].starts_with(" = [") {
            return Some(DecoratorAssignment {
                name_start: identifier_start(text, marker),
                name_end,
                array_open: name_end + 3,
            });
        }
    }
    None
}

fn decorator_assignment_before(
    text: &str,
    from: usize,
    until: usize,
) -> Option<DecoratorAssignment> {
    let mut search = from;
    let mut last = None;
    while let Some(assignment) = next_decorator_assignment(text, search, until) {
        search = assignment.array_open + 1;
        last = Some(assignment);
    }
    last
}

fn await_is_in_decorator_assignment(text: &str, await_at: usize) -> bool {
    let Some(assignment) = decorator_assignment_before(text, 0, await_at) else {
        return false;
    };
    let open = assignment.array_open;
    if !matching_delimiter(text, open, b'[', b']').is_some_and(|close| await_at < close) {
        return false;
    }
    let mut search = open + 1;
    while let Some(offset) = text[search..await_at].find("=>") {
        let arrow = search + offset;
        let body_start = text[arrow + 2..]
            .bytes()
            .position(|byte| !byte.is_ascii_whitespace())
            .map(|offset| arrow + 2 + offset);
        if let Some(body_start) = body_start
            && text.as_bytes().get(body_start) == Some(&b'{')
            && matching_brace(text, body_start).is_some_and(|close| await_at < close)
        {
            return false;
        }
        search = arrow + 2;
    }
    search = open + 1;
    while let Some(offset) = text[search..await_at].find("function") {
        let function_at = search + offset;
        let body_start = text[function_at + "function".len()..await_at]
            .find('{')
            .map(|offset| function_at + "function".len() + offset);
        if let Some(body_start) = body_start
            && matching_brace(text, body_start).is_some_and(|close| await_at < close)
        {
            return false;
        }
        search = function_at + "function".len();
    }
    true
}

fn apply(text: &str, mut edits: Vec<Edit>, resume_at: usize) -> Result<Rewrite, String> {
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
        new: output,
        copies,
        resume_at,
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

pub(crate) fn rewrite(text: &str, from: usize) -> Result<Option<Rewrite>, String> {
    if !text.contains("__esDecorate(") {
        return Ok(None);
    }
    let mut cursor = from;
    let (
        await_at,
        class_at,
        class_open,
        class_close,
        downlevel,
        wrapper_at,
        wrapper_open,
        wrapper_close,
    ) = 'awaits: loop {
        let Some(at) = find_await_set(text, cursor) else {
            return Ok(None);
        };
        let mut class_search = at;
        loop {
            let return_class = text[..class_search].rfind("return class ");
            let assigned_class = text[..class_search].rfind("= class ");
            let Some(class_token) = return_class.into_iter().chain(assigned_class).max() else {
                cursor = at + "await __nmath".len();
                continue 'awaits;
            };
            // A decorator argument can contain an earlier, already closed
            // class. Try its enclosing candidates before discarding
            // this awaited assignment.
            class_search = class_token;
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
            let Some((wrapper_at, wrapper_open, wrapper_close, asynchronous_wrapper)) =
                enclosing_wrapper(text, class_at)
            else {
                continue;
            };
            let downlevel =
                enclosing_downlevel_decorator_block(text, class_close, wrapper_close, at).is_some();
            if at > class_close && !downlevel {
                continue;
            }
            let Some(assignment) = decorator_assignment_before(text, class_open, at) else {
                continue;
            };
            let decorator_var = &text[assignment.name_start..assignment.name_end];
            let field_call = format!("{decorator_var}, {{ kind: \"field\"");
            let accessor_call = format!("{decorator_var}, {{ kind: \"accessor\"");
            if !text[class_open..wrapper_close].contains(&field_call)
                && !text[class_open..wrapper_close].contains(&accessor_call)
            {
                continue;
            }
            if !await_is_in_decorator_assignment(text, at) {
                cursor = at + "await __nmath".len();
                continue 'awaits;
            }
            if asynchronous_wrapper {
                cursor = at + "await __nmath".len();
                continue 'awaits;
            }
            break 'awaits (
                at,
                class_at,
                class_open,
                class_close,
                downlevel,
                wrapper_at,
                wrapper_open,
                wrapper_close,
            );
        }
    };
    if wrapper_close < class_close {
        return Err("Generated class is outside its wrapper".into());
    }
    let mut edits = Vec::new();
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
                .find("const _metadata")
                .map(|offset| static_open + offset)
                .ok_or("Generated decorator metadata is missing")?;
            let metadata_start = line_start(text, metadata_at);
            let metadata_end = line_end(text, metadata_at);
            let metadata_value_start = metadata_at + "const ".len();
            let metadata_name_end = text[metadata_value_start..]
                .bytes()
                .position(|byte| !(byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$')))
                .map(|offset| metadata_value_start + offset)
                .ok_or("Generated metadata identifier is missing")?;
            if !text[metadata_name_end..].trim_start().starts_with('=') {
                return Err("Generated decorator metadata declaration is malformed".into());
            }
            let metadata_identifier = &text[metadata_value_start..metadata_name_end];
            let metadata_value_end = text[metadata_value_start..metadata_end]
                .rfind(';')
                .map(|offset| metadata_value_start + offset)
                .ok_or("Generated metadata terminator is missing")?;
            let mut assignments = Vec::new();
            let mut position = metadata_end;
            while position < static_close {
                let Some(decorator) = next_decorator_assignment(text, position, static_close)
                else {
                    break;
                };
                let assignment_name_start = decorator.name_start;
                let assignment_start = line_start(text, assignment_name_start);
                let end = statement_end(text, assignment_start, static_close)
                    .ok_or("Generated decorator assignment is unbalanced")?;
                let assignment = &text[assignment_start..end];
                if !assignment.contains("__esDecorate(") || find_await_set(assignment, 0).is_some()
                {
                    let value_end = assignment_start + assignment.rfind(';').unwrap();
                    assignments.push((
                        assignment_start,
                        end,
                        assignment_name_start,
                        value_end,
                        find_await_set(assignment, 0).is_some(),
                    ));
                }
                position = end;
            }
            let Some(last) = assignments.iter().rposition(|assignment| assignment.4) else {
                return Ok(None);
            };
            // Start from this expression instead of retesting every sentinel
            // allocated to earlier classes. Still reject names present in input.
            let sentinel = (await_at..)
                .map(|index| format!("__nmath_decorator_sentinel_{index}"))
                .find(|candidate| !text.contains(candidate))
                .unwrap();
            edits.push(Edit {
                start: class_at,
                end: class_at,
                parts: vec![Part::New(format!(
                    "let {metadata_identifier}; let {sentinel};\n        "
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
                ", {sentinel} = globalThis.Symbol())]() {{}}\n        "
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
    if downlevel {
        let (block_at, block_close) =
            enclosing_downlevel_decorator_block(text, class_close, wrapper_close, await_at)
                .ok_or("Generated downlevel decorator block is missing")?;
        let invocation = text
            .get(block_close..block_close + 4)
            .ok_or("Generated downlevel decorator invocation is missing")?;
        if invocation != "})()" {
            return Err("Generated downlevel decorator invocation is malformed".into());
        }
        edits.push(Edit {
            start: block_at,
            end: block_at + "(() => {".len(),
            parts: vec![Part::New("(await (async () => {".into())],
        });
        edits.push(Edit {
            start: block_close + 4,
            end: block_close + 4,
            parts: vec![Part::New(")".into())],
        });
    }
    let return_at = text[wrapper_open..class_open]
        .rfind("return class ")
        .map(|offset| wrapper_open + offset)
        .or_else(|| {
            text[class_close..wrapper_close]
                .find("return ")
                .map(|offset| class_close + offset)
        })
        .ok_or("Generated class wrapper return is missing")?;
    let expression_start = return_at + "return".len();
    let expression_start = expression_start
        + text[expression_start..]
            .len()
            .saturating_sub(text[expression_start..].trim_start().len());
    let return_end = statement_end(text, return_at, wrapper_close)
        .ok_or("Generated class wrapper return is unbalanced")?;
    let expression_end = return_at
        + text[return_at..return_end]
            .rfind(';')
            .ok_or("Generated class wrapper return terminator is missing")?;
    let value_prefix = if downlevel { "{ value: (" } else { "{ value: " };
    let value_suffix = if downlevel { ") }" } else { " }" };
    edits.push(Edit {
        start: wrapper_at,
        end: wrapper_at + "(() => {".len(),
        parts: vec![Part::New("(await (async () => {".into())],
    });
    edits.push(Edit {
        start: expression_start,
        end: expression_start,
        parts: vec![Part::New(value_prefix.into())],
    });
    edits.push(Edit {
        start: expression_end,
        end: expression_end,
        parts: vec![Part::New(value_suffix.into())],
    });
    let invocation = text
        .get(wrapper_close..wrapper_close + 4)
        .ok_or("Generated class wrapper invocation is missing")?;
    if invocation != "})()" {
        return Err("Generated class wrapper invocation is malformed".into());
    }
    edits.push(Edit {
        start: wrapper_close + 4,
        end: wrapper_close + 4,
        parts: vec![Part::New(").value".into())],
    });
    apply(text, edits, wrapper_at).map(Some)
}

#[cfg(test)]
mod tests {
    use super::{await_is_in_decorator_assignment, matching_brace};

    #[test]
    fn matching_brace_ignores_regex_literals() {
        let text = r#"class C {
    method() { return /[}]/.test("}"); }
    [(_decorators = [await __nmath.set(a, 0, 1)])]() {}
}"#;
        let open = text.find("class C {").unwrap() + "class C ".len();
        let close = matching_brace(text, open).unwrap();
        assert_eq!(close, text.rfind('}').unwrap());
    }

    #[test]
    fn matching_brace_ignores_nested_template_literals() {
        let text = "class C { method() { return `outer${`}`}`; } }";
        let open = text.find("class C {").unwrap() + "class C ".len();
        let close = matching_brace(text, open).unwrap();
        assert_eq!(close, text.rfind('}').unwrap());
    }

    #[test]
    fn nested_decorator_callback_await_is_not_eager_assignment() {
        let text = "_decorators = [dec(async () => { await __nmath.set(a, 0, 1); })];";
        let await_at = text.find("await __nmath").unwrap();
        assert!(!await_is_in_decorator_assignment(text, await_at));
    }
}
