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

#[derive(Clone, Copy)]
struct WrapperReturn {
    body_open: usize,
    expression_start: usize,
    expression_end: usize,
}

#[derive(Clone, Copy)]
struct GeneratedClass {
    start: usize,
    open: usize,
    close: usize,
}

#[derive(Clone, Copy)]
struct DecoratorAssignment {
    name_start: usize,
    name_end: usize,
    array_open: usize,
    array_close: usize,
}

fn next_decorator_assignment(
    assignments: &[DecoratorAssignment],
    from: usize,
    until: usize,
) -> Option<DecoratorAssignment> {
    let start = assignments.partition_point(|assignment| assignment.name_start < from);
    assignments
        .get(start)
        .copied()
        .filter(|assignment| assignment.array_open < until)
}

fn decorator_assignment_before(
    assignments: &[DecoratorAssignment],
    from: usize,
    until: usize,
) -> Option<DecoratorAssignment> {
    let end = assignments.partition_point(|assignment| assignment.array_open < until);
    assignments[..end]
        .iter()
        .rev()
        .copied()
        .find(|assignment| assignment.name_start >= from && until < assignment.array_close)
}

fn await_is_in_decorator_assignment(
    assignments: &[DecoratorAssignment],
    await_at: usize,
    function_body: Option<usize>,
) -> bool {
    let Some(assignment) = decorator_assignment_before(assignments, 0, await_at) else {
        return false;
    };
    // A function body that starts inside the decorator array defers the call
    // until invocation. Literal contents cannot introduce a parser body node.
    function_body.is_none_or(|start| start < assignment.array_open)
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

fn parsed_decorator_calls(
    text: &str,
) -> (
    Vec<usize>,
    Vec<DecoratorAssignment>,
    Vec<GeneratedClass>,
    Vec<WrapperReturn>,
) {
    use tsrs_ast::{Kind, SourceFileParseOptions};
    use tsrs_core::{ScriptKind, arena::Region, tspath};

    if !text.contains("await __nmath") {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }
    // Parse the pinned emitter's JavaScript before relocating its invalid
    // static-block awaits. The parser still represents those awaits, while
    // literal text, comments and regular expressions never become await nodes.
    // Keep this temporary tree out of the surrounding compilation arena.
    let region = Region::new((text.len() * 16).clamp(64 << 10, 64 << 20));
    let _scope = region.enter();
    let parsed = tsrs_parser::parse_source_file(
        SourceFileParseOptions {
            file_name: "/__nmath__/decorators.js".into(),
            path: tspath::to_path("/__nmath__/decorators.js", "", true),
            external_module_indicator_options: Default::default(),
        },
        text,
        ScriptKind::JS,
    );
    let mut pending = vec![(parsed.as_node(), None::<usize>)];
    let mut positions = Vec::new();
    let mut assignments = Vec::new();
    let mut classes = Vec::new();
    let mut returns = Vec::new();
    while let Some((node, function_body)) = pending.pop() {
        if node.kind() == Kind::ReturnStatement {
            if let (Some(body_open), Some(expression)) =
                (function_body, node.as_return_statement().expression())
            {
                returns.push(WrapperReturn {
                    body_open,
                    expression_start: tsrs_scanner::skip_trivia(
                        text,
                        expression.pos().expect_source("return expression"),
                    ) as usize,
                    expression_end: expression.end().expect_source("return expression") as usize,
                });
            }
        }
        if node.kind() == Kind::ClassExpression {
            let start = tsrs_scanner::skip_trivia(text, node.pos().expect_source("class")) as usize;
            let open = node
                .as_class_expression()
                .members()
                .pos()
                .expect_source("class members") as usize
                - 1;
            let close = node.end().expect_source("class") as usize - 1;
            // Use parser boundaries: class-like text in literals and braces
            // in heritage expressions cannot identify the generated class.
            if text.as_bytes().get(open) == Some(&b'{') && text.as_bytes().get(close) == Some(&b'}')
            {
                classes.push(GeneratedClass { start, open, close });
            }
        }
        if node.kind() == Kind::BinaryExpression {
            let expression = node.as_binary_expression();
            if expression.operator_token.kind() == Kind::EqualsToken
                && expression.left.kind() == Kind::Identifier
                && expression.right().kind() == Kind::ArrayLiteralExpression
            {
                let name_start = tsrs_scanner::skip_trivia(
                    text,
                    expression.left.pos().expect_source("decorator identifier"),
                ) as usize;
                let name_end = expression.left.end().expect_source("decorator identifier") as usize;
                let name = &text[name_start..name_end];
                // Match the pinned emitter's generated name, including its
                // collision suffix, only on an actual array assignment node.
                let generated = name.rsplit_once("_decorators").is_some_and(|(_, suffix)| {
                    suffix.is_empty()
                        || suffix.strip_prefix('_').is_some_and(|digits| {
                            !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
                        })
                });
                if generated {
                    assignments.push(DecoratorAssignment {
                        name_start,
                        name_end,
                        array_open: tsrs_scanner::skip_trivia(
                            text,
                            expression.right().pos().expect_source("decorator array"),
                        ) as usize,
                        array_close: expression.right().end().expect_source("decorator array")
                            as usize
                            - 1,
                    });
                }
            }
        }
        if node.kind() == Kind::AwaitExpression {
            let at =
                tsrs_scanner::skip_trivia(text, node.pos().expect_source("emitted await")) as usize;
            if let Some(mut suffix) = text[at..].strip_prefix("await __nmath") {
                if let Some(rest) = suffix.strip_prefix('_') {
                    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
                    suffix = if digits > 0 { &rest[digits..] } else { "" };
                }
                if suffix.starts_with(".set(") {
                    positions.push((at, function_body));
                }
            }
        }
        let body = if tsrs_ast::is_function_like(node) {
            node.body()
        } else {
            None
        };
        for child in node.iter_children() {
            let child_body = if Some(child) == body {
                Some(
                    tsrs_scanner::skip_trivia(text, child.pos().expect_source("function body"))
                        as usize,
                )
            } else {
                function_body
            };
            pending.push((child, child_body));
        }
    }
    assignments.sort_unstable_by_key(|assignment| assignment.name_start);
    let mut awaits: Vec<_> = positions
        .into_iter()
        .filter_map(|(at, body)| {
            await_is_in_decorator_assignment(&assignments, at, body).then_some(at)
        })
        .collect();
    awaits.sort_unstable();
    classes.sort_unstable_by_key(|class| class.start);
    (awaits, assignments, classes, returns)
}

pub(crate) struct Rewriter {
    awaits: Vec<usize>,
    assignments: Vec<DecoratorAssignment>,
    classes: Vec<GeneratedClass>,
    returns: Vec<WrapperReturn>,
    cursor: usize,
}

impl Rewriter {
    pub fn new(text: &str) -> Self {
        let (awaits, assignments, classes, returns) = if text.contains("__esDecorate(") {
            parsed_decorator_calls(text)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
        Self {
            awaits,
            assignments,
            classes,
            returns,
            cursor: 0,
        }
    }

    pub fn rewrite(&mut self, text: &str) -> Result<Option<Rewrite>, String> {
        let Some(rewrite) = rewrite(
            text,
            self.cursor,
            &self.awaits,
            &self.assignments,
            &self.classes,
            &self.returns,
        )?
        else {
            return Ok(None);
        };
        // Copy spans preserve the helper calls themselves. Relocate their
        // parsed positions so later passes do not parse already repaired code.
        self.awaits = self
            .awaits
            .iter()
            .map(|at| {
                rewrite
                    .copies
                    .iter()
                    .find(|span| span.old_start <= *at && *at < span.old_end)
                    .map(|span| span.new_start + at - span.old_start)
                    .ok_or_else(|| {
                        "Awaited helper was lost during decorator relocation".to_string()
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.awaits.sort_unstable();
        // Assignment boundaries must follow the same copies as awaits. Keep
        // parsing once per output, including when nested classes are moved.
        let relocate = |at: usize| {
            rewrite.copies.iter().find_map(|span| {
                (span.old_start <= at && at < span.old_end)
                    .then(|| span.new_start + at - span.old_start)
            })
        };
        self.assignments = self
            .assignments
            .iter()
            .filter_map(|assignment| {
                Some(DecoratorAssignment {
                    name_start: relocate(assignment.name_start)?,
                    name_end: relocate(assignment.name_end - 1)? + 1,
                    array_open: relocate(assignment.array_open)?,
                    array_close: relocate(assignment.array_close)?,
                })
            })
            .collect();
        self.assignments
            .sort_unstable_by_key(|assignment| assignment.name_start);
        self.classes = self
            .classes
            .iter()
            .filter_map(|class| {
                Some(GeneratedClass {
                    start: relocate(class.start)?,
                    open: relocate(class.open)?,
                    close: relocate(class.close)?,
                })
            })
            .collect();
        self.classes.sort_unstable_by_key(|class| class.start);
        self.returns = self
            .returns
            .iter()
            .filter_map(|statement| {
                Some(WrapperReturn {
                    body_open: relocate(statement.body_open)?,
                    expression_start: relocate(statement.expression_start)?,
                    expression_end: relocate(statement.expression_end - 1)? + 1,
                })
            })
            .collect();
        self.cursor = rewrite.resume_at;
        Ok(Some(rewrite))
    }
}

fn rewrite(
    text: &str,
    from: usize,
    awaits: &[usize],
    decorator_assignments: &[DecoratorAssignment],
    classes: &[GeneratedClass],
    returns: &[WrapperReturn],
) -> Result<Option<Rewrite>, String> {
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
        let Some(at) = awaits.iter().copied().find(|at| *at >= cursor) else {
            return Ok(None);
        };
        let mut class_search = at;
        loop {
            let end = classes.partition_point(|class| class.start < class_search);
            let Some(class) = classes[..end].last() else {
                cursor = at + "await __nmath".len();
                continue 'awaits;
            };
            // Earlier inner classes can already be closed. Walk real parsed
            // class candidates until the enclosing generated class is found.
            class_search = class.start;
            let class_at = line_start(text, class.start)
                + text[line_start(text, class.start)..class.start]
                    .bytes()
                    .take_while(|byte| *byte == b' ' || *byte == b'\t')
                    .count();
            let class_open = class.open;
            let class_close = class.close;
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
            let Some(assignment) =
                decorator_assignment_before(decorator_assignments, class_open, at)
            else {
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
            if asynchronous_wrapper {
                // A repaired inner wrapper can itself sit in an outer
                // decorator array. Keep walking enclosing classes so its
                // newly introduced await is relocated with that array.
                continue;
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
                let Some(decorator) =
                    next_decorator_assignment(decorator_assignments, position, static_close)
                else {
                    break;
                };
                let assignment_name_start = decorator.name_start;
                let assignment_start = line_start(text, assignment_name_start);
                let end = statement_end(text, assignment_start, static_close)
                    .ok_or("Generated decorator assignment is unbalanced")?;
                let assignment = &text[assignment_start..end];
                let has_await = awaits.iter().any(|at| assignment_start <= *at && *at < end);
                if !assignment.contains("__esDecorate(") || has_await {
                    let value_end = assignment_start + assignment.rfind(';').unwrap();
                    assignments.push((
                        assignment_start,
                        end,
                        assignment_name_start,
                        value_end,
                        has_await,
                    ));
                }
                position = end;
            }
            let Some(last) = assignments.iter().rposition(|assignment| assignment.4) else {
                return Ok(None);
            };
            // Start from this expression instead of retesting every sentinel
            // allocated to earlier classes. Still reject names present in
            // input.
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
            // The lowering pass already chose an uncaptured runtime import.
            // Create the key there rather than resolving user-local globals.
            let runtime = text[await_at + "await ".len()..]
                .split_once(".set(")
                .ok_or("Awaited runtime namespace is missing")?
                .0;
            parts.push(Part::New(format!(
                ", {sentinel} = {runtime}.decoratorKey())]() {{}}\n        "
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
    // Select a return owned by this function body, never text inside a
    // literal or a nested callback. Retain exact parsed expression boundaries
    // so boxing cannot include trivia or a neighboring statement.
    let statement = returns
        .iter()
        .filter(|statement| statement.body_open == wrapper_open)
        .max_by_key(|statement| statement.expression_start)
        .ok_or("Generated class wrapper return is missing")?;
    let expression_start = statement.expression_start;
    let expression_end = statement.expression_end;
    // An array element does not infer an anonymous class name, unlike an
    // object property. It still boxes thenable constructors during awaiting.
    let value_prefix = if downlevel { "[(" } else { "[" };
    let value_suffix = if downlevel { ")]" } else { "]" };
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
        parts: vec![Part::New(")[0]".into())],
    });
    apply(text, edits, wrapper_at).map(Some)
}

#[cfg(test)]
mod tests {
    use super::{matching_brace, parsed_decorator_calls};

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
        assert!(parsed_decorator_calls(text).0.is_empty());
    }
}
