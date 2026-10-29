//! Compose the external emitter's UTF-16 maps with the transform's byte maps.

use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
};

use nmath_nts::TransformOutput;
use serde_json::Value;
use tsrs_sourcemap::{decode_mappings, new_generator};

pub(crate) type Sources = BTreeMap<String, (String, String, TransformOutput)>;

// The emitter counts ECMAScript line terminators in source maps. LSP uses a
// separate CR/LF line model, so keep these conversions local to composition.
fn ecma_offset_of_line_character(text: &str, line: u32, character: u32) -> u32 {
    let starts = tsrs_core::compute_ecma_line_starts(text);
    let line = (line as usize).min(starts.len() - 1);
    let start = starts[line] as usize;
    let end = if let Some(&next) = starts.get(line + 1) {
        let before_next = &text[..next as usize];
        let terminator_len = if before_next.ends_with("\r\n") {
            2
        } else if before_next.ends_with('\r') || before_next.ends_with('\n') {
            1
        } else {
            // The pinned ECMA line map only has these two remaining breaks.
            '\u{2028}'.len_utf8()
        };
        next as usize - terminator_len
    } else {
        text.len()
    };
    let mut units = 0;
    for (index, ch) in text[start..end].char_indices() {
        if units >= character {
            return (start + index) as u32;
        }
        units += ch.len_utf16() as u32;
    }
    end as u32
}

fn ecma_line_character(text: &str, offset: u32) -> (u32, u32) {
    let offset = (offset as usize).min(text.len());
    let starts = tsrs_core::compute_ecma_line_starts(text);
    let line = starts.partition_point(|start| *start as usize <= offset) - 1;
    let start = starts[line] as usize;
    let mut end = offset;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    (line as u32, text[start..end].encode_utf16().count() as u32)
}

pub(crate) fn compose(path: &str, text: &str, sources: &Sources) -> Result<String, String> {
    let mut map: Value =
        serde_json::from_str(text).map_err(|e| format!("Invalid source map: {e}"))?;
    let directory = Path::new(path)
        .parent()
        .ok_or("Source map has no directory")?;
    let entries = map["sources"]
        .as_array()
        .ok_or("Source map has no sources")?;
    let mut generator = new_generator(
        map["file"].as_str().unwrap_or_default(),
        "",
        &directory.to_string_lossy(),
        Default::default(),
    );
    let mut resolved = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        let source = entry.as_str().ok_or("Invalid source path")?;
        // Neither the virtual source nor the buffered output directory must
        // exist on disk. Resolve relative components without filesystem I/O.
        let mut absolute = PathBuf::new();
        for component in directory.join(source).components() {
            match component {
                Component::ParentDir => {
                    absolute.pop();
                }
                Component::CurDir => {}
                other => absolute.push(other.as_os_str()),
            }
        }
        let absolute_name = absolute.to_string_lossy().replace('\\', "/");
        let mapped = sources.get(&absolute_name);
        let name = mapped.map_or_else(|| absolute_name, |s| s.0.clone());
        let id = generator.add_source(&name);
        if let Some((_, original, _)) = mapped {
            generator.set_source_content(id, original)?;
        } else if let Some(content) = map["sourcesContent"][index].as_str() {
            generator.set_source_content(id, content)?;
        }
        resolved.push((id, mapped));
    }
    let mut decoder = decode_mappings(map["mappings"].as_str().ok_or("Missing mappings")?);
    for mapping in decoder.values() {
        let location = mapping.source_index.get().and_then(|index| {
            let (id, mapped) = resolved.get(index as usize)?;
            let line = mapping.source_line?;
            let column = mapping.source_character?;
            if let Some((_, original, transformed)) = mapped {
                let offset = ecma_offset_of_line_character(&transformed.text, line, column);
                let original_offset = transformed.span_map.virtual_to_original(offset)?;
                let (line, column) = ecma_line_character(original, original_offset);
                Some((*id, line, column))
            } else {
                Some((*id, line, column))
            }
        });
        if let Some((id, line, column)) = location {
            if let Some(name) = mapping
                .name_index
                .get()
                .and_then(|index| map["names"][index as usize].as_str())
            {
                let name = generator.add_name(name);
                generator.add_named_source_mapping(
                    mapping.generated_line,
                    mapping.generated_character,
                    id,
                    line,
                    column,
                    name,
                )?;
            } else {
                generator.add_source_mapping(
                    mapping.generated_line,
                    mapping.generated_character,
                    id,
                    line,
                    column,
                )?;
            }
        } else {
            // Explicitly end the preceding source range for synthesized code.
            generator.add_generated_mapping(mapping.generated_line, mapping.generated_character)?;
        }
    }
    if let Some(error) = decoder.error() {
        return Err(format!("Invalid source map mappings: {error}"));
    }
    let composed: Value = serde_json::from_str(&generator.string()).map_err(|e| e.to_string())?;
    for key in ["sources", "sourcesContent", "names", "mappings"] {
        map[key] = composed[key].clone();
    }
    map["sourceRoot"] = Value::String(String::new());
    serde_json::to_string(&map).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use tsrs_sourcemap::{decode_mappings, new_generator};

    use super::*;

    #[test]
    fn synthesized_import_has_no_original_location() {
        let source = "export const x = a + b;\n";
        let mut plan = nmath_nts::semantic::LoweringPlan::default();
        plan.operations.insert((17, 22));
        let transformed = nmath_nts::transform_with_plan(
            "/project/src/main.nts",
            source,
            nmath_nts::TransformOptions::default(),
            &plan,
        );
        let map = serde_json::json!({
            "version": 3, "file": "main.js", "sources": ["../src/main.ts"],
            "sourcesContent": [transformed.text], "names": [], "mappings": "AAAA;AACA"
        });
        let sources = BTreeMap::from([(
            "/project/src/main.ts".into(),
            ("/project/src/main.nts".into(), source.into(), transformed),
        )]);
        let composed: Value = serde_json::from_str(
            &compose("/project/output/main.js.map", &map.to_string(), &sources).unwrap(),
        )
        .unwrap();
        let mut decoder = decode_mappings(composed["mappings"].as_str().unwrap());
        let entries: Vec<_> = decoder.values().collect();
        assert!(!entries[0].is_source_mapping());
        assert_eq!(entries[1].source_line, Some(0));
        assert_eq!(entries[1].source_character, Some(0));
        assert_eq!(composed["sourcesContent"][0], source);
    }

    #[test]
    fn malformed_mappings_fail_instead_of_publishing_wrong_locations() {
        let map = serde_json::json!({"version": 3, "file": "a.js", "sources": [], "names": [], "mappings": "!"});
        assert!(compose("/project/a.js.map", &map.to_string(), &Sources::new()).is_err());
    }

    #[test]
    fn ecma_coordinates_round_trip_all_terminators() {
        for terminator in ["\n", "\r", "\r\n", "\u{2028}", "\u{2029}"] {
            let text = format!("😀x{terminator}{terminator}z");
            let second_start = "😀x".len() + terminator.len();
            let third_start = second_start + terminator.len();
            assert_eq!(ecma_line_character(&text, 0), (0, 0));
            assert_eq!(ecma_line_character(&text, 4), (0, 2));
            assert_eq!(ecma_line_character(&text, 5), (0, 3));
            assert_eq!(ecma_line_character(&text, second_start as u32), (1, 0));
            assert_eq!(ecma_line_character(&text, third_start as u32), (2, 0));
            assert_eq!(ecma_line_character(&text, text.len() as u32), (2, 1));
            for (line, column, offset) in [
                (0, 0, 0),
                (0, 2, 4),
                (0, 3, 5),
                (1, 0, second_start),
                (2, 0, third_start),
                (2, 1, text.len()),
            ] {
                assert_eq!(
                    ecma_offset_of_line_character(&text, line, column),
                    offset as u32
                );
            }
        }
    }

    #[test]
    fn composed_unicode_mappings_keep_original_lines_and_columns() {
        for separator in ['\u{2028}', '\u{2029}'] {
            for helper in [false, true] {
                let source = format!("const a = 1;{separator}export const b = a + a;\n");
                let mut plan = nmath_nts::semantic::LoweringPlan::default();
                if helper {
                    let start = source.rfind("a + a").unwrap() as u32;
                    plan.operations.insert((start, start + 5));
                }
                let transformed = nmath_nts::transform_with_plan(
                    "/project/src/main.nts",
                    &source,
                    nmath_nts::TransformOptions::default(),
                    &plan,
                );
                let export_offset = transformed.text.find("export").unwrap() as u32;
                let (line, column) = ecma_line_character(&transformed.text, export_offset);
                let mut emitter = new_generator("main.js", "", "", Default::default());
                let id = emitter.add_source("../src/main.ts");
                emitter.add_source_mapping(0, 0, id, line, column).unwrap();
                let sources = BTreeMap::from([(
                    "/project/src/main.ts".into(),
                    ("/project/src/main.nts".into(), source.clone(), transformed),
                )]);
                let composed: Value = serde_json::from_str(
                    &compose("/project/output/main.js.map", &emitter.string(), &sources).unwrap(),
                )
                .unwrap();
                let mut decoder = decode_mappings(composed["mappings"].as_str().unwrap());
                let entries: Vec<_> = decoder.values().collect();
                assert_eq!(
                    entries[0].source_line,
                    Some(1),
                    "separator={separator:?}, helper={helper}"
                );
                assert_eq!(entries[0].source_character, Some(0));
                assert_eq!(source.find("export"), Some(15));
                assert_eq!(composed["sourcesContent"][0], source);
            }
        }
    }

    #[test]
    fn composed_astral_columns_and_ascii_line_breaks() {
        for terminator in ["\n", "\r", "\r\n", "\u{2028}", "\u{2029}"] {
            let source = format!("const a = '😀';{terminator}export const b = a;\n");
            let transformed = nmath_nts::transform_with_plan(
                "/project/src/main.nts",
                &source,
                nmath_nts::TransformOptions::default(),
                &nmath_nts::semantic::LoweringPlan::default(),
            );
            let emoji_end = transformed.text.find("';").unwrap() as u32;
            let export = transformed.text.find("export").unwrap() as u32;
            let mut emitter = new_generator("main.js", "", "", Default::default());
            let id = emitter.add_source("../src/main.ts");
            for (generated_column, offset) in [(0, emoji_end), (1, export)] {
                let (line, column) = ecma_line_character(&transformed.text, offset);
                emitter
                    .add_source_mapping(0, generated_column, id, line, column)
                    .unwrap();
            }
            let sources = BTreeMap::from([(
                "/project/src/main.ts".into(),
                ("/project/src/main.nts".into(), source, transformed),
            )]);
            let composed: Value = serde_json::from_str(
                &compose("/project/output/main.js.map", &emitter.string(), &sources).unwrap(),
            )
            .unwrap();
            let mut decoder = decode_mappings(composed["mappings"].as_str().unwrap());
            let entries: Vec<_> = decoder.values().collect();
            assert_eq!(
                (entries[0].source_line, entries[0].source_character),
                (Some(0), Some(13))
            );
            assert_eq!(
                (entries[1].source_line, entries[1].source_character),
                (Some(1), Some(0))
            );
        }
    }
}
