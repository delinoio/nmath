//! Compose the external emitter's UTF-16 maps with the transform's byte maps.

use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
};

use nmath_nts::{Segment, SegmentKind, TransformOutput};
use serde_json::Value;
use tsrs_sourcemap::{decode_mappings, new_generator};

pub(crate) type Sources = BTreeMap<String, (String, String, TransformOutput)>;

// The emitter counts ECMAScript line terminators. LSP has a separate CR/LF
// model. Keep this index local to composition so it cannot outlive the text.
struct EcmaCoordinateIndex {
    starts: Vec<u32>,
    content_ends: Vec<u32>,
    boundaries: Vec<u32>,
    utf16_prefix: Vec<u32>,
}

impl EcmaCoordinateIndex {
    fn new(text: &str) -> Self {
        let starts = tsrs_core::compute_ecma_line_starts(text);
        let content_ends = starts
            .iter()
            .enumerate()
            .map(|(line, _)| {
                let Some(&next) = starts.get(line + 1) else {
                    return text.len() as u32;
                };
                let before_next = &text[..next as usize];
                let terminator_len = if before_next.ends_with("\r\n") {
                    2
                } else if before_next.ends_with('\r') || before_next.ends_with('\n') {
                    1
                } else {
                    '\u{2028}'.len_utf8()
                };
                next - terminator_len as u32
            })
            .collect();
        let mut boundaries = Vec::with_capacity(text.chars().count() + 1);
        let mut utf16_prefix = Vec::with_capacity(boundaries.capacity());
        let mut units = 0;
        for (byte, ch) in text.char_indices() {
            boundaries.push(byte as u32);
            utf16_prefix.push(units);
            units += ch.len_utf16() as u32;
        }
        boundaries.push(text.len() as u32);
        utf16_prefix.push(units);
        Self {
            starts,
            content_ends,
            boundaries,
            utf16_prefix,
        }
    }

    fn offset_of_line_character(&self, line: u32, character: u32) -> u32 {
        let line = (line as usize).min(self.starts.len() - 1);
        let start = self.starts[line];
        let end = self.content_ends[line];
        let first = self.boundaries.partition_point(|&byte| byte < start);
        let after = self.boundaries.partition_point(|&byte| byte <= end);
        let target = self.utf16_prefix[first].saturating_add(character);
        let relative = self.utf16_prefix[first..after].partition_point(|&units| units < target);
        if relative == after - first {
            end
        } else {
            self.boundaries[first + relative]
        }
    }

    fn line_character(&self, offset: u32) -> (u32, u32) {
        let offset = offset.min(*self.boundaries.last().unwrap());
        let line = self.starts.partition_point(|&start| start <= offset) - 1;
        let boundary = self.boundaries.partition_point(|&byte| byte <= offset) - 1;
        let line_start = self
            .boundaries
            .partition_point(|&byte| byte < self.starts[line]);
        (
            line as u32,
            self.utf16_prefix[boundary] - self.utf16_prefix[line_start],
        )
    }
}

// Segment ends are ordered, but equal ends can contain an inclusive Verbatim
// endpoint before another segment. Record its first index to retain find()'s
// precedence even for zero-length virtual spans with nonzero original spans.
struct SpanLookup<'a> {
    segments: &'a [Segment],
    inclusive_ends: Vec<(u32, usize)>,
}

impl<'a> SpanLookup<'a> {
    fn new(segments: &'a [Segment]) -> Self {
        let mut inclusive_ends = Vec::new();
        for (index, segment) in segments.iter().enumerate() {
            if segment.kind == SegmentKind::Verbatim
                && inclusive_ends
                    .last()
                    .is_none_or(|&(end, _)| end != segment.virtual_end)
            {
                inclusive_ends.push((segment.virtual_end, index));
            }
        }
        Self {
            segments,
            inclusive_ends,
        }
    }

    fn virtual_to_original(&self, pos: u32) -> Option<u32> {
        let endpoint = self
            .inclusive_ends
            .binary_search_by_key(&pos, |&(end, _)| end)
            .ok()
            .map(|index| self.inclusive_ends[index].1);
        let interior = self
            .segments
            .partition_point(|segment| segment.virtual_end <= pos);
        let index = endpoint
            .into_iter()
            .chain((interior < self.segments.len()).then_some(interior))
            .min()?;
        let segment = &self.segments[index];
        if segment.virtual_start > pos {
            return None;
        }
        Some(match segment.kind {
            SegmentKind::Verbatim => segment.original_start + (pos - segment.virtual_start),
            SegmentKind::Atom => segment.original_start,
        })
    }
}

struct MappedIndex<'a> {
    virtual_coordinates: EcmaCoordinateIndex,
    original_coordinates: EcmaCoordinateIndex,
    spans: SpanLookup<'a>,
}

#[cfg(test)]
fn ecma_offset_of_line_character(text: &str, line: u32, character: u32) -> u32 {
    EcmaCoordinateIndex::new(text).offset_of_line_character(line, character)
}

#[cfg(test)]
fn ecma_line_character(text: &str, offset: u32) -> (u32, u32) {
    EcmaCoordinateIndex::new(text).line_character(offset)
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
    let mut mapped_indexes = Vec::new();
    let mut index_by_source = BTreeMap::new();
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
        let mapped_index = mapped.map(|(_, original, transformed)| {
            *index_by_source
                .entry(absolute_name.clone())
                .or_insert_with(|| {
                    let index = mapped_indexes.len();
                    mapped_indexes.push(MappedIndex {
                        virtual_coordinates: EcmaCoordinateIndex::new(&transformed.text),
                        original_coordinates: EcmaCoordinateIndex::new(original),
                        spans: SpanLookup::new(transformed.span_map.segments()),
                    });
                    index
                })
        });
        let name = mapped.map_or_else(|| absolute_name, |s| s.0.clone());
        let id = generator.add_source(&name);
        if let Some((_, original, _)) = mapped {
            generator.set_source_content(id, original)?;
        } else if let Some(content) = map["sourcesContent"][index].as_str() {
            generator.set_source_content(id, content)?;
        }
        resolved.push((id, mapped_index));
    }
    let mut decoder = decode_mappings(map["mappings"].as_str().ok_or("Missing mappings")?);
    for mapping in decoder.values() {
        let location = mapping.source_index.get().and_then(|index| {
            let (id, mapped_index) = resolved.get(index as usize)?;
            let line = mapping.source_line?;
            let column = mapping.source_character?;
            if let Some(index) = mapped_index {
                let mapped = &mapped_indexes[*index];
                let offset = mapped
                    .virtual_coordinates
                    .offset_of_line_character(line, column);
                let original_offset = mapped.spans.virtual_to_original(offset)?;
                let (line, column) = mapped.original_coordinates.line_character(original_offset);
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

    #[test]
    fn indexed_coordinates_match_prefix_scans_and_scale_with_source_size() {
        for terminator in ["\n", "\r", "\r\n", "\u{2028}", "\u{2029}"] {
            for count in [64, 128] {
                let text = format!("{}end", format!("a😀é{terminator}").repeat(count));
                let index = EcmaCoordinateIndex::new(&text);
                assert_eq!(index.boundaries.len(), text.chars().count() + 1);
                assert_eq!(index.starts.len(), count + 1);
                for offset in 0..=text.len() {
                    let line = index
                        .starts
                        .partition_point(|&start| start as usize <= offset)
                        - 1;
                    let mut boundary = offset;
                    while !text.is_char_boundary(boundary) {
                        boundary -= 1;
                    }
                    let expected = (
                        line as u32,
                        text[index.starts[line] as usize..boundary]
                            .encode_utf16()
                            .count() as u32,
                    );
                    assert_eq!(index.line_character(offset as u32), expected);
                }
                for line in 0..=(count + 1) as u32 {
                    let selected = (line as usize).min(index.starts.len() - 1);
                    let start = index.starts[selected] as usize;
                    let end = index.content_ends[selected] as usize;
                    for column in 0..=10 {
                        let mut units = 0;
                        let mut expected = end;
                        for (byte, ch) in text[start..end].char_indices() {
                            if units >= column {
                                expected = start + byte;
                                break;
                            }
                            units += ch.len_utf16() as u32;
                        }
                        assert_eq!(
                            index.offset_of_line_character(line, column),
                            expected as u32
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn indexed_spans_keep_first_match_at_touching_and_degenerate_boundaries() {
        let segments = [
            Segment {
                virtual_start: 0,
                virtual_end: 2,
                original_start: 10,
                original_end: 12,
                kind: SegmentKind::Verbatim,
            },
            Segment {
                virtual_start: 2,
                virtual_end: 2,
                original_start: 20,
                original_end: 21,
                kind: SegmentKind::Atom,
            },
            Segment {
                virtual_start: 2,
                virtual_end: 2,
                original_start: 22,
                original_end: 23,
                kind: SegmentKind::Verbatim,
            },
            Segment {
                virtual_start: 2,
                virtual_end: 4,
                original_start: 30,
                original_end: 32,
                kind: SegmentKind::Atom,
            },
            Segment {
                virtual_start: 6,
                virtual_end: 8,
                original_start: 40,
                original_end: 42,
                kind: SegmentKind::Verbatim,
            },
        ];
        let lookup = SpanLookup::new(&segments);
        for pos in 0..=10 {
            let expected = segments
                .iter()
                .find(|s| {
                    s.virtual_start <= pos
                        && (pos < s.virtual_end
                            || (pos == s.virtual_end && s.kind == SegmentKind::Verbatim))
                })
                .map(|s| match s.kind {
                    SegmentKind::Verbatim => s.original_start + pos - s.virtual_start,
                    SegmentKind::Atom => s.original_start,
                });
            assert_eq!(lookup.virtual_to_original(pos), expected, "position {pos}");
        }
        assert_eq!(lookup.virtual_to_original(2), Some(12));
        assert_eq!(lookup.virtual_to_original(4), None);
        assert_eq!(lookup.virtual_to_original(5), None);
    }

    #[test]
    fn named_and_ordinary_sources_keep_their_content_and_location() {
        let original = "export const answer = 42;\n";
        let transformed = nmath_nts::transform_with_plan(
            "/project/src/main.nts",
            original,
            nmath_nts::TransformOptions::default(),
            &nmath_nts::semantic::LoweringPlan::default(),
        );
        let sources = BTreeMap::from([(
            "/project/src/main.ts".into(),
            ("/project/src/main.nts".into(), original.into(), transformed),
        )]);
        let mut emitter = new_generator("main.js", "", "", Default::default());
        let mapped = emitter.add_source("../src/main.ts");
        let ordinary = emitter.add_source("../src/other.ts");
        let name = emitter.add_name("answer");
        emitter
            .add_named_source_mapping(0, 0, mapped, 0, 0, name)
            .unwrap();
        emitter.add_source_mapping(0, 5, ordinary, 3, 7).unwrap();
        let mut map: Value = serde_json::from_str(&emitter.string()).unwrap();
        map["sourcesContent"] = serde_json::json!(["virtual", "ordinary source"]);
        let composed: Value = serde_json::from_str(
            &compose("/project/out/main.js.map", &map.to_string(), &sources).unwrap(),
        )
        .unwrap();
        let mut decoder = decode_mappings(composed["mappings"].as_str().unwrap());
        let entries: Vec<_> = decoder.values().collect();
        assert_eq!(
            (entries[0].source_line, entries[0].source_character),
            (Some(0), Some(0))
        );
        assert_eq!(
            (entries[1].source_line, entries[1].source_character),
            (Some(3), Some(7))
        );
        assert!(entries[0].name_index.get().is_some());
        assert_eq!(composed["sourcesContent"][0], original);
        assert_eq!(composed["sourcesContent"][1], "ordinary source");
        assert_eq!(composed["names"][0], "answer");
    }
}
