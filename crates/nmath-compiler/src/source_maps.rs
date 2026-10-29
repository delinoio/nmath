//! Compose the external emitter's UTF-16 maps with the transform's byte maps.

use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
};

use nmath_nts::TransformOutput;
use serde_json::Value;
use tsrs_sourcemap::{decode_mappings, new_generator};

pub(crate) type Sources = BTreeMap<String, (String, String, TransformOutput)>;

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
                let offset = nmath_nts::offset_of_line_character(&transformed.text, line, column);
                let original_offset = transformed.span_map.virtual_to_original(offset)?;
                let (line, column) = nmath_nts::line_column(original, original_offset);
                Some((*id, line - 1, column - 1))
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
    use tsrs_sourcemap::decode_mappings;

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
}
