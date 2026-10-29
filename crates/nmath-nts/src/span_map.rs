//! A span map between the original `.nts` text and the lowered (virtual)
//! TypeScript text.
//!
//! The shape follows the tsrs content-mapper protocol (`tsrs_spanmap`):
//! segments are ordered and disjoint in virtual space, positions are absolute
//! UTF-8 byte offsets, and virtual positions outside every segment are
//! synthesized.

/// How positions inside a segment relate the virtual span to the original span.
///
/// The numeric values are the tsrs span-map kinds and are part of the
/// content-mapper protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum SegmentKind {
    /// Length-preserving: interior positions map 1:1.
    Verbatim = 0,
    /// The virtual span maps to the original span as a whole; interior
    /// positions clamp to its endpoints.
    Atom = 1,
}

/// One mapped range: virtual `[virtual_start, virtual_end)` to original
/// `[original_start, original_end)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Segment {
    pub virtual_start: u32,
    pub virtual_end: u32,
    pub original_start: u32,
    pub original_end: u32,
    pub kind: SegmentKind,
}

/// An ordered, disjoint (in virtual space) list of segments.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpanMap {
    segments: Vec<Segment>,
}

impl SpanMap {
    /// The identity map over a text of `len` bytes.
    pub fn identity(len: u32) -> SpanMap {
        let mut map = SpanMap::default();
        map.push(Segment {
            virtual_start: 0,
            virtual_end: len,
            original_start: 0,
            original_end: len,
            kind: SegmentKind::Verbatim,
        });
        map
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Appends a segment. Empty segments are dropped and adjacent verbatim
    /// segments are merged.
    pub(crate) fn push(&mut self, segment: Segment) {
        if segment.virtual_start == segment.virtual_end
            && segment.original_start == segment.original_end
        {
            return;
        }
        if let Some(last) = self.segments.last_mut()
            && segment.kind == SegmentKind::Verbatim
            && last.kind == SegmentKind::Verbatim
            && last.virtual_end == segment.virtual_start
            && last.original_end == segment.original_start
        {
            last.virtual_end = segment.virtual_end;
            last.original_end = segment.original_end;
            return;
        }
        debug_assert!(
            self.segments
                .last()
                .is_none_or(|last| last.virtual_end <= segment.virtual_start)
        );
        self.segments.push(segment);
    }

    /// The content-mapper protocol tuples: `[virtualStart, virtualLength,
    /// originalStart, originalLength, kind]`.
    pub fn to_tuples(&self) -> Vec<[i64; 5]> {
        self.segments
            .iter()
            .map(|s| {
                [
                    i64::from(s.virtual_start),
                    i64::from(s.virtual_end - s.virtual_start),
                    i64::from(s.original_start),
                    i64::from(s.original_end - s.original_start),
                    s.kind as i64,
                ]
            })
            .collect()
    }

    /// Maps a virtual position back to the original text. `None` means the
    /// position is synthesized.
    ///
    /// A position at the end of a verbatim segment maps to the end of its
    /// original span.
    pub fn virtual_to_original(&self, pos: u32) -> Option<u32> {
        let segment = self.segments.iter().find(|s| {
            s.virtual_start <= pos
                && (pos < s.virtual_end
                    || (pos == s.virtual_end && s.kind == SegmentKind::Verbatim))
        })?;
        Some(match segment.kind {
            SegmentKind::Verbatim => segment.original_start + (pos - segment.virtual_start),
            SegmentKind::Atom => segment.original_start,
        })
    }

    /// Maps an original position to its verbatim virtual position, if the
    /// original text at `pos` was copied.
    pub fn original_to_virtual(&self, pos: u32) -> Option<u32> {
        // Prefer the segment that contains `pos`; a segment that ends at `pos`
        // only matches the end of the text or a position followed by
        // synthesized text.
        let verbatim = || {
            self.segments
                .iter()
                .filter(|s| s.kind == SegmentKind::Verbatim)
        };
        verbatim()
            .find(|s| s.original_start <= pos && pos < s.original_end)
            .or_else(|| verbatim().find(|s| s.original_end == pos))
            .map(|s| s.virtual_start + (pos - s.original_start))
    }
}
