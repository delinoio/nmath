//! Shapes, broadcasting and index selectors.

use crate::error::{Error, ErrorCode, Result};

/// NumPy broadcasting of two shapes.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![0; rank];
    for i in 0..rank {
        let da = dim_from_end(a, rank - 1 - i);
        let db = dim_from_end(b, rank - 1 - i);
        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            (x, y) => {
                return Err(Error::new(
                    ErrorCode::ShapeBroadcast,
                    format!(
                        "Shapes {a:?} and {b:?} cannot broadcast: dimension {x} differs from {y}."
                    ),
                ));
            }
        };
    }
    Ok(out)
}

fn dim_from_end(shape: &[usize], from_end: usize) -> usize {
    if from_end < shape.len() {
        shape[shape.len() - 1 - from_end]
    } else {
        1
    }
}

/// C-order element strides.
pub fn contiguous_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut acc = 1isize;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i].max(1) as isize;
    }
    strides
}

/// Strides that read a `from`-shaped C-order buffer as if broadcast to `to`.
pub fn broadcast_strides(from: &[usize], to: &[usize]) -> Vec<isize> {
    let base = contiguous_strides(from);
    let offset = to.len() - from.len();
    (0..to.len())
        .map(|i| {
            if i < offset || from[i - offset] == 1 {
                0
            } else {
                base[i - offset]
            }
        })
        .collect()
}

/// Calls `f(linear_offset)` for every element of `shape` in C order, reading
/// with `strides`.
pub fn for_each_offset(shape: &[usize], strides: &[isize], start: isize, mut f: impl FnMut(usize)) {
    let size: usize = shape.iter().product();
    if size == 0 {
        return;
    }
    let mut index = vec![0usize; shape.len()];
    let mut offset = start;
    for _ in 0..size {
        f(offset as usize);
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            offset += strides[axis];
            if index[axis] < shape[axis] {
                break;
            }
            offset -= strides[axis] * shape[axis] as isize;
            index[axis] = 0;
        }
    }
}

/// One axis selector. `stop` is excluded; `None` omits a bound.
#[derive(Clone, Debug, PartialEq)]
pub enum Selector {
    Index(i64),
    Slice {
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    },
    Indices(Vec<i64>),
    Mask(Vec<bool>),
}

impl Selector {
    pub fn is_basic(&self) -> bool {
        matches!(self, Selector::Index(_) | Selector::Slice { .. })
    }
}

pub fn normalize_index(index: i64, len: usize, axis: usize) -> Result<usize> {
    let resolved = if index < 0 { index + len as i64 } else { index };
    if resolved < 0 || resolved >= len as i64 {
        return Err(Error::new(
            ErrorCode::IndexOutOfBounds,
            format!("Index {index} is out of bounds for axis {axis} with size {len}."),
        ));
    }
    Ok(resolved as usize)
}

/// Resolved Python slice: `(start, step, length)`.
pub fn resolve_slice(
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
    len: usize,
) -> Result<(isize, isize, usize)> {
    let step = step.unwrap_or(1);
    if step == 0 {
        return Err(Error::new(
            ErrorCode::IndexInvalid,
            "A slice step cannot be zero.",
        ));
    }
    let len = len as i64;
    let clamp = |v: i64, lo: i64, hi: i64| v.max(lo).min(hi);
    let norm = |v: i64| if v < 0 { v + len } else { v };
    let (start, stop) = if step > 0 {
        (
            start.map_or(0, |s| clamp(norm(s), 0, len)),
            stop.map_or(len, |s| clamp(norm(s), 0, len)),
        )
    } else {
        (
            start.map_or(len - 1, |s| clamp(norm(s), -1, len - 1)),
            stop.map_or(-1, |s| clamp(norm(s), -1, len - 1)),
        )
    };
    // Ceiling division in i128: `step` may be `i64::MAX` or `i64::MIN`, which
    // overflow the textbook i64 formula and its negation.
    let (distance, magnitude) = if step > 0 {
        (i128::from(stop) - i128::from(start), i128::from(step))
    } else {
        (i128::from(start) - i128::from(stop), -i128::from(step))
    };
    let count = if distance > 0 {
        (distance + magnitude - 1) / magnitude
    } else {
        0
    };
    Ok((start as isize, step as isize, count as usize))
}

/// Output shape of indexing `shape` with `selectors`. Supports any number of
/// basic selectors and at most one integer-array or mask selector.
pub fn index_shape(shape: &[usize], selectors: &[Selector]) -> Result<Vec<usize>> {
    validate_selectors(shape, selectors)?;
    let mut out = Vec::new();
    for (axis, &len) in shape.iter().enumerate() {
        match selectors.get(axis) {
            None => out.push(len),
            Some(Selector::Index(i)) => {
                normalize_index(*i, len, axis)?;
            }
            Some(Selector::Slice { start, stop, step }) => {
                out.push(resolve_slice(*start, *stop, *step, len)?.2);
            }
            Some(Selector::Indices(ix)) => {
                for &i in ix {
                    normalize_index(i, len, axis)?;
                }
                out.push(ix.len());
            }
            Some(Selector::Mask(m)) => out.push(m.iter().filter(|&&b| b).count()),
        }
    }
    Ok(out)
}

pub fn validate_selectors(shape: &[usize], selectors: &[Selector]) -> Result<()> {
    if selectors.len() > shape.len() {
        return Err(Error::new(
            ErrorCode::IndexInvalid,
            format!(
                "Too many indices: the array has {} dimensions but {} were indexed.",
                shape.len(),
                selectors.len()
            ),
        ));
    }
    let advanced = selectors.iter().filter(|s| !s.is_basic()).count();
    if advanced > 1 {
        // Gap: NumPy combines several advanced selectors by broadcasting them
        // together. This build supports one advanced selector per
        // indexing operation.
        return Err(Error::new(
            ErrorCode::IndexInvalid,
            "This build supports at most one integer-array or mask selector per indexing \
             operation.",
        ));
    }
    for (axis, sel) in selectors.iter().enumerate() {
        if let Selector::Mask(m) = sel
            && m.len() != shape[axis]
        {
            return Err(Error::new(
                ErrorCode::IndexInvalid,
                format!(
                    "A boolean mask of length {} does not match axis {axis} with size {}.",
                    m.len(),
                    shape[axis]
                ),
            ));
        }
    }
    Ok(())
}

/// For every output element in C order, the linear element offset inside a
/// source array described by `(shape, strides, base)`.
pub fn index_offsets(
    shape: &[usize],
    strides: &[isize],
    base: isize,
    selectors: &[Selector],
) -> Result<Vec<usize>> {
    validate_selectors(shape, selectors)?;
    // Per axis, the list of source positions it contributes, or None for a
    // dropped axis.
    let mut axes: Vec<(Vec<isize>, bool)> = Vec::with_capacity(shape.len());
    for (axis, &len) in shape.iter().enumerate() {
        let positions: (Vec<isize>, bool) = match selectors.get(axis) {
            None => ((0..len as isize).collect(), true),
            Some(Selector::Index(i)) => (vec![normalize_index(*i, len, axis)? as isize], false),
            Some(Selector::Slice { start, stop, step }) => {
                let (s, st, n) = resolve_slice(*start, *stop, *step, len)?;
                ((0..n as isize).map(|k| s + k * st).collect(), true)
            }
            Some(Selector::Indices(ix)) => (
                ix.iter()
                    .map(|&i| normalize_index(i, len, axis).map(|v| v as isize))
                    .collect::<Result<_>>()?,
                true,
            ),
            Some(Selector::Mask(m)) => (
                m.iter()
                    .enumerate()
                    .filter(|(_, b)| **b)
                    .map(|(i, _)| i as isize)
                    .collect(),
                true,
            ),
        };
        axes.push(positions);
    }
    let mut out = Vec::new();
    let mut cursor = vec![0usize; axes.len()];
    if axes.iter().any(|(p, _)| p.is_empty()) {
        return Ok(out);
    }
    loop {
        let offset: isize = base
            + axes
                .iter()
                .zip(&cursor)
                .zip(strides)
                .map(|(((p, _), &c), &s)| p[c] * s)
                .sum::<isize>();
        out.push(offset as usize);
        let mut axis = axes.len();
        loop {
            if axis == 0 {
                return Ok(out);
            }
            axis -= 1;
            cursor[axis] += 1;
            if cursor[axis] < axes[axis].0.len() {
                break;
            }
            cursor[axis] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcasting_matches_numpy() {
        assert_eq!(broadcast_shapes(&[3, 1], &[4]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[], &[2, 2]).unwrap(), vec![2, 2]);
        assert_eq!(broadcast_shapes(&[0, 1], &[1, 5]).unwrap(), vec![0, 5]);
        let err = broadcast_shapes(&[3], &[4]).unwrap_err();
        assert_eq!(err.code, ErrorCode::ShapeBroadcast);
    }

    #[test]
    fn slices_match_python() {
        // range(10)[2:8:3] == [2, 5]
        assert_eq!(
            resolve_slice(Some(2), Some(8), Some(3), 10).unwrap(),
            (2, 3, 2)
        );
        // range(10)[::-1] has 10 elements starting at 9
        assert_eq!(
            resolve_slice(None, None, Some(-1), 10).unwrap(),
            (9, -1, 10)
        );
        // range(5)[-2:] == [3, 4]
        assert_eq!(resolve_slice(Some(-2), None, None, 5).unwrap(), (3, 1, 2));
        // range(5)[10:20] is empty
        assert_eq!(resolve_slice(Some(10), Some(20), None, 5).unwrap().2, 0);
        // range(3)[::sys.maxsize] == [0] and range(3)[::-sys.maxsize - 1] ==
        // [2]
        assert_eq!(
            resolve_slice(None, None, Some(i64::MAX), 3).unwrap(),
            (0, i64::MAX as isize, 1)
        );
        assert_eq!(
            resolve_slice(None, None, Some(i64::MIN), 3).unwrap(),
            (2, i64::MIN as isize, 1)
        );
        assert!(resolve_slice(None, None, Some(0), 5).is_err());
    }

    #[test]
    fn index_offsets_cover_all_selector_kinds() {
        let shape = [3, 4];
        let strides = contiguous_strides(&shape);
        let take = |sel: Vec<Selector>| index_offsets(&shape, &strides, 0, &sel).unwrap();
        assert_eq!(take(vec![Selector::Index(-1)]), vec![8, 9, 10, 11]);
        assert_eq!(
            take(vec![
                Selector::Slice {
                    start: None,
                    stop: None,
                    step: Some(2)
                },
                Selector::Index(1)
            ]),
            vec![1, 9]
        );
        assert_eq!(take(vec![Selector::Indices(vec![2, 0])]).len(), 8);
        assert_eq!(
            take(vec![Selector::Mask(vec![true, false, true])]),
            vec![0, 1, 2, 3, 8, 9, 10, 11]
        );
        let err = index_offsets(&shape, &strides, 0, &[Selector::Index(3)]).unwrap_err();
        assert_eq!(err.code, ErrorCode::IndexOutOfBounds);
    }

    #[test]
    fn for_each_offset_handles_broadcast_strides() {
        let strides = broadcast_strides(&[3], &[2, 3]);
        let mut seen = Vec::new();
        for_each_offset(&[2, 3], &strides, 0, |o| seen.push(o));
        assert_eq!(seen, vec![0, 1, 2, 0, 1, 2]);
    }
}
