//! Shapes, broadcasting and index selectors.

use crate::{
    error::{Error, ErrorCode, Result},
    memory::{MemoryBudget, Reservation},
};

pub(crate) fn index_scratch_overflow() -> Error {
    Error::new(
        ErrorCode::MemoryBudgetExceeded,
        "Index scratch exceeds the addressable allocation size.",
    )
}

fn scratch_bytes(count: usize, element_size: usize) -> Result<usize> {
    count
        .checked_mul(element_size)
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(index_scratch_overflow)
}

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

/// NumPy matrix product shape: broadcast leading axes and remove only the
/// axes introduced to promote vector operands. Scalars are never accepted.
pub fn matmul_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let Some(&k) = a.last() else {
        return Err(matmul_scalar_error());
    };
    let k2 = match b {
        [] => return Err(matmul_scalar_error()),
        [k] => *k,
        _ => b[b.len() - 2],
    };
    if k != k2 {
        return Err(Error::new(
            ErrorCode::ShapeMismatch,
            "matmul core dimensions do not match. Use operands with equal contraction dimensions.",
        ));
    }
    let mut shape = broadcast_shapes(
        &a[..a.len().saturating_sub(2)],
        &b[..b.len().saturating_sub(2)],
    )?;
    if a.len() > 1 {
        shape.push(a[a.len() - 2]);
    }
    if b.len() > 1 {
        shape.push(b[b.len() - 1]);
    }
    Ok(shape)
}

fn matmul_scalar_error() -> Error {
    Error::new(
        ErrorCode::ShapeMismatch,
        "matmul does not accept scalar operands. Use arrays with rank at least one.",
    )
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

/// The axis of the integer-array or mask selector when NumPy moves its
/// dimension to the front of the result.
///
/// With an array selector present, integer scalars also count as advanced
/// indexes. When a slice or omitted axis separates the advanced indexes, NumPy
/// places the advanced dimension first; otherwise it stays in place.
fn separated_advanced_axis(selectors: &[Selector]) -> Option<usize> {
    let array_axis = selectors.iter().position(|s| !s.is_basic())?;
    let mut first = None;
    let mut last = 0;
    let mut count = 0;
    for (axis, selector) in selectors.iter().enumerate() {
        if !matches!(selector, Selector::Slice { .. }) {
            first.get_or_insert(axis);
            last = axis;
            count += 1;
        }
    }
    (last - first.expect("an array selector is present") + 1 != count).then_some(array_axis)
}

/// Output shape of indexing `shape` with `selectors`. Integer-array and mask
/// selectors broadcast together into one advanced dimension.
/// Length of the advanced block: the broadcast length of every integer-array
/// and mask selector, with masks counting their true entries. All selectors
/// are one-dimensional, so the block is one output dimension.
fn advanced_length(selectors: &[Selector]) -> Result<Option<usize>> {
    let mut block: Option<usize> = None;
    for selector in selectors {
        let len = match selector {
            Selector::Indices(ix) => ix.len(),
            Selector::Mask(m) => m.iter().filter(|&&b| b).count(),
            _ => continue,
        };
        block = Some(match block {
            None => len,
            Some(b) if b == len || len == 1 => b,
            Some(1) => len,
            Some(b) => {
                return Err(Error::new(
                    ErrorCode::IndexInvalid,
                    format!(
                        "Indexing arrays of lengths {b} and {len} cannot be broadcast together."
                    ),
                ));
            }
        });
    }
    Ok(block)
}

pub fn index_shape(shape: &[usize], selectors: &[Selector]) -> Result<Vec<usize>> {
    validate_selectors(shape, selectors)?;
    let separated = separated_advanced_axis(selectors);
    let block = advanced_length(selectors)?;
    let first_array = selectors.iter().position(|s| !s.is_basic());
    let mut out = Vec::new();
    for (axis, &len) in shape.iter().enumerate() {
        match selectors.get(axis) {
            Some(Selector::Indices(ix)) if first_array != Some(axis) => {
                for &i in ix {
                    normalize_index(i, len, axis)?;
                }
                continue;
            }
            Some(Selector::Mask(_)) if first_array != Some(axis) => continue,
            Some(Selector::Indices(ix)) => {
                for &i in ix {
                    normalize_index(i, len, axis)?;
                }
                out.push(block.expect("an array selector is present"));
            }
            Some(Selector::Mask(_)) => out.push(block.expect("an array selector is present")),
            None => out.push(len),
            Some(Selector::Index(i)) => {
                normalize_index(*i, len, axis)?;
            }
            Some(Selector::Slice { start, stop, step }) => {
                out.push(resolve_slice(*start, *stop, *step, len)?.2);
            }
        }
        if separated == Some(axis) {
            let len = out.pop().expect("the advanced selector pushed a dimension");
            out.insert(0, len);
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
    index_offsets_inner(shape, strides, base, selectors, None).map(|offsets| offsets.values)
}

pub(crate) struct BudgetedOffsets {
    pub values: Vec<usize>,
    // Field drop order releases the vector before its charge.
    _reservation: Reservation,
}

pub(crate) fn index_offsets_budgeted(
    shape: &[usize],
    strides: &[isize],
    base: isize,
    selectors: &[Selector],
    budget: &MemoryBudget,
) -> Result<BudgetedOffsets> {
    index_offsets_inner(shape, strides, base, selectors, Some(budget))
}

fn index_offsets_inner(
    shape: &[usize],
    strides: &[isize],
    base: isize,
    selectors: &[Selector],
    budget: Option<&MemoryBudget>,
) -> Result<BudgetedOffsets> {
    validate_selectors(shape, selectors)?;
    // Per axis, the source positions it contributes; a dropped integer axis
    // contributes exactly one.
    let separated = separated_advanced_axis(selectors);
    let first_array = selectors.iter().position(|s| !s.is_basic());
    let advanced_count = selectors.iter().filter(|s| !s.is_basic()).count();
    let block_len = advanced_length(selectors)?.unwrap_or(0);
    let mut position_count = 0usize;
    for (axis, &len) in shape.iter().enumerate() {
        let count = match selectors.get(axis) {
            None => len,
            Some(Selector::Index(i)) => {
                normalize_index(*i, len, axis)?;
                1
            }
            Some(Selector::Slice { start, stop, step }) => {
                resolve_slice(*start, *stop, *step, len)?.2
            }
            Some(Selector::Indices(ix)) => ix.len(),
            Some(Selector::Mask(mask)) => mask.iter().filter(|&&selected| selected).count(),
        };
        position_count = position_count
            .checked_add(count)
            .ok_or_else(index_scratch_overflow)?;
    }
    let axis_bytes = scratch_bytes(shape.len(), std::mem::size_of::<(Vec<isize>, isize)>())?;
    let array_bytes = scratch_bytes(advanced_count, std::mem::size_of::<(Vec<isize>, isize)>())?;
    let cursor_bytes = scratch_bytes(shape.len(), std::mem::size_of::<usize>())?;
    let working_bytes = scratch_bytes(position_count, std::mem::size_of::<isize>())?
        .checked_add(scratch_bytes(block_len, std::mem::size_of::<isize>())?)
        .and_then(|n| n.checked_add(axis_bytes))
        .and_then(|n| n.checked_add(array_bytes))
        .and_then(|n| n.checked_add(cursor_bytes))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(index_scratch_overflow)?;
    let _working_charge = match budget {
        Some(budget) => budget.reserve(working_bytes as u64)?,
        None => Reservation::none(),
    };
    let mut arrays: Vec<(Vec<isize>, isize)> = Vec::with_capacity(advanced_count);
    let mut block_slot = None;
    let mut axes: Vec<(Vec<isize>, isize)> = Vec::with_capacity(shape.len());
    for (axis, &len) in shape.iter().enumerate() {
        let positions: Vec<isize> = match selectors.get(axis) {
            None => (0..len as isize).collect(),
            Some(Selector::Index(i)) => vec![normalize_index(*i, len, axis)? as isize],
            Some(Selector::Slice { start, stop, step }) => {
                let (s, st, n) = resolve_slice(*start, *stop, *step, len)?;
                (0..n as isize).map(|k| s + k * st).collect()
            }
            Some(Selector::Indices(ix)) => {
                let mut positions = Vec::with_capacity(ix.len());
                for &index in ix {
                    positions.push(normalize_index(index, len, axis)? as isize);
                }
                positions
            }
            Some(Selector::Mask(m)) => {
                let mut positions = Vec::with_capacity(m.iter().filter(|&&b| b).count());
                positions.extend(
                    m.iter()
                        .enumerate()
                        .filter(|(_, b)| **b)
                        .map(|(i, _)| i as isize),
                );
                positions
            }
        };
        if matches!(
            selectors.get(axis),
            Some(Selector::Indices(_) | Selector::Mask(_))
        ) {
            arrays.push((positions, strides[axis]));
            if first_array != Some(axis) {
                continue;
            }
            // Placeholder for the combined advanced block, filled below.
            let entry = (Vec::new(), 1);
            if separated == Some(axis) {
                axes.insert(0, entry);
                block_slot = Some(0);
            } else {
                block_slot = Some(axes.len());
                axes.push(entry);
            }
            continue;
        }
        axes.push((positions, strides[axis]));
    }
    if let Some(slot) = block_slot {
        // Broadcast the array selectors together: each block entry is the sum
        // of the selected offsets on every advanced axis.
        let len = advanced_length(selectors)?.expect("an array selector is present");
        let mut positions = Vec::with_capacity(len);
        for k in 0..len {
            positions.push(
                arrays
                    .iter()
                    .map(|(p, s)| p[if p.len() == 1 { 0 } else { k }] * s)
                    .sum(),
            );
        }
        axes[slot].0 = positions;
    }
    let out_len = axes.iter().try_fold(1usize, |n, (positions, _)| {
        n.checked_mul(positions.len())
            .ok_or_else(index_scratch_overflow)
    })?;
    let out_bytes = scratch_bytes(out_len, std::mem::size_of::<usize>())?;
    let out_charge = match budget {
        Some(budget) => budget.reserve(out_bytes as u64)?,
        None => Reservation::none(),
    };
    let mut out = Vec::with_capacity(out_len);
    let mut cursor = vec![0usize; axes.len()];
    if axes.iter().any(|(p, _)| p.is_empty()) {
        return Ok(BudgetedOffsets {
            values: out,
            _reservation: out_charge,
        });
    }
    loop {
        let offset: isize = base
            + axes
                .iter()
                .zip(&cursor)
                .map(|((p, s), &c)| p[c] * s)
                .sum::<isize>();
        out.push(offset as usize);
        let mut axis = axes.len();
        loop {
            if axis == 0 {
                return Ok(BudgetedOffsets {
                    values: out,
                    _reservation: out_charge,
                });
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
    use crate::memory::Device;

    #[test]
    fn budgeted_offsets_reject_unaddressable_capacity_before_allocation() {
        let budget = MemoryBudget::new(Device::Cpu, u64::MAX);
        let result = index_offsets_budgeted(&[usize::MAX], &[1], 0, &[], &budget);
        assert!(matches!(
            result,
            Err(Error {
                code: ErrorCode::MemoryBudgetExceeded,
                ..
            })
        ));
        assert_eq!(budget.usage().used, 0);
    }

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
    fn several_array_selectors_broadcast_into_one_block() {
        let shape = [3, 4];
        let strides = contiguous_strides(&shape);
        let both = vec![Selector::Indices(vec![0, 2]), Selector::Indices(vec![1, 3])];
        assert_eq!(index_shape(&shape, &both).unwrap(), vec![2]);
        assert_eq!(
            index_offsets(&shape, &strides, 0, &both).unwrap(),
            vec![1, 11]
        );
        let broadcast = vec![
            Selector::Indices(vec![2]),
            Selector::Mask(vec![true, false, true, true]),
        ];
        assert_eq!(
            index_offsets(&shape, &strides, 0, &broadcast).unwrap(),
            vec![8, 10, 11]
        );
        let mismatch = vec![
            Selector::Indices(vec![0, 1]),
            Selector::Indices(vec![0, 1, 2]),
        ];
        assert_eq!(
            index_shape(&shape, &mismatch).unwrap_err().code,
            ErrorCode::IndexInvalid
        );
        let empty = vec![
            Selector::Mask(vec![false, false, false]),
            Selector::Indices(vec![1]),
        ];
        assert_eq!(index_shape(&shape, &empty).unwrap(), vec![0]);
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
    fn separated_advanced_indexes_move_to_the_front() {
        // NumPy 2.5.4: a = np.arange(24).reshape(2, 3, 4)
        let shape = [2, 3, 4];
        let strides = contiguous_strides(&shape);
        let all = Selector::Slice {
            start: None,
            stop: None,
            step: None,
        };
        let check = |sel: Vec<Selector>, want_shape: Vec<usize>, want: Vec<usize>| {
            assert_eq!(index_shape(&shape, &sel).unwrap(), want_shape);
            assert_eq!(index_offsets(&shape, &strides, 0, &sel).unwrap(), want);
        };
        // a[0, :, [0, 1]] has shape (2, 3): the advanced dimension goes first.
        check(
            vec![
                Selector::Index(0),
                all.clone(),
                Selector::Indices(vec![0, 1]),
            ],
            vec![2, 3],
            vec![0, 4, 8, 1, 5, 9],
        );
        // a[[1, 0], :, 2] has shape (2, 3).
        check(
            vec![
                Selector::Indices(vec![1, 0]),
                all.clone(),
                Selector::Index(2),
            ],
            vec![2, 3],
            vec![14, 18, 22, 2, 6, 10],
        );
        // a[:, 1, [0, 3]] keeps adjacent advanced indexes in place: (2, 2).
        check(
            vec![
                all.clone(),
                Selector::Index(1),
                Selector::Indices(vec![0, 3]),
            ],
            vec![2, 2],
            vec![4, 7, 16, 19],
        );
        // a[0, :, 1] has no array selector, so integers stay basic: (3,).
        check(
            vec![Selector::Index(0), all, Selector::Index(1)],
            vec![3],
            vec![1, 5, 9],
        );
    }

    #[test]
    fn for_each_offset_handles_broadcast_strides() {
        let strides = broadcast_strides(&[3], &[2, 3]);
        let mut seen = Vec::new();
        for_each_offset(&[2, 3], &strides, 0, |o| seen.push(o));
        assert_eq!(seen, vec![0, 1, 2, 0, 1, 2]);
    }
}
