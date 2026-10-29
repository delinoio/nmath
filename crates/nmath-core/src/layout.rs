// SPDX-License-Identifier: Apache-2.0
//! Checked C-order reshape and axis permutation metadata. These operations
//! preserve element values and never cast dtype metadata.
use crate::error::{Error, ErrorCode, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyPolicy {
    IfNeeded,
    Always,
    Never,
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorCode::ShapeMismatch, message)
}

pub fn reshape_shape(size: usize, requested: &[i64]) -> Result<Vec<usize>> {
    let mut inferred = None;
    let mut shape = Vec::with_capacity(requested.len());
    for (axis, &dim) in requested.iter().enumerate() {
        if dim == -1 {
            if inferred.replace(axis).is_some() {
                return Err(invalid("A reshape can infer only one dimension."));
            }
            shape.push(1);
        } else {
            let dim = usize::try_from(dim)
                .map_err(|_| invalid("Reshape dimensions must be nonnegative or -1."))?;
            // Public shape transport uses unsigned 32-bit dimensions.
            if dim > u32::MAX as usize {
                return Err(invalid("A reshape dimension exceeds the supported range."));
            }
            shape.push(dim);
        }
    }
    let known = if shape.contains(&0) {
        0
    } else {
        shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| invalid("The reshape size is too large."))?
    };
    if let Some(axis) = inferred {
        if known == 0 || !size.is_multiple_of(known) {
            return Err(invalid(
                "The inferred reshape dimension is ambiguous or incompatible.",
            ));
        }
        let dim = size / known;
        if dim > u32::MAX as usize {
            return Err(invalid(
                "The inferred reshape dimension exceeds the supported range.",
            ));
        }
        shape[axis] = dim;
    } else if known != size {
        return Err(invalid("A reshape must preserve the number of elements."));
    }
    // Strides are signed. Bound even zero-sized layouts before constructing
    // metadata so empty axes cannot hide overflowing stride products.
    checked_strides(&shape)?;
    Ok(shape)
}

pub fn checked_strides(shape: &[usize]) -> Result<Vec<isize>> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1isize;
    for (axis, &dim) in shape.iter().enumerate().rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(dim.max(1) as isize)
            .ok_or_else(|| invalid("The array strides are too large."))?;
    }
    Ok(strides)
}

pub fn permutation(rank: usize, axes: Option<&[i64]>) -> Result<Vec<usize>> {
    let Some(axes) = axes else {
        return Ok((0..rank).rev().collect());
    };
    if axes.len() != rank {
        return Err(invalid(
            "Transpose axes must include every input axis exactly once.",
        ));
    }
    let mut result = Vec::with_capacity(rank);
    for &axis in axes {
        let axis = if axis < 0 {
            axis.checked_add(rank as i64)
        } else {
            Some(axis)
        }
        .and_then(|v| usize::try_from(v).ok())
        .filter(|&v| v < rank)
        .ok_or_else(|| invalid("A transpose axis is out of bounds."))?;
        if result.contains(&axis) {
            return Err(invalid("Transpose axes cannot repeat."));
        }
        result.push(axis);
    }
    Ok(result)
}

/// Find strides for a C-order reshape without moving elements. Adjacent axes
/// may merge only when their signed strides describe one regular run. This
/// also handles stepped and negative-stride views, without an offsets buffer.
pub fn reshape_strides(old: &[usize], strides: &[isize], new: &[usize]) -> Option<Vec<isize>> {
    if old.contains(&0) {
        return checked_strides(new).ok();
    }
    let axes: Vec<_> = old
        .iter()
        .copied()
        .zip(strides.iter().copied())
        .filter(|&(dim, _)| dim != 1)
        .collect();
    let mut result = vec![0isize; new.len()];
    let (mut oi, mut ni) = (0, 0);
    while oi < axes.len() && ni < new.len() {
        let (mut oj, mut nj) = (oi + 1, ni + 1);
        let (mut old_count, mut new_count) = (axes[oi].0, new[ni]);
        while old_count != new_count {
            if old_count < new_count {
                old_count = old_count.checked_mul(axes.get(oj)?.0)?;
                oj += 1;
            } else {
                new_count = new_count.checked_mul(*new.get(nj)?)?;
                nj += 1;
            }
        }
        for k in oi..oj - 1 {
            if axes[k].1 != axes[k + 1].1.checked_mul(axes[k + 1].0 as isize)? {
                return None;
            }
        }
        result[nj - 1] = axes[oj - 1].1;
        for k in (ni..nj - 1).rev() {
            result[k] = result[k + 1].checked_mul(new[k + 1] as isize)?;
        }
        oi = oj;
        ni = nj;
    }
    if oi != axes.len() || new[ni..].iter().any(|&d| d != 1) {
        return None;
    }
    let last = if ni > 0 { result[ni - 1] } else { 1 };
    result[ni..].fill(last);
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inferred_empty_scalar_and_invalid_shapes() {
        assert_eq!(reshape_shape(6, &[2, -1]).unwrap(), [2, 3]);
        assert_eq!(reshape_shape(0, &[-1]).unwrap(), [0]);
        assert!(reshape_shape(1, &[]).unwrap().is_empty());
        for shape in [vec![0, -1], vec![-1, -1], vec![-2], vec![2, 4]] {
            assert!(reshape_shape(6, &shape).is_err());
        }
        assert!(reshape_shape(0, &[0, -1]).is_err());
        assert!(reshape_shape(0, &[0, u32::MAX as i64, u32::MAX as i64]).is_err());
    }
    #[test]
    fn regular_signed_runs_and_transpose_copy_boundary() {
        assert_eq!(reshape_strides(&[2, 3], &[6, 2], &[3, 2]), Some(vec![4, 2]));
        assert_eq!(reshape_strides(&[2, 3], &[-3, -1], &[6]), Some(vec![-1]));
        assert_eq!(reshape_strides(&[3, 2], &[1, 3], &[6]), None);
        assert_eq!(
            reshape_strides(&[2, 1, 3], &[3, 0, 1], &[6, 1]),
            Some(vec![1, 1])
        );
        assert_eq!(reshape_strides(&[], &[], &[1, 1]), Some(vec![1, 1]));
    }
    #[test]
    fn transpose_permutations() {
        assert_eq!(permutation(3, None).unwrap(), [2, 1, 0]);
        assert_eq!(permutation(3, Some(&[-1, 0, 1])).unwrap(), [2, 0, 1]);
        assert!(permutation(0, Some(&[])).unwrap().is_empty());
        for axes in [&[0, 0][..], &[0][..], &[-3, 1][..], &[0, 2][..]] {
            assert!(permutation(2, Some(axes)).is_err());
        }
    }
}
