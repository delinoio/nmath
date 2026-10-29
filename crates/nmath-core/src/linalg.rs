// SPDX-License-Identifier: Apache-2.0
//! Batched square linear systems, inverse and determinant on CPU.
//! Partial pivoting and double-precision workspace follow NumPy's common-type
//! computation boundary. No conjugation is applied to a linear system.
use crate::{
    buffer::DenseArray,
    dtype::DType,
    error::{Error, ErrorCode, Result},
    memory::MemoryBudget,
    numerical::{C, bytes, overflow, read, size, write, zeros},
    shape::broadcast_shapes,
    sync::CancellationToken,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinalgOp {
    Solve,
    Inv,
    Det,
}
impl LinalgOp {
    pub fn id(self) -> &'static str {
        match self {
            Self::Solve => "solve",
            Self::Inv => "inv",
            Self::Det => "det",
        }
    }
}
fn invalid(message: &str) -> Error {
    Error::new(ErrorCode::ShapeMismatch, message)
}
fn result_dtype(a: DType, b: Option<DType>) -> Result<DType> {
    let mut single = true;
    let mut complex = false;
    for d in std::iter::once(a).chain(b) {
        crate::numerical::numeric_dtype(d)?;
        if d == DType::Float16 {
            return Err(Error::new(
                ErrorCode::DTypeUnsupported,
                "linalg does not support float16. Cast explicitly.",
            ));
        }
        single &= matches!(d, DType::Float32 | DType::Complex64);
        complex |= matches!(d, DType::Complex64 | DType::Complex128);
    }
    Ok(match (single, complex) {
        (true, false) => DType::Float32,
        (false, false) => DType::Float64,
        (true, true) => DType::Complex64,
        (false, true) => DType::Complex128,
    })
}
pub fn infer(
    op: LinalgOp,
    a: &[usize],
    ad: DType,
    b: Option<(&[usize], DType)>,
) -> Result<(Vec<usize>, DType)> {
    if a.len() < 2 || a[a.len() - 2] != a[a.len() - 1] {
        return Err(invalid(
            "linalg requires square matrices on the final two axes.",
        ));
    }
    if op != LinalgOp::Solve && b.is_some() {
        return Err(invalid("Only solve accepts a right-hand side."));
    }
    let n = a[a.len() - 1];
    let dtype = result_dtype(ad, b.map(|(_, d)| d))?;
    let mut shape = a[..a.len() - 2].to_vec();
    match op {
        LinalgOp::Det => {}
        LinalgOp::Inv => shape.extend([n, n]),
        LinalgOp::Solve => {
            let (b, _) = b.ok_or_else(|| invalid("solve requires a right-hand side."))?;
            if b.len() == 1 {
                if b[0] != n {
                    return Err(invalid("solve vector dimension does not match."));
                }
                shape.push(n);
            } else {
                if b.len() < 2 || b[b.len() - 2] != n {
                    return Err(invalid(
                        "solve matrix dimension does not match; only a 1-D right-hand side is a \
                         vector.",
                    ));
                }
                shape = broadcast_shapes(&shape, &b[..b.len() - 2])?;
                shape.extend([n, b[b.len() - 1]]);
            }
        }
    }
    size(&shape)?;
    Ok((shape, dtype))
}
fn batch_index(mut linear: usize, out: &[usize], input: &[usize]) -> usize {
    let mut index = 0;
    let mut stride = 1;
    for axis in (0..out.len()).rev() {
        let coord = linear % out[axis];
        linear /= out[axis];
        if axis + input.len() >= out.len() {
            let d = input[axis + input.len() - out.len()];
            if d != 1 {
                index += coord * stride;
            }
            stride *= d;
        }
    }
    index
}
pub fn execute(
    op: LinalgOp,
    a: &DenseArray,
    b: Option<&DenseArray>,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    token.check()?;
    let (shape, dtype) = infer(
        op,
        &a.shape,
        a.buffer.dtype(),
        b.map(|v| (v.shape.as_slice(), v.buffer.dtype())),
    )?;
    let count = size(&shape)?;
    let charge = budget.reserve(bytes(count, dtype.item_size())?)?;
    let mut output = zeros(dtype, count)?;
    let n = a.shape[a.shape.len() - 1];
    let vector = b.is_some_and(|v| v.shape.len() == 1);
    let cols = if op == LinalgOp::Inv {
        n
    } else {
        b.map_or(0, |v| if vector { 1 } else { *v.shape.last().unwrap() })
    };
    let rank = match op {
        LinalgOp::Det => shape.len(),
        LinalgOp::Inv => shape.len() - 2,
        LinalgOp::Solve => shape.len() - if vector { 1 } else { 2 },
    };
    let batch = &shape[..rank];
    let batches = size(batch)?;
    if batches == 0 {
        return DenseArray::new(shape, output, charge);
    }
    let square = n.checked_mul(n).ok_or_else(overflow)?;
    let rhs_len = n.checked_mul(cols).ok_or_else(overflow)?;
    let _workspace = budget.reserve(bytes(
        square.checked_add(rhs_len).ok_or_else(overflow)?,
        std::mem::size_of::<C>(),
    )?)?;
    let mut lu = vec![C::new(0., 0.); square];
    let mut rhs = vec![C::new(0., 0.); rhs_len];
    for batch_id in 0..batches {
        token.check()?;
        let offset = batch_index(batch_id, batch, &a.shape[..a.shape.len() - 2]) * square;
        for (i, v) in lu.iter_mut().enumerate() {
            if i % 1024 == 0 {
                token.check()?;
            }
            *v = read(&a.buffer, offset + i);
        }
        if let Some(b) = b {
            let offset = if vector {
                0
            } else {
                batch_index(batch_id, batch, &b.shape[..b.shape.len() - 2]) * rhs_len
            };
            for (i, v) in rhs.iter_mut().enumerate() {
                if i % 1024 == 0 {
                    token.check()?;
                }
                *v = read(&b.buffer, offset + i);
            }
        } else if op == LinalgOp::Inv {
            rhs.fill(C::new(0., 0.));
            for i in 0..n {
                rhs[i * n + i] = C::new(1., 0.);
            }
        }
        let mut determinant = C::new(1., 0.);
        for k in 0..n {
            token.check()?;
            let mut pivot = k;
            for i in k + 1..n {
                if lu[i * n + k].norm() > lu[pivot * n + k].norm() {
                    pivot = i;
                }
            }
            if lu[pivot * n + k] == C::new(0., 0.) {
                if op == LinalgOp::Det {
                    determinant = C::new(0., 0.);
                    break;
                }
                return Err(Error::new(
                    ErrorCode::LinalgSingular,
                    "The matrix is singular.",
                ));
            }
            if pivot != k {
                for j in 0..n {
                    lu.swap(k * n + j, pivot * n + j);
                }
                for j in 0..cols {
                    rhs.swap(k * cols + j, pivot * cols + j);
                }
                determinant = -determinant;
            }
            let diagonal = lu[k * n + k];
            determinant *= diagonal;
            for i in k + 1..n {
                token.check()?;
                let factor = lu[i * n + k] / diagonal;
                lu[i * n + k] = factor;
                for j in k + 1..n {
                    let v = lu[k * n + j];
                    lu[i * n + j] -= factor * v;
                }
                for j in 0..cols {
                    let v = rhs[k * cols + j];
                    rhs[i * cols + j] -= factor * v;
                }
            }
        }
        if op == LinalgOp::Det {
            write(&mut output, batch_id, determinant);
            continue;
        }
        for i in (0..n).rev() {
            token.check()?;
            for j in 0..cols {
                let mut v = rhs[i * cols + j];
                for k in i + 1..n {
                    v -= lu[i * n + k] * rhs[k * cols + j];
                }
                rhs[i * cols + j] = v / lu[i * n + i];
            }
        }
        for (i, &v) in rhs.iter().enumerate() {
            write(&mut output, batch_id * rhs_len + i, v);
        }
    }
    token.check()?;
    DenseArray::new(shape, output, charge)
}
