//! CPU kernels over dense arrays.

use crate::{
    buffer::{Buffer, DenseArray},
    dtype::{DType, DTypeKind},
    error::{Error, ErrorCode, Result},
    memory::MemoryBudget,
    shape::{broadcast_shapes, broadcast_strides, for_each_offset},
    sync::CancellationToken,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
}

impl BinaryOp {
    pub fn id(self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Mod => "mod",
            BinaryOp::Pow => "pow",
        }
    }

    pub fn from_id(id: &str) -> Result<Self> {
        [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Mod,
            BinaryOp::Pow,
        ]
        .into_iter()
        .find(|op| op.id() == id)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::IndexInvalid,
                format!("Unknown binary operator '{id}'."),
            )
        })
    }

    /// Result dtype given the promoted operand dtype.
    pub fn result_dtype(self, promoted: DType) -> Result<DType> {
        if promoted == DType::Bool && matches!(self, BinaryOp::Sub | BinaryOp::Mod | BinaryOp::Pow)
        {
            return Err(Error::new(
                ErrorCode::DTypeUnsupported,
                format!(
                    "The '{}' operator is not supported for bool operands.",
                    self.id()
                ),
            ));
        }
        Ok(match self {
            BinaryOp::Div => promoted.true_divide(),
            _ => promoted,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Pos,
    Neg,
}

impl UnaryOp {
    pub fn id(self) -> &'static str {
        match self {
            UnaryOp::Pos => "pos",
            UnaryOp::Neg => "neg",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Mean,
    Min,
    Max,
}

impl ReduceOp {
    pub fn id(self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Mean => "mean",
            ReduceOp::Min => "min",
            ReduceOp::Max => "max",
        }
    }

    pub fn result_dtype(self, input: DType) -> DType {
        match self {
            ReduceOp::Sum => input.sum_dtype(),
            ReduceOp::Mean => input.true_divide(),
            ReduceOp::Min | ReduceOp::Max => input,
        }
    }
}

/// Elements processed between cancellation checks.
const CHUNK: usize = 1 << 14;

trait Num: Copy + PartialOrd + Send + Sync {
    fn add(a: Self, b: Self) -> Self;
    fn sub(a: Self, b: Self) -> Self;
    fn mul(a: Self, b: Self) -> Self;
    fn div(a: Self, b: Self) -> Self;
    fn rem(a: Self, b: Self) -> Self;
    fn pow(a: Self, b: Self) -> Result<Self>;
    fn neg(a: Self) -> Self;
}

macro_rules! int_num {
    ($($t:ty),*) => {$(
        impl Num for $t {
            fn add(a: Self, b: Self) -> Self { a.wrapping_add(b) }
            fn sub(a: Self, b: Self) -> Self { a.wrapping_sub(b) }
            fn mul(a: Self, b: Self) -> Self { a.wrapping_mul(b) }
            // Integer true division is promoted to float before reaching kernels.
            fn div(a: Self, b: Self) -> Self { if b == 0 { 0 } else { a.wrapping_div(b) } }
            // NumPy follows Python: the remainder has the divisor's sign; x % 0 == 0.
            #[allow(unused_comparisons)]
            fn rem(a: Self, b: Self) -> Self {
                if b == 0 { return 0; }
                let r = a.wrapping_rem(b);
                if r != 0 && ((r < 0) != (b < 0)) { r.wrapping_add(b) } else { r }
            }
            #[allow(unused_comparisons)]
            fn pow(a: Self, b: Self) -> Result<Self> {
                if b < 0 {
                    return Err(Error::new(
                        ErrorCode::DTypeUnsupported,
                        "Integers to negative integer powers are not allowed.",
                    ));
                }
                Ok(a.wrapping_pow(b as u32))
            }
            fn neg(a: Self) -> Self { a.wrapping_neg() }
        }
    )*};
}
int_num!(i32, i64, u32, u64);

macro_rules! float_num {
    ($($t:ty),*) => {$(
        impl Num for $t {
            fn add(a: Self, b: Self) -> Self { a + b }
            fn sub(a: Self, b: Self) -> Self { a - b }
            fn mul(a: Self, b: Self) -> Self { a * b }
            fn div(a: Self, b: Self) -> Self { a / b }
            fn rem(a: Self, b: Self) -> Self {
                if b == 0.0 { return <$t>::NAN; }
                let r = a % b;
                if r != 0.0 && ((r < 0.0) != (b < 0.0)) { r + b } else if r == 0.0 { (0.0 as $t).copysign(b) } else { r }
            }
            fn pow(a: Self, b: Self) -> Result<Self> { Ok(a.powf(b)) }
            fn neg(a: Self) -> Self { -a }
        }
    )*};
}
float_num!(f32, f64);

fn apply<T: Num>(op: BinaryOp, a: T, b: T) -> Result<T> {
    Ok(match op {
        BinaryOp::Add => T::add(a, b),
        BinaryOp::Sub => T::sub(a, b),
        BinaryOp::Mul => T::mul(a, b),
        BinaryOp::Div => T::div(a, b),
        BinaryOp::Mod => T::rem(a, b),
        BinaryOp::Pow => T::pow(a, b)?,
    })
}

fn binary_loop<T: Num>(
    op: BinaryOp,
    a: &[T],
    b: &[T],
    a_off: &[usize],
    b_off: &[usize],
    token: &CancellationToken,
) -> Result<Vec<T>> {
    let mut out = Vec::with_capacity(a_off.len());
    for (chunk_a, chunk_b) in a_off.chunks(CHUNK).zip(b_off.chunks(CHUNK)) {
        token.check()?;
        for (&i, &j) in chunk_a.iter().zip(chunk_b) {
            out.push(apply(op, a[i], b[j])?);
        }
    }
    Ok(out)
}

fn reserve_for(
    budget: &MemoryBudget,
    dtype: DType,
    len: usize,
) -> Result<crate::memory::Reservation> {
    budget.reserve((len * dtype.item_size()) as u64)
}

/// Elementwise binary operation with broadcasting.
pub fn binary(
    op: BinaryOp,
    lhs: &DenseArray,
    rhs: &DenseArray,
    compute_dtype: DType,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    let shape = broadcast_shapes(&lhs.shape, &rhs.shape)?;
    let result_dtype = op.result_dtype(compute_dtype)?;
    let work_dtype = if op == BinaryOp::Div {
        result_dtype
    } else {
        compute_dtype
    };
    let reservation = reserve_for(budget, result_dtype, shape.iter().product())?;
    let a = lhs.buffer.cast(work_dtype)?;
    let b = rhs.buffer.cast(work_dtype)?;
    let mut a_off = Vec::new();
    let mut b_off = Vec::new();
    for_each_offset(&shape, &broadcast_strides(&lhs.shape, &shape), 0, |o| {
        a_off.push(o)
    });
    for_each_offset(&shape, &broadcast_strides(&rhs.shape, &shape), 0, |o| {
        b_off.push(o)
    });
    let buffer = match (&a, &b) {
        (Buffer::Bool(x), Buffer::Bool(y)) => {
            token.check()?;
            let f: fn(bool, bool) -> bool = match op {
                BinaryOp::Add => |p, q| p | q,
                BinaryOp::Mul => |p, q| p & q,
                _ => unreachable!("rejected by result_dtype"),
            };
            Buffer::Bool(
                a_off
                    .iter()
                    .zip(&b_off)
                    .map(|(&i, &j)| f(x[i], y[j]))
                    .collect(),
            )
        }
        (Buffer::I32(x), Buffer::I32(y)) => binary_loop(op, x, y, &a_off, &b_off, token)?.into(),
        (Buffer::I64(x), Buffer::I64(y)) => binary_loop(op, x, y, &a_off, &b_off, token)?.into(),
        (Buffer::U32(x), Buffer::U32(y)) => binary_loop(op, x, y, &a_off, &b_off, token)?.into(),
        (Buffer::U64(x), Buffer::U64(y)) => binary_loop(op, x, y, &a_off, &b_off, token)?.into(),
        (Buffer::F32(x), Buffer::F32(y)) => binary_loop(op, x, y, &a_off, &b_off, token)?.into(),
        (Buffer::F64(x), Buffer::F64(y)) => binary_loop(op, x, y, &a_off, &b_off, token)?.into(),
        _ => unreachable!("operands were cast to one dtype"),
    };
    DenseArray::new(shape, buffer, reservation)
}

pub fn unary(
    op: UnaryOp,
    input: &DenseArray,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    token.check()?;
    if op == UnaryOp::Neg && input.dtype() == DType::Bool {
        return Err(Error::new(
            ErrorCode::DTypeUnsupported,
            "The negative operator is not supported for bool operands.",
        ));
    }
    let reservation = reserve_for(budget, input.dtype(), input.size())?;
    let buffer = match op {
        UnaryOp::Pos => input.buffer.clone(),
        UnaryOp::Neg => match &input.buffer {
            Buffer::I32(v) => v.iter().map(|&x| Num::neg(x)).collect::<Vec<_>>().into(),
            Buffer::I64(v) => v.iter().map(|&x| Num::neg(x)).collect::<Vec<_>>().into(),
            Buffer::U32(v) => v.iter().map(|&x| Num::neg(x)).collect::<Vec<_>>().into(),
            Buffer::U64(v) => v.iter().map(|&x| Num::neg(x)).collect::<Vec<_>>().into(),
            Buffer::F32(v) => v.iter().map(|&x| -x).collect::<Vec<_>>().into(),
            Buffer::F64(v) => v.iter().map(|&x| -x).collect::<Vec<_>>().into(),
            Buffer::Bool(_) => unreachable!(),
        },
    };
    DenseArray::new(input.shape.clone(), buffer, reservation)
}

/// Normalizes reduction axes; `None` reduces every axis.
pub fn reduce_axes(rank: usize, axes: Option<&[i64]>) -> Result<Vec<bool>> {
    let mut reduced = vec![axes.is_none(); rank];
    for &axis in axes.unwrap_or(&[]) {
        let a = if axis < 0 { axis + rank as i64 } else { axis };
        if a < 0 || a >= rank as i64 {
            return Err(Error::new(
                ErrorCode::IndexOutOfBounds,
                format!("Axis {axis} is out of bounds for an array of dimension {rank}."),
            ));
        }
        if reduced[a as usize] {
            return Err(Error::new(
                ErrorCode::IndexInvalid,
                format!("Axis {axis} is repeated."),
            ));
        }
        reduced[a as usize] = true;
    }
    Ok(reduced)
}

pub fn reduce_shape(shape: &[usize], reduced: &[bool]) -> Vec<usize> {
    shape
        .iter()
        .zip(reduced)
        .filter(|(_, r)| !**r)
        .map(|(d, _)| *d)
        .collect()
}

pub fn reduce(
    op: ReduceOp,
    input: &DenseArray,
    axes: Option<&[i64]>,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    let reduced = reduce_axes(input.shape.len(), axes)?;
    let out_shape = reduce_shape(&input.shape, &reduced);
    let out_len: usize = out_shape.iter().product();
    let in_dtype = input.dtype();
    let dtype = op.result_dtype(in_dtype);
    if matches!(op, ReduceOp::Min | ReduceOp::Max) && input.size() == 0 {
        return Err(Error::new(
            ErrorCode::ShapeMismatch,
            format!(
                "A zero-size array has no identity for the '{}' reduction.",
                op.id()
            ),
        ));
    }
    let reservation = reserve_for(budget, dtype, out_len)?;
    // Map each input element to its output slot.
    let mut out_strides = vec![0isize; input.shape.len()];
    let mut acc = 1isize;
    for axis in (0..input.shape.len()).rev() {
        if !reduced[axis] {
            out_strides[axis] = acc;
            acc *= input.shape[axis] as isize;
        }
    }
    let mut slots = Vec::with_capacity(input.size());
    for_each_offset(&input.shape, &out_strides, 0, |o| slots.push(o));
    let count = if out_len == 0 {
        0
    } else {
        input.size() / out_len
    };
    token.check()?;

    let buffer = if in_dtype.kind() == DTypeKind::Float {
        let values = input.buffer.cast(DType::Float64)?;
        let Buffer::F64(v) = values else {
            unreachable!()
        };
        let mut acc = vec![
            match op {
                ReduceOp::Sum | ReduceOp::Mean => 0.0,
                ReduceOp::Min => f64::INFINITY,
                ReduceOp::Max => f64::NEG_INFINITY,
            };
            out_len
        ];
        for (chunk, slot_chunk) in v.chunks(CHUNK).zip(slots.chunks(CHUNK)) {
            token.check()?;
            for (&x, &s) in chunk.iter().zip(slot_chunk) {
                acc[s] = match op {
                    ReduceOp::Sum | ReduceOp::Mean => acc[s] + x,
                    // NaN propagates, as in NumPy.
                    ReduceOp::Min => {
                        if x.is_nan() || acc[s].is_nan() {
                            f64::NAN
                        } else {
                            acc[s].min(x)
                        }
                    }
                    ReduceOp::Max => {
                        if x.is_nan() || acc[s].is_nan() {
                            f64::NAN
                        } else {
                            acc[s].max(x)
                        }
                    }
                };
            }
        }
        if op == ReduceOp::Mean {
            for a in &mut acc {
                *a /= count as f64;
            }
        }
        Buffer::F64(acc).cast(dtype)?
    } else {
        let values: Vec<i128> = match &input.buffer {
            Buffer::Bool(v) => v.iter().map(|&x| x as i128).collect(),
            Buffer::I32(v) => v.iter().map(|&x| x as i128).collect(),
            Buffer::I64(v) => v.iter().map(|&x| x as i128).collect(),
            Buffer::U32(v) => v.iter().map(|&x| x as i128).collect(),
            Buffer::U64(v) => v.iter().map(|&x| x as i128).collect(),
            _ => unreachable!(),
        };
        if op == ReduceOp::Mean {
            let mut acc = vec![0f64; out_len];
            for (&x, &s) in values.iter().zip(&slots) {
                acc[s] += x as f64;
            }
            Buffer::F64(acc.into_iter().map(|a| a / count as f64).collect())
        } else {
            let mut acc = vec![
                match op {
                    ReduceOp::Sum => 0i128,
                    ReduceOp::Min => i128::MAX,
                    _ => i128::MIN,
                };
                out_len
            ];
            for (chunk, slot_chunk) in values.chunks(CHUNK).zip(slots.chunks(CHUNK)) {
                token.check()?;
                for (&x, &s) in chunk.iter().zip(slot_chunk) {
                    acc[s] = match op {
                        ReduceOp::Sum => acc[s].wrapping_add(x),
                        ReduceOp::Min => acc[s].min(x),
                        _ => acc[s].max(x),
                    };
                }
            }
            let ints = Buffer::I64(acc.iter().map(|&a| a as i64).collect());
            match dtype {
                // Keep full unsigned range for uint64 results.
                DType::UInt64 => Buffer::U64(acc.iter().map(|&a| a as u64).collect()),
                DType::Bool => Buffer::Bool(acc.iter().map(|&a| a != 0).collect()),
                other => ints.cast(other)?,
            }
        }
    };
    DenseArray::new(out_shape, buffer, reservation)
}

/// `matmul` for 1-D and 2-D operands, following NumPy's 1-D promotion rules.
pub fn matmul(
    lhs: &DenseArray,
    rhs: &DenseArray,
    dtype: DType,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    let (n, k, out_shape_rows) = match lhs.shape.as_slice() {
        [k] => (1, *k, false),
        [n, k] => (*n, *k, true),
        other => return Err(matmul_rank_error(other)),
    };
    let (k2, m, out_shape_cols) = match rhs.shape.as_slice() {
        [k] => (*k, 1, false),
        [k, m] => (*k, *m, true),
        other => return Err(matmul_rank_error(other)),
    };
    if k != k2 {
        return Err(Error::new(
            ErrorCode::ShapeMismatch,
            format!(
                "matmul: the core dimension of shape {:?} ({k}) does not match shape {:?} ({k2}).",
                lhs.shape, rhs.shape
            ),
        ));
    }
    let mut out_shape = Vec::new();
    if out_shape_rows {
        out_shape.push(n);
    }
    if out_shape_cols {
        out_shape.push(m);
    }
    let reservation = reserve_for(budget, dtype, n * m)?;
    let a = lhs.buffer.cast(dtype)?;
    let b = rhs.buffer.cast(dtype)?;
    fn mm<T: Num + Default>(
        a: &[T],
        b: &[T],
        n: usize,
        k: usize,
        m: usize,
        token: &CancellationToken,
    ) -> Result<Vec<T>> {
        let mut out = vec![T::default(); n * m];
        for i in 0..n {
            token.check()?;
            for p in 0..k {
                let x = a[i * k + p];
                for j in 0..m {
                    out[i * m + j] = T::add(out[i * m + j], T::mul(x, b[p * m + j]));
                }
            }
        }
        Ok(out)
    }
    let buffer = match (&a, &b) {
        (Buffer::Bool(x), Buffer::Bool(y)) => {
            let mut out = vec![false; n * m];
            for i in 0..n {
                token.check()?;
                for j in 0..m {
                    out[i * m + j] = (0..k).any(|p| x[i * k + p] && y[p * m + j]);
                }
            }
            Buffer::Bool(out)
        }
        (Buffer::I32(x), Buffer::I32(y)) => mm(x, y, n, k, m, token)?.into(),
        (Buffer::I64(x), Buffer::I64(y)) => mm(x, y, n, k, m, token)?.into(),
        (Buffer::U32(x), Buffer::U32(y)) => mm(x, y, n, k, m, token)?.into(),
        (Buffer::U64(x), Buffer::U64(y)) => mm(x, y, n, k, m, token)?.into(),
        (Buffer::F32(x), Buffer::F32(y)) => mm(x, y, n, k, m, token)?.into(),
        (Buffer::F64(x), Buffer::F64(y)) => mm(x, y, n, k, m, token)?.into(),
        _ => unreachable!(),
    };
    DenseArray::new(out_shape, buffer, reservation)
}

fn matmul_rank_error(shape: &[usize]) -> Error {
    // Gap: NumPy also supports stacked (batched) matmul for rank > 2.
    Error::new(
        ErrorCode::ShapeMismatch,
        format!("matmul in this build supports 1-D and 2-D operands; got shape {shape:?}."),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Device, Reservation};

    fn dense(shape: Vec<usize>, buffer: Buffer) -> DenseArray {
        DenseArray::new(shape, buffer, Reservation::none()).unwrap()
    }

    fn env() -> (MemoryBudget, CancellationToken) {
        (
            MemoryBudget::new(Device::Cpu, u64::MAX),
            CancellationToken::new(),
        )
    }

    #[test]
    fn integer_mod_follows_python_sign_rules() {
        // NumPy 2.5.4: np.array([-7, 7], np.int32) % np.array([3, -3],
        // np.int32) == [2, -2]
        let (b, t) = env();
        let a = dense(vec![2], Buffer::I32(vec![-7, 7]));
        let d = dense(vec![2], Buffer::I32(vec![3, -3]));
        let r = binary(BinaryOp::Mod, &a, &d, DType::Int32, &b, &t).unwrap();
        assert_eq!(r.buffer, Buffer::I32(vec![2, -2]));
    }

    #[test]
    fn float_mod_follows_numpy() {
        // NumPy 2.5.4: np.array([-7.5, 7.5]) % np.array([2.0, -2.0]) == [0.5,
        // -0.5]
        let (b, t) = env();
        let a = dense(vec![2], Buffer::F64(vec![-7.5, 7.5]));
        let d = dense(vec![2], Buffer::F64(vec![2.0, -2.0]));
        let r = binary(BinaryOp::Mod, &a, &d, DType::Float64, &b, &t).unwrap();
        assert_eq!(r.buffer, Buffer::F64(vec![0.5, -0.5]));
    }

    #[test]
    fn integer_division_is_true_division() {
        let (b, t) = env();
        let a = dense(vec![2], Buffer::I32(vec![1, 3]));
        let d = dense(vec![], Buffer::I32(vec![2]));
        let r = binary(BinaryOp::Div, &a, &d, DType::Int32, &b, &t).unwrap();
        assert_eq!(r.buffer, Buffer::F64(vec![0.5, 1.5]));
    }

    #[test]
    fn negative_integer_power_fails() {
        let (b, t) = env();
        let a = dense(vec![1], Buffer::I64(vec![2]));
        let e = dense(vec![1], Buffer::I64(vec![-1]));
        assert!(binary(BinaryOp::Pow, &a, &e, DType::Int64, &b, &t).is_err());
    }

    #[test]
    fn reductions_by_axis() {
        let (b, t) = env();
        let a = dense(vec![2, 3], Buffer::I32(vec![1, 2, 3, 4, 5, 6]));
        let s = reduce(ReduceOp::Sum, &a, Some(&[0]), &b, &t).unwrap();
        assert_eq!(
            (s.shape.clone(), s.buffer),
            (vec![3], Buffer::I64(vec![5, 7, 9]))
        );
        let m = reduce(ReduceOp::Mean, &a, Some(&[-1]), &b, &t).unwrap();
        assert_eq!(m.buffer, Buffer::F64(vec![2.0, 5.0]));
        let mx = reduce(ReduceOp::Max, &a, None, &b, &t).unwrap();
        assert_eq!(
            (mx.shape.clone(), mx.buffer),
            (vec![], Buffer::I32(vec![6]))
        );
        let empty = dense(vec![0], Buffer::F64(vec![]));
        assert!(reduce(ReduceOp::Min, &empty, None, &b, &t).is_err());
        assert_eq!(
            reduce(ReduceOp::Sum, &empty, None, &b, &t).unwrap().buffer,
            Buffer::F64(vec![0.0])
        );
    }

    #[test]
    fn matmul_2d_and_1d() {
        let (b, t) = env();
        let a = dense(vec![2, 2], Buffer::F64(vec![1.0, 2.0, 3.0, 4.0]));
        let v = dense(vec![2], Buffer::F64(vec![1.0, 1.0]));
        let r = matmul(&a, &v, DType::Float64, &b, &t).unwrap();
        assert_eq!(
            (r.shape.clone(), r.buffer),
            (vec![2], Buffer::F64(vec![3.0, 7.0]))
        );
        let rr = matmul(&a, &a, DType::Float64, &b, &t).unwrap();
        assert_eq!(rr.buffer, Buffer::F64(vec![7.0, 10.0, 15.0, 22.0]));
        let bad = dense(vec![3], Buffer::F64(vec![1.0; 3]));
        assert_eq!(
            matmul(&a, &bad, DType::Float64, &b, &t).unwrap_err().code,
            ErrorCode::ShapeMismatch
        );
    }

    #[test]
    fn kernels_stop_when_cancelled() {
        let (b, t) = env();
        t.cancel();
        let a = dense(vec![2], Buffer::F64(vec![1.0, 2.0]));
        assert_eq!(
            binary(BinaryOp::Add, &a, &a, DType::Float64, &b, &t)
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
    }

    #[test]
    fn kernels_charge_result_budget() {
        let budget = MemoryBudget::new(Device::Cpu, 8);
        let t = CancellationToken::new();
        let a = dense(vec![2], Buffer::F64(vec![1.0, 2.0]));
        assert_eq!(
            binary(BinaryOp::Add, &a, &a, DType::Float64, &budget, &t)
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded
        );
    }
}
