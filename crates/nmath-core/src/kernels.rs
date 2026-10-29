//! CPU kernels over dense arrays.

use std::borrow::Cow;

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
                // Square-and-multiply over the full exponent: casting to the
                // `u32` that `wrapping_pow` takes would truncate 64-bit
                // exponents.
                let (mut base, mut exp, mut acc) = (a, b, 1 as $t);
                while exp != 0 {
                    if exp & 1 == 1 {
                        acc = acc.wrapping_mul(base);
                    }
                    base = base.wrapping_mul(base);
                    exp >>= 1;
                }
                Ok(acc)
            }
            fn neg(a: Self) -> Self { a.wrapping_neg() }
        }
    )*};
}
int_num!(i8, i16, u8, u16, i32, i64, u32, u64);

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
    pairs: BroadcastPairs,
    token: &CancellationToken,
) -> Result<Vec<T>> {
    let mut out = Vec::with_capacity(pairs.remaining);
    for (n, (i, j)) in pairs.enumerate() {
        if n % CHUNK == 0 {
            token.check()?;
        }
        out.push(apply(op, a[i], b[j])?);
    }
    Ok(out)
}

/// Walks a broadcast output shape in C order and yields the matching element
/// offsets of both operands without materializing offset vectors.
struct BroadcastPairs {
    shape: Vec<usize>,
    a_strides: Vec<isize>,
    b_strides: Vec<isize>,
    index: Vec<usize>,
    a: isize,
    b: isize,
    remaining: usize,
}

impl BroadcastPairs {
    fn new(shape: &[usize], a_strides: Vec<isize>, b_strides: Vec<isize>) -> Self {
        Self {
            shape: shape.to_vec(),
            a_strides,
            b_strides,
            index: vec![0; shape.len()],
            a: 0,
            b: 0,
            remaining: shape.iter().product(),
        }
    }
}

impl Iterator for BroadcastPairs {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let item = (self.a as usize, self.b as usize);
        for axis in (0..self.shape.len()).rev() {
            self.index[axis] += 1;
            self.a += self.a_strides[axis];
            self.b += self.b_strides[axis];
            if self.index[axis] < self.shape[axis] {
                break;
            }
            let len = self.shape[axis] as isize;
            self.a -= self.a_strides[axis] * len;
            self.b -= self.b_strides[axis] * len;
            self.index[axis] = 0;
        }
        Some(item)
    }
}

fn reserve_for(
    budget: &MemoryBudget,
    dtype: DType,
    len: usize,
) -> Result<crate::memory::Reservation> {
    budget.reserve((len * dtype.item_size()) as u64)
}

/// Casts an operand to `dtype`. A matching buffer is borrowed; a converted
/// temporary copy is charged to `budget` while it lives.
fn cast_operand<'a>(
    buffer: &'a Buffer,
    dtype: DType,
    budget: &MemoryBudget,
) -> Result<(Cow<'a, Buffer>, crate::memory::Reservation)> {
    if buffer.dtype() == dtype {
        return Ok((Cow::Borrowed(buffer), crate::memory::Reservation::none()));
    }
    let reservation = reserve_for(budget, dtype, buffer.len())?;
    Ok((Cow::Owned(buffer.cast(dtype)?), reservation))
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
    let (a, _a_reservation) = cast_operand(&lhs.buffer, work_dtype, budget)?;
    let (b, _b_reservation) = cast_operand(&rhs.buffer, work_dtype, budget)?;
    let pairs = BroadcastPairs::new(
        &shape,
        broadcast_strides(&lhs.shape, &shape),
        broadcast_strides(&rhs.shape, &shape),
    );
    let buffer = match (a.as_ref(), b.as_ref()) {
        (Buffer::Bool(x), Buffer::Bool(y)) => {
            let f: fn(bool, bool) -> bool = match op {
                BinaryOp::Add => |p, q| p | q,
                BinaryOp::Mul => |p, q| p & q,
                _ => unreachable!("rejected by result_dtype"),
            };
            token.check()?;
            let mut out = Vec::with_capacity(pairs.remaining);
            for (n, (i, j)) in pairs.enumerate() {
                if n % CHUNK == 0 {
                    token.check()?;
                }
                out.push(f(x[i], y[j]));
            }
            Buffer::Bool(out)
        }
        (Buffer::I8(x), Buffer::I8(y)) => binary_loop(op, x, y, pairs, token)?.into(),
        (Buffer::I16(x), Buffer::I16(y)) => binary_loop(op, x, y, pairs, token)?.into(),
        (Buffer::I32(x), Buffer::I32(y)) => binary_loop(op, x, y, pairs, token)?.into(),
        (Buffer::I64(x), Buffer::I64(y)) => binary_loop(op, x, y, pairs, token)?.into(),
        (Buffer::U8(x), Buffer::U8(y)) => binary_loop(op, x, y, pairs, token)?.into(),
        (Buffer::U16(x), Buffer::U16(y)) => binary_loop(op, x, y, pairs, token)?.into(),
        (Buffer::U32(x), Buffer::U32(y)) => binary_loop(op, x, y, pairs, token)?.into(),
        (Buffer::U64(x), Buffer::U64(y)) => binary_loop(op, x, y, pairs, token)?.into(),
        (Buffer::F32(x), Buffer::F32(y)) => binary_loop(op, x, y, pairs, token)?.into(),
        (Buffer::F64(x), Buffer::F64(y)) => binary_loop(op, x, y, pairs, token)?.into(),
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
            Buffer::I8(v) => v.iter().map(|&x| Num::neg(x)).collect::<Vec<_>>().into(),
            Buffer::I16(v) => v.iter().map(|&x| Num::neg(x)).collect::<Vec<_>>().into(),
            Buffer::I32(v) => v.iter().map(|&x| Num::neg(x)).collect::<Vec<_>>().into(),
            Buffer::I64(v) => v.iter().map(|&x| Num::neg(x)).collect::<Vec<_>>().into(),
            Buffer::U8(v) => v.iter().map(|&x| Num::neg(x)).collect::<Vec<_>>().into(),
            Buffer::U16(v) => v.iter().map(|&x| Num::neg(x)).collect::<Vec<_>>().into(),
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
    // Only an empty reduced dimension lacks an identity; empty kept dimensions
    // simply produce an empty result.
    let empty_reduction = input
        .shape
        .iter()
        .zip(&reduced)
        .any(|(&len, &r)| r && len == 0);
    if matches!(op, ReduceOp::Min | ReduceOp::Max) && empty_reduction {
        return Err(Error::new(
            ErrorCode::ShapeMismatch,
            format!(
                "The '{}' reduction has no identity over a zero-size axis.",
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
            Buffer::I8(v) => v.iter().map(|&x| x as i128).collect(),
            Buffer::I16(v) => v.iter().map(|&x| x as i128).collect(),
            Buffer::I32(v) => v.iter().map(|&x| x as i128).collect(),
            Buffer::I64(v) => v.iter().map(|&x| x as i128).collect(),
            Buffer::U8(v) => v.iter().map(|&x| x as i128).collect(),
            Buffer::U16(v) => v.iter().map(|&x| x as i128).collect(),
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
        (Buffer::I8(x), Buffer::I8(y)) => mm(x, y, n, k, m, token)?.into(),
        (Buffer::I16(x), Buffer::I16(y)) => mm(x, y, n, k, m, token)?.into(),
        (Buffer::I32(x), Buffer::I32(y)) => mm(x, y, n, k, m, token)?.into(),
        (Buffer::I64(x), Buffer::I64(y)) => mm(x, y, n, k, m, token)?.into(),
        (Buffer::U8(x), Buffer::U8(y)) => mm(x, y, n, k, m, token)?.into(),
        (Buffer::U16(x), Buffer::U16(y)) => mm(x, y, n, k, m, token)?.into(),
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
    fn integer_power_keeps_64_bit_exponents() {
        // NumPy 2.5.4: np.int64(2) ** np.int64(2**32) == 0 and
        // np.int64(3) ** np.int64(2**32 + 1) == 3 ** (2**32 + 1) mod 2**64.
        let (b, t) = env();
        let base = dense(vec![3], Buffer::I64(vec![2, 3, -1]));
        let exp = dense(
            vec![3],
            Buffer::I64(vec![1 << 32, (1 << 32) + 1, (1 << 32) + 1]),
        );
        let r = binary(BinaryOp::Pow, &base, &exp, DType::Int64, &b, &t).unwrap();
        assert_eq!(
            r.buffer,
            Buffer::I64(vec![
                0,
                3i64.wrapping_pow(1 << 31).wrapping_pow(2).wrapping_mul(3),
                -1
            ])
        );
        let ubase = dense(vec![2], Buffer::U64(vec![2, 7]));
        let uexp = dense(vec![2], Buffer::U64(vec![u64::MAX, 0]));
        let r = binary(BinaryOp::Pow, &ubase, &uexp, DType::UInt64, &b, &t).unwrap();
        assert_eq!(r.buffer, Buffer::U64(vec![0, 1]));
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
    fn min_max_allow_empty_inputs_with_nonempty_reduced_axes() {
        // NumPy 2.5.4: np.zeros((0, 2)).min(axis=1) has shape (0,), while
        // np.zeros((0, 2)).max(axis=0) raises ValueError.
        let (b, t) = env();
        let empty = dense(vec![0, 2], Buffer::F64(vec![]));
        for op in [ReduceOp::Min, ReduceOp::Max] {
            let r = reduce(op, &empty, Some(&[1]), &b, &t).unwrap();
            assert_eq!((r.shape.clone(), r.buffer), (vec![0], Buffer::F64(vec![])));
            assert_eq!(
                reduce(op, &empty, Some(&[0]), &b, &t).unwrap_err().code,
                ErrorCode::ShapeMismatch
            );
        }
        let ints = dense(vec![0, 2], Buffer::I32(vec![]));
        let r = reduce(ReduceOp::Max, &ints, Some(&[-1]), &b, &t).unwrap();
        assert_eq!((r.shape.clone(), r.buffer), (vec![0], Buffer::I32(vec![])));
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
    fn binary_broadcasts_without_offset_vectors() {
        let (b, t) = env();
        let col = dense(vec![2, 1], Buffer::I32(vec![10, 20]));
        let row = dense(vec![3], Buffer::I32(vec![1, 2, 3]));
        let r = binary(BinaryOp::Add, &col, &row, DType::Int32, &b, &t).unwrap();
        assert_eq!(r.shape, vec![2, 3]);
        assert_eq!(r.buffer, Buffer::I32(vec![11, 12, 13, 21, 22, 23]));
        // Only the result is charged when no operand needs a cast.
        assert_eq!(b.usage().peak, 24);
    }

    #[test]
    fn binary_charges_operand_casts() {
        // The int32 operand is cast to float64 (16 bytes) on top of the
        // 16-byte result; 24 bytes are not enough.
        let budget = MemoryBudget::new(Device::Cpu, 24);
        let t = CancellationToken::new();
        let a = dense(vec![2], Buffer::I32(vec![1, 2]));
        let f = dense(vec![2], Buffer::F64(vec![0.5, 0.5]));
        assert_eq!(
            binary(BinaryOp::Add, &a, &f, DType::Float64, &budget, &t)
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded
        );
        let budget = MemoryBudget::new(Device::Cpu, 32);
        let r = binary(BinaryOp::Add, &a, &f, DType::Float64, &budget, &t).unwrap();
        assert_eq!(r.buffer, Buffer::F64(vec![1.5, 2.5]));
        assert_eq!(budget.usage().used, 16, "the cast copy is released");
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
