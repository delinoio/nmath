//! CPU kernels over dense arrays.

use std::borrow::Cow;

use crate::{
    buffer::{Buffer, DenseArray, Float16},
    dtype::{DType, DTypeKind},
    error::{Error, ErrorCode, Result},
    memory::MemoryBudget,
    shape::{broadcast_shapes, broadcast_strides},
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
        if promoted.kind() == DTypeKind::Complex && self == BinaryOp::Mod {
            return Err(Error::new(
                ErrorCode::DTypeUnsupported,
                "Remainder is not defined for complex operands. Use real operands.",
            ));
        }
        if promoted == DType::Bool && self == BinaryOp::Sub {
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
            BinaryOp::Mod | BinaryOp::Pow if promoted == DType::Bool => DType::Int8,
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
    Prod,
    ArgMin,
    ArgMax,
}

impl ReduceOp {
    pub fn id(self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Mean => "mean",
            ReduceOp::Min => "min",
            ReduceOp::Max => "max",
            ReduceOp::Prod => "prod",
            ReduceOp::ArgMin => "argmin",
            ReduceOp::ArgMax => "argmax",
        }
    }

    pub fn is_arg(self) -> bool {
        matches!(self, ReduceOp::ArgMin | ReduceOp::ArgMax)
    }

    pub fn result_dtype(self, input: DType) -> DType {
        match self {
            ReduceOp::Sum | ReduceOp::Prod => input.sum_dtype(),
            ReduceOp::Mean if input.temporal_unit().is_some() => input,
            ReduceOp::Mean => input.true_divide(),
            ReduceOp::Min | ReduceOp::Max => input,
            ReduceOp::ArgMin | ReduceOp::ArgMax => DType::Int64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScanOp {
    CumSum,
    CumProd,
}

impl ScanOp {
    pub fn id(self) -> &'static str {
        match self {
            ScanOp::CumSum => "cumsum",
            ScanOp::CumProd => "cumprod",
        }
    }

    pub fn result_dtype(self, input: DType) -> DType {
        input.sum_dtype()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VarianceOp {
    Var,
    Std,
}

impl VarianceOp {
    pub fn id(self) -> &'static str {
        match self {
            VarianceOp::Var => "var",
            VarianceOp::Std => "std",
        }
    }

    /// NumPy computes integer and boolean variance in float64, keeps float16,
    /// and uses the real component width of a complex input.
    pub fn result_dtype(self, input: DType) -> DType {
        match input {
            DType::Float16 => DType::Float16,
            DType::Float32 | DType::Complex64 => DType::Float32,
            _ => DType::Float64,
        }
    }
}

/// Elements processed between cancellation checks.
const CHUNK: usize = 1 << 14;

trait Num: Copy + PartialOrd + Send + Sync + Default {
    fn nan(self) -> bool {
        false
    }
    fn take_equal() -> bool {
        true
    }
    fn pairwise_sum(_: &[Self], _: &CancellationToken) -> Result<Option<Self>> {
        Ok(None)
    }
    fn mean(self, count: usize) -> Self;
    /// Multiplicative identity, used by the product reduction.
    fn one() -> Self;

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
            fn mean(self, count: usize) -> Self { self / count as $t }
            fn one() -> Self { 1 }
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
            fn nan(self) -> bool { self.is_nan() }
            fn mean(self, count: usize) -> Self { self / count as $t }
            fn one() -> Self { 1.0 }
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
float_num!(f64);

impl Num for f32 {
    fn nan(self) -> bool {
        self.is_nan()
    }

    fn pairwise_sum(values: &[Self], token: &CancellationToken) -> Result<Option<Self>> {
        fn sum(values: &[f32], token: &CancellationToken) -> Result<f32> {
            token.check()?;
            let n = values.len();
            if n < 8 {
                return Ok(values.iter().fold(-0.0_f32, |total, &value| total + value));
            }
            if n <= 128 {
                // Match NumPy 2.5.4's eight-lane real pairwise block order.
                let mut lanes: [f32; 8] = values[..8].try_into().unwrap();
                let end = n - n % 8;
                for block in values[8..end].chunks_exact(8) {
                    for lane in 0..8 {
                        lanes[lane] += block[lane];
                    }
                }
                let mut total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
                    + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
                for &value in &values[end..] {
                    total += value;
                }
                return Ok(total);
            }
            let mid = (n / 2) / 8 * 8;
            Ok(sum(&values[..mid], token)? + sum(&values[mid..], token)?)
        }
        Ok(Some(sum(values, token)?))
    }

    fn mean(self, count: usize) -> Self {
        self / count as f32
    }

    fn one() -> Self {
        1.0
    }

    fn add(a: Self, b: Self) -> Self {
        a + b
    }

    fn sub(a: Self, b: Self) -> Self {
        a - b
    }

    fn mul(a: Self, b: Self) -> Self {
        a * b
    }

    fn div(a: Self, b: Self) -> Self {
        a / b
    }

    fn rem(a: Self, b: Self) -> Self {
        if b == 0.0 {
            return f32::NAN;
        }
        let r = a % b;
        if r != 0.0 && ((r < 0.0) != (b < 0.0)) {
            r + b
        } else if r == 0.0 {
            0.0_f32.copysign(b)
        } else {
            r
        }
    }

    fn pow(a: Self, b: Self) -> Result<Self> {
        Ok(a.powf(b))
    }

    fn neg(a: Self) -> Self {
        -a
    }
}

// NumPy's half loops compute in binary32 and round each element to binary16.
impl Num for Float16 {
    fn nan(self) -> bool {
        self.is_nan()
    }

    fn take_equal() -> bool {
        false
    }

    fn mean(self, count: usize) -> Self {
        Self::from_f32(self.to_f32() / count as f32)
    }

    fn one() -> Self {
        Self::from_f32(1.0)
    }

    fn add(a: Self, b: Self) -> Self {
        Self::from_f32(a.to_f32() + b.to_f32())
    }

    fn sub(a: Self, b: Self) -> Self {
        Self::from_f32(a.to_f32() - b.to_f32())
    }

    fn mul(a: Self, b: Self) -> Self {
        Self::from_f32(a.to_f32() * b.to_f32())
    }

    fn div(a: Self, b: Self) -> Self {
        Self::from_f32(a.to_f32() / b.to_f32())
    }

    fn rem(a: Self, b: Self) -> Self {
        Self::from_f32(<f32 as Num>::rem(a.to_f32(), b.to_f32()))
    }

    fn pow(a: Self, b: Self) -> Result<Self> {
        Ok(Self::from_f32(a.to_f32().powf(b.to_f32())))
    }

    fn neg(a: Self) -> Self {
        -a
    }
}

// Match the eight-lane binary32 accumulation order of NumPy's half reduction.
fn half_pairwise(values: &[Float16], token: &CancellationToken) -> Result<f32> {
    token.check()?;
    let n = values.len();
    if n < 8 {
        return Ok(values.iter().fold(-0.0, |a, b| a + b.to_f32()));
    }
    if n > 128 {
        let mid = (n / 2) / 8 * 8;
        return Ok(half_pairwise(&values[..mid], token)? + half_pairwise(&values[mid..], token)?);
    }
    let mut lanes = [0.0f32; 8];
    for i in 0..8 {
        lanes[i] = values[i].to_f32();
    }
    let end = n / 8 * 8;
    for i in (8..end).step_by(8) {
        for j in 0..8 {
            lanes[j] += values[i + j].to_f32();
        }
    }
    let mut total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for value in &values[end..] {
        total += value.to_f32();
    }
    Ok(total)
}

// Complex operations keep component precision. Division uses a ratio to avoid
// squaring the denominator. Power handles zero and small integer exponents
// before the logarithmic path, as NumPy 2.5.4 does.
macro_rules! complex_num {
    ($($t:ty),*) => {$(
        impl Num for [$t; 2] {
            fn take_equal() -> bool { false }
            fn pairwise_sum(values: &[Self], token: &CancellationToken) -> Result<Option<Self>> {
                token.check()?;
                let n = values.len();
                let sum = if n < 4 {
                    values.iter().fold([-0.0, -0.0], |a, &b| Self::add(a, b))
                } else if n <= 64 {
                    // Four complex lanes keep the component addition order of
                    // NumPy's eight real-component pairwise accumulators.
                    let mut lanes = [values[0], values[1], values[2], values[3]];
                    let end = n - n % 4;
                    for i in (4..end).step_by(4) {
                        for j in 0..4 { lanes[j] = Self::add(lanes[j], values[i+j]); }
                    }
                    let mut total = Self::add(Self::add(lanes[0], lanes[1]), Self::add(lanes[2], lanes[3]));
                    for &value in &values[end..] { total = Self::add(total, value); }
                    total
                } else {
                    let mid = (n/2) / 4 * 4;
                    Self::add(Self::pairwise_sum(&values[..mid], token)?.unwrap(), Self::pairwise_sum(&values[mid..], token)?.unwrap())
                };
                Ok(Some(sum))
            }
            fn nan(self) -> bool { self[0].is_nan() || self[1].is_nan() }
            fn mean(self, count: usize) -> Self { Self::div(self, [count as $t, 0.0]) }
            fn one() -> Self { [1.0, 0.0] }
            fn add(a: Self, b: Self) -> Self { [a[0] + b[0], a[1] + b[1]] }
            fn sub(a: Self, b: Self) -> Self { [a[0] - b[0], a[1] - b[1]] }
            fn mul(a: Self, b: Self) -> Self { [a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0]] }
            fn div(a: Self, b: Self) -> Self {
                if b[0].abs() >= b[1].abs() {
                    if b[0] == 0.0 && b[1] == 0.0 { return [a[0]/b[0].abs(), a[1]/b[1].abs()]; }
                    let r = b[1]/b[0];
                    let scale = 1.0/(b[0]+b[1]*r);
                    [(a[0]+a[1]*r)*scale, (a[1]-a[0]*r)*scale]
                } else {
                    let r = b[0]/b[1];
                    let scale = 1.0/(b[1]+b[0]*r);
                    [(a[0]*r+a[1])*scale, (a[1]*r-a[0])*scale]
                }
            }
            fn rem(_: Self, _: Self) -> Self { unreachable!("complex remainder is rejected before execution") }
            fn pow(a: Self, b: Self) -> Result<Self> {
                if b == [0.0, 0.0] { return Ok([1.0, 0.0]); }
                if a == [0.0, 0.0] { return Ok(if b[0] > 0.0 { [0.0, 0.0] } else { [<$t>::NAN; 2] }); }
                if b[1] == 0.0 && b[0] > -100.0 && b[0] < 100.0 && b[0].trunc() == b[0] {
                    if b[0] == 1.0 { return Ok(a); }
                    if b[0] == 2.0 { return Ok(Self::mul(a, a)); }
                    if b[0] == 3.0 { return Ok(Self::mul(a, Self::mul(a, a))); }
                    let mut n = b[0].abs() as u32;
                    let (mut base, mut result) = (a, [1.0, 0.0]);
                    while n != 0 {
                        if n & 1 != 0 { result = Self::mul(result, base); }
                        n >>= 1;
                        if n != 0 { base = Self::mul(base, base); }
                    }
                    return Ok(if b[0] < 0.0 { Self::div([1.0, 0.0], result) } else { result });
                }
                let log_r = a[0].hypot(a[1]).ln();
                let theta = a[1].atan2(a[0]);
                let mut exponent = b[0]*log_r - b[1]*theta;
                let mut angle = b[0]*theta + b[1]*log_r;
                // C99 multiplication used by complex power recovers infinite
                // products when both preliminary components are NaN.
                if exponent.is_nan() && angle.is_nan() && (log_r.is_infinite() || b[0].is_infinite() || b[1].is_infinite()) {
                    let normalize = |z: [$t; 2]| {
                        if z[0].is_infinite() || z[1].is_infinite() {
                            z.map(|v| (if v.is_infinite() { 1.0 as $t } else { 0.0 as $t }).copysign(v))
                        } else { z.map(|v| if v.is_nan() { (0.0 as $t).copysign(v) } else { v }) }
                    };
                    let l = normalize([log_r, theta]);
                    let r = normalize(b);
                    exponent = <$t>::INFINITY * (l[0]*r[0]-l[1]*r[1]);
                    angle = <$t>::INFINITY * (l[0]*r[1]+l[1]*r[0]);
                }
                // Preserve the fixed NumPy oracle's non-finite power components.
                // A negative infinite phase differs from a positive one.
                if exponent == <$t>::INFINITY && !angle.is_finite() {
                    return Ok([if angle == <$t>::NEG_INFINITY { <$t>::NAN } else { exponent }, <$t>::NAN]);
                }
                if exponent == <$t>::NEG_INFINITY && !angle.is_finite() { return Ok([0.0, 0.0]); }
                if !exponent.is_finite() && angle == 0.0 { return Ok([exponent.exp(), angle]); }
                let magnitude = exponent.exp();
                Ok([magnitude * angle.cos(), magnitude * angle.sin()])
            }
            fn neg(a: Self) -> Self { [-a[0], -a[1]] }
        }
    )*};
}
complex_num!(f32, f64);

// bool min/max use the same ordering, but sum/mean were already widened.
impl Num for bool {
    fn mean(self, _: usize) -> Self {
        unreachable!()
    }

    fn one() -> Self {
        true
    }

    fn add(a: Self, b: Self) -> Self {
        a || b
    }

    fn sub(_: Self, _: Self) -> Self {
        unreachable!()
    }

    fn mul(a: Self, b: Self) -> Self {
        a && b
    }

    fn div(_: Self, _: Self) -> Self {
        unreachable!()
    }

    fn rem(_: Self, _: Self) -> Self {
        unreachable!()
    }

    fn pow(_: Self, _: Self) -> Result<Self> {
        unreachable!()
    }

    fn neg(_: Self) -> Self {
        unreachable!()
    }
}
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
    budget.reserve(checked_reduction_bytes(len, dtype.item_size())?)
}

fn checked_reduction_bytes(len: usize, item_size: usize) -> Result<u64> {
    len.checked_mul(item_size)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .filter(|&bytes| bytes <= isize::MAX as u64)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::MemoryBudgetExceeded,
                "Reduction allocation size exceeds the supported vector capacity.",
            )
        })
}

/// Casts an operand to `dtype`. A matching buffer is borrowed; a converted
/// temporary copy is charged to `budget` while it lives.
fn cast_operand<'a>(
    buffer: &'a Buffer,
    dtype: DType,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<(Cow<'a, Buffer>, crate::memory::Reservation)> {
    if buffer.dtype() == dtype {
        return Ok((Cow::Borrowed(buffer), crate::memory::Reservation::none()));
    }
    let reservation = reserve_for(budget, dtype, buffer.len())?;
    Ok((
        Cow::Owned(buffer.cast_cancellable(dtype, token)?),
        reservation,
    ))
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
    if lhs.dtype().kind() == DTypeKind::Other || rhs.dtype().kind() == DTypeKind::Other {
        return special_binary(op, lhs, rhs, budget, token);
    }
    let shape = broadcast_shapes(&lhs.shape, &rhs.shape)?;
    let result_dtype = op.result_dtype(compute_dtype)?;
    let work_dtype = if op == BinaryOp::Div
        || compute_dtype == DType::Bool && matches!(op, BinaryOp::Mod | BinaryOp::Pow)
    {
        result_dtype
    } else {
        compute_dtype
    };
    let reservation = reserve_for(budget, result_dtype, shape.iter().product())?;
    let (a, _a_reservation) = cast_operand(&lhs.buffer, work_dtype, budget, token)?;
    let (b, _b_reservation) = cast_operand(&rhs.buffer, work_dtype, budget, token)?;
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
        (Buffer::F16(x), Buffer::F16(y)) => binary_loop(op, x, y, pairs, token)?.into(),
        (Buffer::F32(x), Buffer::F32(y)) => binary_loop(op, x, y, pairs, token)?.into(),
        (Buffer::F64(x), Buffer::F64(y)) => binary_loop(op, x, y, pairs, token)?.into(),
        (Buffer::C64(x), Buffer::C64(y)) => binary_loop(op, x, y, pairs, token)?.into(),
        (Buffer::C128(x), Buffer::C128(y)) => binary_loop(op, x, y, pairs, token)?.into(),
        _ => unreachable!("operands were cast to one dtype"),
    };
    DenseArray::new(shape, buffer, reservation)
}

/// Horner evaluation reuses the arithmetic loops' typed operations. In
/// particular, integers never cross a floating-point scalar transport.
pub(crate) fn polynomial(
    options: crate::polynomial::PolynomialOptions,
    coefficients: &DenseArray,
    points: &DenseArray,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    use crate::polynomial::PolynomialOrder;
    token.check()?;
    let tail = coefficients.shape.get(1..).unwrap_or(&[]);
    let empty = coefficients.shape.first() == Some(&0);
    let rank = if empty {
        points.shape.len()
    } else if options.tensor {
        tail.len()
            .checked_add(points.shape.len())
            .ok_or_else(crate::numerical::overflow)?
    } else {
        tail.len().max(points.shape.len())
    };
    // Charge shape discovery and the two broadcast-stride vectors before
    // creating them. The output reservation retains its shape charge below.
    let _metadata = budget.reserve(crate::numerical::bytes(
        rank,
        3 * std::mem::size_of::<usize>(),
    )?)?;
    let (shape, dtype) = options.infer(
        &coefficients.shape,
        coefficients.dtype(),
        &points.shape,
        points.dtype(),
    )?;
    let count = if shape.contains(&0) {
        0
    } else {
        crate::numerical::size(&shape)?
    };
    let data_bytes = crate::buffer::checked_byte_len(count, dtype)?;
    let shape_bytes = rank
        .checked_mul(std::mem::size_of::<usize>())
        .ok_or_else(crate::numerical::overflow)?;
    let charge = budget.reserve(
        u64::try_from(
            data_bytes
                .checked_add(shape_bytes)
                .ok_or_else(crate::numerical::overflow)?,
        )
        .map_err(|_| crate::numerical::overflow())?,
    )?;
    if count == 0 || empty {
        return DenseArray::new(
            shape,
            Buffer::filled(dtype, count, crate::Scalar::Int(0))?,
            charge,
        );
    }
    let (c, _c_charge) = cast_operand(&coefficients.buffer, dtype, budget, token)?;
    let (x, _x_charge) = cast_operand(&points.buffer, dtype, budget, token)?;
    // NumPy performs the initial zero multiplication in the points' dtype,
    // before promotion with coefficients. For real non-finite points and
    // complex coefficients, promoting first would corrupt the imaginary part.
    let _zero_charge = reserve_for(budget, points.dtype(), 1)?;
    let zero = DenseArray::new(
        vec![],
        Buffer::filled(points.dtype(), 1, crate::Scalar::Int(0))?,
        crate::memory::Reservation::none(),
    )?;
    let initial = binary(BinaryOp::Mul, points, &zero, points.dtype(), budget, token)?;
    let (z, _z_charge) = cast_operand(&initial.buffer, dtype, budget, token)?;
    let degrees = coefficients.shape.first().copied().unwrap_or(1);
    let stride = c.len() / degrees;
    let c_strides = if options.tensor {
        Vec::new()
    } else {
        broadcast_strides(tail, &shape)
    };
    let x_strides = if options.tensor {
        Vec::new()
    } else {
        broadcast_strides(&points.shape, &shape)
    };
    let offset = |mut i: usize, strides: &[isize]| {
        let mut out = 0;
        for axis in (0..shape.len()).rev() {
            out += i % shape[axis] * strides[axis] as usize;
            i /= shape[axis];
        }
        out
    };
    let positions = |i: usize| {
        if options.tensor {
            (i / points.size(), i % points.size())
        } else {
            (offset(i, &c_strides), offset(i, &x_strides))
        }
    };
    fn evaluate<T: Copy + Default>(
        c: &[T],
        x: &[T],
        z: &[T],
        count: usize,
        degrees: usize,
        stride: usize,
        ascending: bool,
        positions: impl Fn(usize) -> (usize, usize),
        add: fn(T, T) -> T,
        mul: fn(T, T) -> T,
        token: &CancellationToken,
    ) -> Result<Vec<T>> {
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            if i % CHUNK == 0 {
                token.check()?;
            }
            let (ci, xi) = positions(i);
            let point = x[xi];
            let mut value = if ascending {
                add(c[(degrees - 1) * stride + ci], z[xi])
            } else {
                add(z[xi], c[ci])
            };
            for j in 1..degrees {
                if j % CHUNK == 0 {
                    token.check()?;
                }
                value = if ascending {
                    add(c[(degrees - 1 - j) * stride + ci], mul(value, point))
                } else {
                    add(mul(value, point), c[j * stride + ci])
                };
            }
            out.push(value);
        }
        token.check()?;
        Ok(out)
    }
    macro_rules! run {
        ($a:expr, $b:expr, $z:expr, $add:expr, $mul:expr) => {
            evaluate(
                $a,
                $b,
                $z,
                count,
                degrees,
                stride,
                options.order == PolynomialOrder::Ascending,
                positions,
                $add,
                $mul,
                token,
            )?
            .into()
        };
    }
    let buffer = match (c.as_ref(), x.as_ref(), z.as_ref()) {
        (Buffer::Bool(a), Buffer::Bool(b), Buffer::Bool(z)) => {
            run!(a, b, z, |p, q| p | q, |p, q| p & q)
        }
        (Buffer::I8(a), Buffer::I8(b), Buffer::I8(z)) => run!(a, b, z, Num::add, Num::mul),
        (Buffer::I16(a), Buffer::I16(b), Buffer::I16(z)) => run!(a, b, z, Num::add, Num::mul),
        (Buffer::I32(a), Buffer::I32(b), Buffer::I32(z)) => run!(a, b, z, Num::add, Num::mul),
        (Buffer::I64(a), Buffer::I64(b), Buffer::I64(z)) => run!(a, b, z, Num::add, Num::mul),
        (Buffer::U8(a), Buffer::U8(b), Buffer::U8(z)) => run!(a, b, z, Num::add, Num::mul),
        (Buffer::U16(a), Buffer::U16(b), Buffer::U16(z)) => run!(a, b, z, Num::add, Num::mul),
        (Buffer::U32(a), Buffer::U32(b), Buffer::U32(z)) => run!(a, b, z, Num::add, Num::mul),
        (Buffer::U64(a), Buffer::U64(b), Buffer::U64(z)) => run!(a, b, z, Num::add, Num::mul),
        (Buffer::F16(a), Buffer::F16(b), Buffer::F16(z)) => run!(a, b, z, Num::add, Num::mul),
        (Buffer::F32(a), Buffer::F32(b), Buffer::F32(z)) => run!(a, b, z, Num::add, Num::mul),
        (Buffer::F64(a), Buffer::F64(b), Buffer::F64(z)) => run!(a, b, z, Num::add, Num::mul),
        (Buffer::C64(a), Buffer::C64(b), Buffer::C64(z)) => run!(a, b, z, Num::add, Num::mul),
        (Buffer::C128(a), Buffer::C128(b), Buffer::C128(z)) => run!(a, b, z, Num::add, Num::mul),
        _ => unreachable!("polynomial operands were cast to their inferred numeric dtype"),
    };
    DenseArray::new(shape, buffer, charge)
}

pub(crate) fn polynomial_derivative(
    options: crate::polynomial::DerivativeOptions,
    coefficients: &DenseArray,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    use crate::polynomial::{DerivativeScale, PolynomialOrder};
    token.check()?;
    let rank = coefficients.shape.len().max(1);
    let shape_bytes = crate::numerical::bytes(rank, std::mem::size_of::<usize>())?;
    let _metadata = budget.reserve(shape_bytes)?;
    let (shape, dtype, axis) = options.infer(&coefficients.shape, coefficients.dtype())?;
    let count = if shape.contains(&0) {
        0
    } else {
        crate::numerical::size(&shape)?
    };
    let reservation = budget.reserve(
        (crate::buffer::checked_byte_len(count, dtype)? as u64)
            .checked_add(shape_bytes)
            .ok_or_else(crate::numerical::overflow)?,
    )?;
    if count == 0 {
        return DenseArray::new(
            shape,
            Buffer::filled(dtype, 0, crate::Scalar::Int(0))?,
            reservation,
        );
    }
    let (c, _cast) = cast_operand(&coefficients.buffer, dtype, budget, token)?;
    let n = coefficients.shape.get(axis).copied().unwrap_or(1);
    let remaining = shape[axis];
    let inner = crate::numerical::size(&shape[axis + 1..])?;
    let ascending = options.order == PolynomialOrder::Ascending;
    let zero = options.count > 0 && options.count >= n as u64;
    let steps = if zero { 0 } else { options.count as usize };
    let source = |i: usize| {
        let degree = (i / inner) % remaining;
        let original_degree = degree + if ascending && !zero { steps } else { 0 };
        let index = (i / inner / remaining * n + original_degree) * inner + i % inner;
        (index, original_degree)
    };
    fn run<T: Num>(
        c: &[T],
        count: usize,
        source: impl Fn(usize) -> (usize, usize),
        ascending: bool,
        n: usize,
        steps: usize,
        zero: bool,
        scale: T,
        factor: impl Fn(usize) -> T,
        token: &CancellationToken,
    ) -> Result<Vec<T>> {
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            if i % CHUNK == 0 {
                token.check()?;
            }
            let (index, degree) = source(i);
            let mut value = c[index];
            if zero {
                // NumPy's c[:1] * 0 deliberately preserves NaN/Inf effects.
                value = T::mul(value, factor(0));
            } else {
                for step in 0..steps {
                    if step % CHUNK == 0 {
                        token.check()?;
                    }
                    if ascending {
                        value = T::mul(value, scale);
                        value = T::mul(factor(degree - step), value);
                    } else {
                        value = T::mul(value, factor(n - 1 - degree - step));
                    }
                }
            }
            out.push(value);
        }
        token.check()?;
        Ok(out)
    }
    let scale = match options.scale {
        DerivativeScale::Real(v) => [v, 0.0],
        DerivativeScale::Complex(v) => v,
    };
    macro_rules! real {
        ($c:expr, $variant:ident, $t:ty) => {
            Buffer::$variant(run(
                $c,
                count,
                source,
                ascending,
                n,
                steps,
                zero,
                scale[0] as $t,
                |v| v as $t,
                token,
            )?)
        };
    }
    let buffer = match c.as_ref() {
        Buffer::Bool(v) => Buffer::Bool(run(
            v,
            count,
            source,
            ascending,
            n,
            steps,
            zero,
            scale[0] != 0.0,
            |v| v != 0,
            token,
        )?),
        Buffer::I8(v) => real!(v, I8, i8),
        Buffer::I16(v) => real!(v, I16, i16),
        Buffer::I32(v) => real!(v, I32, i32),
        Buffer::I64(v) => real!(v, I64, i64),
        Buffer::U8(v) => real!(v, U8, u8),
        Buffer::U16(v) => real!(v, U16, u16),
        Buffer::U32(v) => real!(v, U32, u32),
        Buffer::U64(v) => real!(v, U64, u64),
        Buffer::F16(v) => Buffer::F16(run(
            v,
            count,
            source,
            ascending,
            n,
            steps,
            zero,
            Float16::from_f64(scale[0]),
            |v| Float16::from_f64(v as f64),
            token,
        )?),
        Buffer::F32(v) => real!(v, F32, f32),
        Buffer::F64(v) => real!(v, F64, f64),
        Buffer::C64(v) => Buffer::C64(run(
            v,
            count,
            source,
            ascending,
            n,
            steps,
            zero,
            [scale[0] as f32, scale[1] as f32],
            |v| [v as f32, 0.0],
            token,
        )?),
        Buffer::C128(v) => Buffer::C128(run(
            v,
            count,
            source,
            ascending,
            n,
            steps,
            zero,
            scale,
            |v| [v as f64, 0.0],
            token,
        )?),
        _ => unreachable!("derivative coefficients have a numeric dtype"),
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
    if input.dtype() == DType::Bool || matches!(input.dtype(), DType::Str(_) | DType::Datetime64(_))
    {
        return Err(Error::new(
            ErrorCode::DTypeUnsupported,
            "Unary arithmetic is not supported for bool, string or datetime operands. Use a \
             supported numeric or duration dtype.",
        ));
    }
    let reservation = reserve_for(budget, input.dtype(), input.size())?;
    let buffer = match op {
        UnaryOp::Pos => input.buffer.cast_cancellable(input.dtype(), token)?,
        UnaryOp::Neg => match &input.buffer {
            Buffer::I8(v) => crate::buffer::map_elements(v, token, |x| Num::neg(x))?.into(),
            Buffer::I16(v) => crate::buffer::map_elements(v, token, |x| Num::neg(x))?.into(),
            Buffer::I32(v) => crate::buffer::map_elements(v, token, |x| Num::neg(x))?.into(),
            Buffer::I64(v) => crate::buffer::map_elements(v, token, |x| Num::neg(x))?.into(),
            Buffer::U8(v) => crate::buffer::map_elements(v, token, |x| Num::neg(x))?.into(),
            Buffer::U16(v) => crate::buffer::map_elements(v, token, |x| Num::neg(x))?.into(),
            Buffer::U32(v) => crate::buffer::map_elements(v, token, |x| Num::neg(x))?.into(),
            Buffer::U64(v) => crate::buffer::map_elements(v, token, |x| Num::neg(x))?.into(),
            Buffer::F16(v) => crate::buffer::map_elements(v, token, |x| -x)?.into(),
            Buffer::F32(v) => crate::buffer::map_elements(v, token, |x| -x)?.into(),
            Buffer::F64(v) => crate::buffer::map_elements(v, token, |x| -x)?.into(),
            Buffer::C64(v) => crate::buffer::map_elements(v, token, |x| Num::neg(x))?.into(),
            Buffer::C128(v) => crate::buffer::map_elements(v, token, |x| Num::neg(x))?.into(),
            Buffer::Temporal { dtype, ticks } => Buffer::Temporal {
                dtype: *dtype,
                ticks: crate::buffer::map_elements(ticks, token, |x| x.wrapping_neg())?,
            },
            Buffer::Unicode { .. } | Buffer::Bool(_) => return Err(crate::dtype::unsupported()),
        },
    };
    DenseArray::new(input.shape.clone(), buffer, reservation)
}

/// Normalizes reduction axes; `None` reduces every axis.
fn reduce_axes_with_check(
    rank: usize,
    axes: Option<&[i64]>,
    mut check: impl FnMut() -> Result<()>,
) -> Result<Vec<bool>> {
    let mut reduced = vec![axes.is_none(); rank];
    for (position, &axis) in axes.unwrap_or(&[]).iter().enumerate() {
        if position % CHUNK == 0 {
            check()?;
        }
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
    check()?;
    Ok(reduced)
}

pub fn reduce_axes(rank: usize, axes: Option<&[i64]>) -> Result<Vec<bool>> {
    reduce_axes_with_check(rank, axes, || Ok(()))
}

fn reduce_axes_cancellable(
    rank: usize,
    axes: Option<&[i64]>,
    token: &CancellationToken,
) -> Result<Vec<bool>> {
    reduce_axes_with_check(rank, axes, || token.check())
}

pub fn reduce_shape(shape: &[usize], reduced: &[bool]) -> Vec<usize> {
    let mut result = Vec::with_capacity(reduced.iter().filter(|&&r| !r).count());
    for (&dimension, &is_reduced) in shape.iter().zip(reduced) {
        if !is_reduced {
            result.push(dimension);
        }
    }
    result
}

/// First-occurrence index selection. `axis` stores the reduced axis length and
/// the stride of the trailing dimensions; `None` selects the flat C-order
/// offset for a full reduction. NaNs are sticky and keep the first NaN index,
/// and equal values keep the first index, matching NumPy.
fn arg_reduce<T: Num>(
    op: ReduceOp,
    values: &[T],
    slots: BroadcastPairs,
    out_len: usize,
    axis: Option<(usize, usize)>,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<Vec<i64>> {
    let scratch = out_len
        .checked_mul(std::mem::size_of::<T>() + 1)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| {
            Error::new(
                ErrorCode::MemoryBudgetExceeded,
                "Reduction index scratch exceeds the supported vector capacity.",
            )
        })?;
    let _scratch = budget.reserve(scratch)?;
    let mut best = vec![T::default(); out_len];
    let mut seen = vec![false; out_len];
    let mut index = vec![0i64; out_len];
    for (i, (slot, _)) in slots.enumerate() {
        if i % CHUNK == 0 {
            token.check()?;
        }
        let x = values[i];
        let take = !seen[slot]
            || (!best[slot].nan()
                && (x.nan()
                    || if op == ReduceOp::ArgMin {
                        x < best[slot]
                    } else {
                        x > best[slot]
                    }));
        if take {
            best[slot] = x;
            index[slot] = match axis {
                None => i64::try_from(i).map_err(|_| crate::numerical::overflow())?,
                Some((length, inner)) => {
                    i64::try_from((i / inner) % length).map_err(|_| crate::numerical::overflow())?
                }
            };
            seen[slot] = true;
        }
    }
    Ok(index)
}

trait VarianceSample: Copy {
    fn parts(self) -> [f64; 2];

    fn distance(self, mean: [f64; 2]) -> f64 {
        let parts = self.parts();
        let real = parts[0] - mean[0];
        let imag = parts[1] - mean[1];
        real * real + imag * imag
    }
}

macro_rules! real_sample {
    ($($t:ty),*) => {$(
        impl VarianceSample for $t {
            fn parts(self) -> [f64; 2] { [self as f64, 0.0] }
        }
    )*};
}
real_sample!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

impl VarianceSample for Float16 {
    fn parts(self) -> [f64; 2] {
        [self.to_f32() as f64, 0.0]
    }
}

impl VarianceSample for bool {
    fn parts(self) -> [f64; 2] {
        [if self { 1.0 } else { 0.0 }, 0.0]
    }
}

impl VarianceSample for [f32; 2] {
    fn parts(self) -> [f64; 2] {
        [self[0] as f64, self[1] as f64]
    }
}

impl VarianceSample for [f64; 2] {
    fn parts(self) -> [f64; 2] {
        self
    }
}

fn variance_loop<T: VarianceSample>(
    op: VarianceOp,
    values: &[T],
    shape: &[usize],
    out_strides: &[isize],
    out_len: usize,
    ddof: f64,
    token: &CancellationToken,
) -> Result<Vec<f64>> {
    let mut mean = vec![[0.0f64; 2]; out_len];
    let mut counts = vec![0usize; out_len];
    for (i, (slot, _)) in
        BroadcastPairs::new(shape, out_strides.to_vec(), vec![0; shape.len()]).enumerate()
    {
        if i % CHUNK == 0 {
            token.check()?;
        }
        let parts = values[i].parts();
        mean[slot][0] += parts[0];
        mean[slot][1] += parts[1];
        counts[slot] += 1;
    }
    for (slot, count) in counts.iter().enumerate() {
        if *count != 0 {
            mean[slot][0] /= *count as f64;
            mean[slot][1] /= *count as f64;
        }
    }
    let mut squares = vec![0.0f64; out_len];
    for (i, (slot, _)) in
        BroadcastPairs::new(shape, out_strides.to_vec(), vec![0; shape.len()]).enumerate()
    {
        if i % CHUNK == 0 {
            token.check()?;
        }
        squares[slot] += values[i].distance(mean[slot]);
    }
    Ok(squares
        .into_iter()
        .zip(counts)
        .map(|(squares, count)| {
            // NumPy clamps the degrees of freedom at zero before division, so a
            // nonpositive divisor yields NaN for a zero numerator and infinity
            // for a positive numerator.
            let variance = squares / (count as f64 - ddof).max(0.0);
            if op == VarianceOp::Std {
                variance.sqrt()
            } else {
                variance
            }
        })
        .collect())
}

/// Variance or standard deviation over the flattened input or one axis.
pub fn variance(
    op: VarianceOp,
    input: &DenseArray,
    axis: Option<i64>,
    ddof: f64,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    validate_variance(input.dtype())?;
    if !ddof.is_finite() {
        return Err(Error::new(
            ErrorCode::IndexInvalid,
            "ddof must be a finite number.",
        ));
    }
    let rank = input.shape.len();
    let axes = match axis {
        None => None,
        Some(raw) => {
            let normalized = if raw < 0 { raw + rank as i64 } else { raw };
            if normalized < 0 || normalized >= rank as i64 {
                return Err(Error::new(
                    ErrorCode::IndexOutOfBounds,
                    format!("Axis {raw} is out of bounds for an array of dimension {rank}."),
                ));
            }
            Some(vec![normalized])
        }
    };
    let reduced = reduce_axes_cancellable(rank, axes.as_deref(), token)?;
    let out_shape = reduce_shape(&input.shape, &reduced);
    let out_len: usize = out_shape.iter().try_fold(1usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or_else(|| crate::numerical::overflow())
    })?;
    let metadata = checked_reduction_bytes(rank, 1 + 4 * std::mem::size_of::<usize>())?;
    let _metadata = budget.reserve(metadata)?;
    let scratch = checked_reduction_bytes(out_len, 5 * std::mem::size_of::<f64>())?;
    let _scratch = budget.reserve(scratch)?;
    let dtype = op.result_dtype(input.dtype());
    let shape_bytes = checked_reduction_bytes(out_shape.len(), std::mem::size_of::<usize>())?;
    let data_bytes = checked_reduction_bytes(out_len, dtype.item_size())?;
    let reservation = budget.reserve(
        shape_bytes
            .checked_add(data_bytes)
            .ok_or_else(|| crate::numerical::overflow())?,
    )?;
    let mut out_strides = vec![0isize; rank];
    let mut stride = 1isize;
    for axis in (0..rank).rev() {
        if !reduced[axis] {
            out_strides[axis] = stride;
            stride *= input.shape[axis] as isize;
        }
    }
    token.check()?;
    macro_rules! run {
        ($values:expr) => {
            variance_loop(
                op,
                $values,
                &input.shape,
                &out_strides,
                out_len,
                ddof,
                token,
            )?
        };
    }
    let values = match &input.buffer {
        Buffer::Bool(x) => run!(x),
        Buffer::I8(x) => run!(x),
        Buffer::I16(x) => run!(x),
        Buffer::I32(x) => run!(x),
        Buffer::I64(x) => run!(x),
        Buffer::U8(x) => run!(x),
        Buffer::U16(x) => run!(x),
        Buffer::U32(x) => run!(x),
        Buffer::U64(x) => run!(x),
        Buffer::F16(x) => run!(x),
        Buffer::F32(x) => run!(x),
        Buffer::F64(x) => run!(x),
        Buffer::C64(x) => run!(x),
        Buffer::C128(x) => run!(x),
        Buffer::Temporal { .. } | Buffer::Unicode { .. } => {
            return Err(Error::new(
                ErrorCode::DTypeUnsupported,
                "Variance requires a numeric or boolean dtype. Cast explicitly.",
            ));
        }
    };
    let buffer = match dtype {
        DType::Float16 => Buffer::F16(
            values
                .into_iter()
                .map(|v| Float16::from_f32(v as f32))
                .collect(),
        ),
        DType::Float32 => Buffer::F32(values.into_iter().map(|v| v as f32).collect()),
        DType::Float64 => Buffer::F64(values),
        _ => {
            return Err(Error::new(
                ErrorCode::DTypeUnsupported,
                "Variance result dtype is not floating point.",
            ));
        }
    };
    DenseArray::new(out_shape, buffer, reservation)
}

pub fn validate_variance(dtype: DType) -> Result<()> {
    if matches!(dtype, DType::Str(_)) || dtype.temporal_unit().is_some() {
        return Err(Error::new(
            ErrorCode::DTypeUnsupported,
            "Variance does not support string or temporal arrays.",
        ));
    }
    Ok(())
}

fn scan_loop<T: Num>(
    op: ScanOp,
    values: &[T],
    shape: &[usize],
    axis: Option<usize>,
    token: &CancellationToken,
) -> Result<Vec<T>> {
    let mut out = vec![T::default(); values.len()];
    if values.is_empty() {
        return Ok(out);
    }
    let accumulate = |left: T, right: T| -> T {
        match op {
            ScanOp::CumSum => T::add(left, right),
            ScanOp::CumProd => T::mul(left, right),
        }
    };
    match axis {
        None => {
            let mut accumulator = values[0];
            out[0] = accumulator;
            for (i, &value) in values.iter().enumerate().skip(1) {
                if i % CHUNK == 0 {
                    token.check()?;
                }
                accumulator = accumulate(accumulator, value);
                out[i] = accumulator;
            }
        }
        Some(axis) => {
            let length = shape[axis];
            if length == 0 {
                return Ok(out);
            }
            let inner: usize = shape[axis + 1..].iter().product();
            let outer: usize = shape[..axis].iter().product();
            for batch in 0..outer {
                for lane in 0..inner {
                    let base = batch * length * inner + lane;
                    let mut accumulator = values[base];
                    out[base] = accumulator;
                    for step in 1..length {
                        let i = base + step * inner;
                        if i % CHUNK == 0 {
                            token.check()?;
                        }
                        accumulator = accumulate(accumulator, values[i]);
                        out[i] = accumulator;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Cumulative sum or product over the flattened input or one axis.
pub fn scan(
    op: ScanOp,
    input: &DenseArray,
    axis: Option<i64>,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    validate_scan(input.dtype())?;
    let rank = input.shape.len();
    let axis = match axis {
        None => None,
        Some(raw) => {
            let normalized = if raw < 0 { raw + rank as i64 } else { raw };
            if normalized < 0 || normalized >= rank as i64 {
                return Err(Error::new(
                    ErrorCode::IndexOutOfBounds,
                    format!("Axis {raw} is out of bounds for an array of dimension {rank}."),
                ));
            }
            Some(normalized as usize)
        }
    };
    let out_shape = match axis {
        None => vec![input.size()],
        Some(_) => input.shape.clone(),
    };
    let dtype = op.result_dtype(input.dtype());
    let count = crate::numerical::size(&out_shape)?;
    let data_bytes = checked_reduction_bytes(count, dtype.item_size())?;
    let shape_bytes = checked_reduction_bytes(out_shape.len(), std::mem::size_of::<usize>())?;
    let reservation = budget.reserve(
        data_bytes
            .checked_add(shape_bytes)
            .ok_or_else(|| crate::numerical::overflow())?,
    )?;
    let (values, _cast_reservation) = cast_operand(&input.buffer, dtype, budget, token)?;
    macro_rules! run {
        ($values:expr) => {
            scan_loop(op, $values, &input.shape, axis, token)?
        };
    }
    let buffer = match values.as_ref() {
        Buffer::I8(x) => run!(x).into(),
        Buffer::I16(x) => run!(x).into(),
        Buffer::I32(x) => run!(x).into(),
        Buffer::I64(x) => run!(x).into(),
        Buffer::U8(x) => run!(x).into(),
        Buffer::U16(x) => run!(x).into(),
        Buffer::U32(x) => run!(x).into(),
        Buffer::U64(x) => run!(x).into(),
        Buffer::F16(x) => run!(x).into(),
        Buffer::F32(x) => run!(x).into(),
        Buffer::F64(x) => run!(x).into(),
        Buffer::C64(x) => run!(x).into(),
        Buffer::C128(x) => run!(x).into(),
        Buffer::Bool(_) | Buffer::Temporal { .. } | Buffer::Unicode { .. } => {
            return Err(Error::new(
                ErrorCode::DTypeUnsupported,
                "Cumulative scans require a numeric or boolean dtype. Cast explicitly.",
            ));
        }
    };
    DenseArray::new(out_shape, buffer, reservation)
}

pub fn validate_scan(dtype: DType) -> Result<()> {
    if matches!(dtype, DType::Str(_)) || dtype.temporal_unit().is_some() {
        return Err(Error::new(
            ErrorCode::DTypeUnsupported,
            "Cumulative scans do not support string or temporal arrays.",
        ));
    }
    Ok(())
}

pub fn reduce(
    op: ReduceOp,
    input: &DenseArray,
    axes: Option<&[i64]>,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    validate_special_reduce(op, input.dtype(), axes)?;
    let rank = input.shape.len();
    let reduced = reduce_axes_cancellable(input.shape.len(), axes, token)?;
    // An empty reduced dimension has no min/max identity, regardless of the
    // available budget.
    let empty_reduction = input
        .shape
        .iter()
        .zip(&reduced)
        .any(|(&len, &r)| r && len == 0);
    if (matches!(op, ReduceOp::Min | ReduceOp::Max) || op.is_arg()) && empty_reduction {
        return Err(Error::new(
            ErrorCode::ShapeMismatch,
            if op.is_arg() {
                format!(
                    "The '{}' reduction has no element over a zero-size axis. Use a nonempty \
                     reduced axis.",
                    op.id()
                )
            } else {
                format!(
                    "The '{}' reduction has no identity over a zero-size axis.",
                    op.id()
                )
            },
        ));
    }
    // Reduced flags and the four rank-sized vectors in BroadcastPairs and
    // its stride input all remain live until the reduction buffer is built.
    let metadata_bytes = checked_reduction_bytes(rank, 1 + 4 * std::mem::size_of::<usize>())?;
    let _metadata_reservation = budget.reserve(metadata_bytes)?;
    let kept = reduced.iter().filter(|&&is_reduced| !is_reduced).count();
    let shape_bytes = checked_reduction_bytes(kept, std::mem::size_of::<usize>())?;
    let out_len: usize = input
        .shape
        .iter()
        .zip(&reduced)
        .filter(|(_, is_reduced)| !**is_reduced)
        .map(|(dimension, _)| *dimension)
        .try_fold(1usize, |product, dimension| product.checked_mul(dimension))
        .ok_or_else(|| {
            Error::new(
                ErrorCode::MemoryBudgetExceeded,
                "Reduction result size overflows.",
            )
        })?;
    let data_bytes = checked_reduction_bytes(out_len, op.result_dtype(input.dtype()).item_size())?;
    let reservation = budget.reserve(data_bytes.checked_add(shape_bytes).ok_or_else(|| {
        Error::new(
            ErrorCode::MemoryBudgetExceeded,
            "Reduction result size overflows.",
        )
    })?)?;
    let out_shape = reduce_shape(&input.shape, &reduced);
    let in_dtype = input.dtype();
    let dtype = op.result_dtype(in_dtype);
    // Map each input element to its output slot.
    let mut out_strides = vec![0isize; input.shape.len()];
    let mut acc = 1isize;
    for axis in (0..input.shape.len()).rev() {
        if !reduced[axis] {
            out_strides[axis] = acc;
            acc *= input.shape[axis] as isize;
        }
    }
    token.check()?;
    let slots = BroadcastPairs::new(&input.shape, out_strides, vec![0; input.shape.len()]);
    let count = if out_len == 0 {
        0
    } else {
        input.size() / out_len
    };
    if op.is_arg() {
        // `validate_reduce` admits only a full reduction or one named axis, so
        // the selected coordinate is either the flat C-order offset or the
        // position inside the single reduced axis.
        let axis = axes.as_ref().map(|list| {
            let raw = list[0];
            let position = if raw < 0 { raw + rank as i64 } else { raw } as usize;
            let inner = input.shape[position + 1..]
                .iter()
                .try_fold(1usize, |n, &dimension| n.checked_mul(dimension))
                .unwrap_or(0);
            (input.shape[position], inner)
        });
        let out = match &input.buffer {
            Buffer::Bool(x) => arg_reduce(op, x, slots, out_len, axis, budget, token)?,
            Buffer::I8(x) => arg_reduce(op, x, slots, out_len, axis, budget, token)?,
            Buffer::I16(x) => arg_reduce(op, x, slots, out_len, axis, budget, token)?,
            Buffer::I32(x) => arg_reduce(op, x, slots, out_len, axis, budget, token)?,
            Buffer::I64(x) => arg_reduce(op, x, slots, out_len, axis, budget, token)?,
            Buffer::U8(x) => arg_reduce(op, x, slots, out_len, axis, budget, token)?,
            Buffer::U16(x) => arg_reduce(op, x, slots, out_len, axis, budget, token)?,
            Buffer::U32(x) => arg_reduce(op, x, slots, out_len, axis, budget, token)?,
            Buffer::U64(x) => arg_reduce(op, x, slots, out_len, axis, budget, token)?,
            Buffer::F16(x) => arg_reduce(op, x, slots, out_len, axis, budget, token)?,
            Buffer::F32(x) => arg_reduce(op, x, slots, out_len, axis, budget, token)?,
            Buffer::F64(x) => arg_reduce(op, x, slots, out_len, axis, budget, token)?,
            Buffer::C64(x) => arg_reduce(op, x, slots, out_len, axis, budget, token)?,
            Buffer::C128(x) => arg_reduce(op, x, slots, out_len, axis, budget, token)?,
            Buffer::Temporal { .. } | Buffer::Unicode { .. } => {
                return Err(crate::dtype::unsupported());
            }
        };
        return DenseArray::new(out_shape, Buffer::I64(out), reservation);
    }
    if let Buffer::Temporal { dtype, ticks } = &input.buffer {
        let _seen = budget.reserve(out_len as u64)?;
        let mut seen = vec![false; out_len];
        let mut out = vec![0i64; out_len];
        for (i, (slot, _)) in slots.enumerate() {
            if i % CHUNK == 0 {
                token.check()?;
            }
            let t = ticks[i];
            out[slot] =
                if t == crate::temporal::NAT || (seen[slot] && out[slot] == crate::temporal::NAT) {
                    crate::temporal::NAT
                } else {
                    match op {
                        ReduceOp::Sum | ReduceOp::Mean => {
                            crate::temporal::checked_tick(out[slot].checked_add(t))?
                        }
                        ReduceOp::Min => {
                            if seen[slot] {
                                out[slot].min(t)
                            } else {
                                t
                            }
                        }
                        ReduceOp::Max => {
                            if seen[slot] {
                                out[slot].max(t)
                            } else {
                                t
                            }
                        }
                        // Temporal products and index reductions are rejected
                        // during expression construction.
                        ReduceOp::Prod | ReduceOp::ArgMin | ReduceOp::ArgMax => {
                            return Err(Error::new(
                                ErrorCode::DTypeUnsupported,
                                "This reduction is not supported for temporal arrays.",
                            ));
                        }
                    }
                };
            seen[slot] = true;
        }
        if op == ReduceOp::Mean {
            for x in &mut out {
                if count == 0 {
                    *x = crate::temporal::NAT;
                } else if *x != crate::temporal::NAT {
                    *x /= count as i64;
                }
            }
        }
        return DenseArray::new(
            out_shape,
            Buffer::Temporal {
                dtype: *dtype,
                ticks: out,
            },
            reservation,
        );
    }
    // Borrow matching input storage. Widening for sum/mean is reserved before
    // conversion; no input-sized offset or i128 scratch vectors are needed.
    let (values, _cast_reservation) = cast_operand(&input.buffer, dtype, budget, token)?;
    let _seen_reservation = budget.reserve(if matches!(op, ReduceOp::Min | ReduceOp::Max) {
        out_len as u64
    } else {
        0
    })?;
    fn run<T: Num>(
        op: ReduceOp,
        values: &[T],
        slots: BroadcastPairs,
        out_len: usize,
        count: usize,
        contiguous: bool,
        token: &CancellationToken,
    ) -> Result<Vec<T>> {
        let mut out = if op == ReduceOp::Prod {
            vec![T::one(); out_len]
        } else {
            vec![T::default(); out_len]
        };
        if contiguous && count > 0 && matches!(op, ReduceOp::Sum | ReduceOp::Mean) {
            if T::pairwise_sum(&[], token)?.is_some() {
                for (i, chunk) in values.chunks_exact(count).enumerate() {
                    let total = T::add(T::default(), T::pairwise_sum(chunk, token)?.unwrap());
                    out[i] = if op == ReduceOp::Mean {
                        total.mean(count)
                    } else {
                        total
                    };
                }
                return Ok(out);
            }
        }
        let mut seen = if matches!(op, ReduceOp::Min | ReduceOp::Max) {
            vec![false; out_len]
        } else {
            Vec::new()
        };
        for (i, (slot, _)) in slots.enumerate() {
            if i % CHUNK == 0 {
                token.check()?;
            }
            let x = values[i];
            out[slot] = match op {
                ReduceOp::Sum | ReduceOp::Mean => T::add(out[slot], x),
                ReduceOp::Prod => T::mul(out[slot], x),
                ReduceOp::ArgMin | ReduceOp::ArgMax => {
                    unreachable!("argument reductions use the dedicated index loop")
                }
                _ => {
                    let old = out[slot];
                    let take = !seen[slot]
                        || (!old.nan()
                            && (x.nan()
                                || if op == ReduceOp::Min {
                                    x < old || (T::take_equal() && x == old)
                                } else {
                                    x > old || (T::take_equal() && x == old)
                                }));
                    seen[slot] = true;
                    if take { x } else { old }
                }
            };
        }
        if op == ReduceOp::Mean {
            for (i, value) in out.iter_mut().enumerate() {
                if i % CHUNK == 0 {
                    token.check()?;
                }
                *value = value.mean(count);
            }
        }
        Ok(out)
    }
    // A suffix of reduced dimensions is contiguous in a dense C-order input.
    let first_reduced = reduced.iter().position(|&r| r).unwrap_or(reduced.len());
    let contiguous = reduced[first_reduced..].iter().all(|&r| r);
    let buffer = match values.as_ref() {
        Buffer::Bool(x) => run(op, x, slots, out_len, count, contiguous, token)?.into(),
        Buffer::I8(x) => run(op, x, slots, out_len, count, contiguous, token)?.into(),
        Buffer::I16(x) => run(op, x, slots, out_len, count, contiguous, token)?.into(),
        Buffer::I32(x) => run(op, x, slots, out_len, count, contiguous, token)?.into(),
        Buffer::I64(x) => run(op, x, slots, out_len, count, contiguous, token)?.into(),
        Buffer::U8(x) => run(op, x, slots, out_len, count, contiguous, token)?.into(),
        Buffer::U16(x) => run(op, x, slots, out_len, count, contiguous, token)?.into(),
        Buffer::U32(x) => run(op, x, slots, out_len, count, contiguous, token)?.into(),
        Buffer::U64(x) => run(op, x, slots, out_len, count, contiguous, token)?.into(),
        Buffer::F16(x) if matches!(op, ReduceOp::Sum | ReduceOp::Mean) => {
            let mut out = vec![Float16::ZERO; out_len];
            if contiguous && count > 0 {
                for (i, chunk) in x.chunks_exact(count).enumerate() {
                    let total = 0.0 + half_pairwise(chunk, token)?;
                    out[i] = Float16::from_f32(if op == ReduceOp::Mean {
                        total / count as f32
                    } else {
                        total
                    });
                }
            } else if op == ReduceOp::Mean {
                // Mean retains binary32 accumulators until division, including
                // non-contiguous axes. Charge scratch before allocating it.
                let _scratch = reserve_for(budget, DType::Float32, out_len)?;
                let mut totals = vec![0.0f32; out_len];
                for (i, (slot, _)) in slots.enumerate() {
                    if i % CHUNK == 0 {
                        token.check()?;
                    }
                    totals[slot] += x[i].to_f32();
                }
                for (i, value) in totals.into_iter().enumerate() {
                    if i % CHUNK == 0 {
                        token.check()?;
                    }
                    out[i] = Float16::from_f32(value / count as f32);
                }
            } else {
                for (i, (slot, _)) in slots.enumerate() {
                    if i % CHUNK == 0 {
                        token.check()?;
                    }
                    out[slot] = <Float16 as Num>::add(out[slot], x[i]);
                }
            }
            Buffer::F16(out)
        }
        Buffer::F16(x) => run(op, x, slots, out_len, count, contiguous, token)?.into(),
        Buffer::F32(x) => run(op, x, slots, out_len, count, contiguous, token)?.into(),
        Buffer::F64(x) => run(op, x, slots, out_len, count, contiguous, token)?.into(),
        Buffer::C64(x) => run(op, x, slots, out_len, count, contiguous, token)?.into(),
        Buffer::C128(x) => run(op, x, slots, out_len, count, contiguous, token)?.into(),
        _ => return Err(crate::dtype::unsupported()),
    };
    DenseArray::new(out_shape, buffer, reservation)
}

/// Matrix products with broadcast leading axes and NumPy vector promotion.
pub fn matmul(
    lhs: &DenseArray,
    rhs: &DenseArray,
    dtype: DType,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    if lhs.dtype().kind() == DTypeKind::Other
        || rhs.dtype().kind() == DTypeKind::Other
        || dtype.kind() == DTypeKind::Other
    {
        return Err(crate::dtype::unsupported());
    }
    let out_shape = crate::shape::matmul_shape(&lhs.shape, &rhs.shape)?;
    // Check zero axes first: an empty result needs no matrix offsets, even
    // when multiplication of its other dimensions would overflow.
    let count = if out_shape.contains(&0) {
        0
    } else {
        out_shape.iter().try_fold(1usize, |n, &d| {
            n.checked_mul(d).ok_or_else(crate::numerical::overflow)
        })?
    };
    let reservation = reserve_for(budget, dtype, count)?;
    token.check()?;
    if count == 0 {
        return DenseArray::new(
            out_shape,
            Buffer::filled(dtype, 0, crate::Scalar::Int(0))?,
            reservation,
        );
    }
    let k = *lhs.shape.last().unwrap();
    let n = if lhs.shape.len() > 1 {
        lhs.shape[lhs.shape.len() - 2]
    } else {
        1
    };
    let m = if rhs.shape.len() > 1 {
        *rhs.shape.last().unwrap()
    } else {
        1
    };
    // A nonempty result guarantees n*m fits, but an empty contraction can
    // still have huge logical operand dimensions. Check before indexing.
    let matrix_len = n.checked_mul(m).ok_or_else(crate::numerical::overflow)?;
    let batch = &out_shape
        [..out_shape.len() - usize::from(lhs.shape.len() > 1) - usize::from(rhs.shape.len() > 1)];
    let left_batch = &lhs.shape[..lhs.shape.len().saturating_sub(2)];
    let right_batch = &rhs.shape[..rhs.shape.len().saturating_sub(2)];
    let (a, _a_reservation) = cast_operand(&lhs.buffer, dtype, budget, token)?;
    let (b, _b_reservation) = cast_operand(&rhs.buffer, dtype, budget, token)?;

    // Walk broadcast batch coordinates without expanded operands, offset
    // vectors, or per-batch result allocations. All output is reserved once.
    let fill = |run: &mut dyn FnMut(usize, usize, usize) -> Result<()>| -> Result<()> {
        for i in 0..count / matrix_len {
            token.check()?;
            let ao = matmul_batch_offset(i, left_batch, batch) * n * k;
            let bo = matmul_batch_offset(i, right_batch, batch) * k * m;
            run(ao, bo, i * matrix_len)?;
        }
        Ok(())
    };
    macro_rules! numeric {
        ($x:expr, $y:expr, $t:ty) => {{
            let mut out = vec![<$t>::default(); count];
            fill(&mut |ao, bo, start| {
                let out = &mut out[start..start + matrix_len];
                for i in 0..n {
                    token.check()?;
                    for p in 0..k {
                        let x = $x[ao + i * k + p];
                        for j in 0..m {
                            if (p * m + j) % CHUNK == 0 {
                                token.check()?;
                            }
                            out[i * m + j] = <$t as Num>::add(
                                out[i * m + j],
                                <$t as Num>::mul(x, $y[bo + p * m + j]),
                            );
                        }
                    }
                }
                Ok(())
            })?;
            out.into()
        }};
    }
    let buffer = match (a.as_ref(), b.as_ref()) {
        (Buffer::Bool(x), Buffer::Bool(y)) => {
            let mut out = vec![false; count];
            fill(&mut |ao, bo, start| {
                boolean_matmul_into(
                    &x[ao..ao + n * k],
                    &y[bo..bo + k * m],
                    &mut out[start..start + matrix_len],
                    n,
                    k,
                    m,
                    token,
                    || {},
                )
            })?;
            Buffer::Bool(out)
        }
        (Buffer::I8(x), Buffer::I8(y)) => numeric!(x, y, i8),
        (Buffer::I16(x), Buffer::I16(y)) => numeric!(x, y, i16),
        (Buffer::I32(x), Buffer::I32(y)) => numeric!(x, y, i32),
        (Buffer::I64(x), Buffer::I64(y)) => numeric!(x, y, i64),
        (Buffer::U8(x), Buffer::U8(y)) => numeric!(x, y, u8),
        (Buffer::U16(x), Buffer::U16(y)) => numeric!(x, y, u16),
        (Buffer::U32(x), Buffer::U32(y)) => numeric!(x, y, u32),
        (Buffer::U64(x), Buffer::U64(y)) => numeric!(x, y, u64),
        (Buffer::F16(x), Buffer::F16(y)) => {
            let mut out = vec![Float16::ZERO; count];
            fill(&mut |ao, bo, start| {
                for i in 0..n {
                    for j in 0..m {
                        token.check()?;
                        let mut total = 0.0f32;
                        for p in 0..k {
                            if p % CHUNK == 0 {
                                token.check()?;
                            }
                            total += x[ao + i * k + p].to_f32() * y[bo + p * m + j].to_f32();
                        }
                        out[start + i * m + j] = Float16::from_f32(total);
                    }
                }
                Ok(())
            })?;
            Buffer::F16(out)
        }
        (Buffer::F32(x), Buffer::F32(y)) => numeric!(x, y, f32),
        (Buffer::F64(x), Buffer::F64(y)) => numeric!(x, y, f64),
        (Buffer::C64(x), Buffer::C64(y)) => numeric!(x, y, [f32; 2]),
        (Buffer::C128(x), Buffer::C128(y)) => numeric!(x, y, [f64; 2]),
        _ => unreachable!(),
    };
    DenseArray::new(out_shape, buffer, reservation)
}

/// Map a linear output batch index to an operand's broadcast batch index.
/// The output is nonempty, so every divisor is positive.
fn matmul_batch_offset(mut index: usize, source: &[usize], output: &[usize]) -> usize {
    let mut offset = 0;
    let mut stride = 1;
    for axis in (0..output.len()).rev() {
        let coordinate = index % output[axis];
        index /= output[axis];
        if let Some(i) = axis.checked_sub(output.len() - source.len()) {
            if source[i] != 1 {
                offset += coordinate * stride;
            }
            stride *= source[i];
        }
    }
    offset
}

#[cfg(test)]
fn boolean_matmul(
    a: &[bool],
    b: &[bool],
    n: usize,
    k: usize,
    m: usize,
    token: &CancellationToken,
    visit_product: impl FnMut(),
) -> Result<Vec<bool>> {
    let mut out = vec![false; n * m];
    boolean_matmul_into(a, b, &mut out, n, k, m, token, visit_product)?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn boolean_matmul_into(
    a: &[bool],
    b: &[bool],
    out: &mut [bool],
    n: usize,
    k: usize,
    m: usize,
    token: &CancellationToken,
    mut visit_product: impl FnMut(),
) -> Result<()> {
    for i in 0..n {
        for j in 0..m {
            token.check()?;
            for p in 0..k {
                if p % CHUNK == 0 {
                    token.check()?;
                }
                visit_product();
                if a[i * k + p] && b[p * m + j] {
                    out[i * m + j] = true;
                    break;
                }
            }
        }
    }
    Ok(())
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
    fn half_mean_scratch_is_charged_and_released() {
        let input = dense(vec![2, 2], Buffer::F16(vec![Float16::from_f32(0.5); 4]));
        let token = CancellationToken::new();
        // Rank metadata, retained result shape/data, and two float32 scratch
        // values.
        let metadata = (2 * (1 + 4 * std::mem::size_of::<usize>())) as u64;
        let retained = 4 + std::mem::size_of::<usize>() as u64;
        let peak = metadata + retained + 8;
        let tight = MemoryBudget::new(Device::Cpu, peak - 1);
        assert_eq!(
            reduce(ReduceOp::Mean, &input, Some(&[0]), &tight, &token)
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded
        );
        assert_eq!(tight.usage().used, 0);
        let fits = MemoryBudget::new(Device::Cpu, peak);
        let result = reduce(ReduceOp::Mean, &input, Some(&[0]), &fits, &token).unwrap();
        assert_eq!(result.buffer, Buffer::F16(vec![Float16::from_f32(0.5); 2]));
        assert_eq!(fits.usage().peak, peak);
        assert_eq!(fits.usage().used, retained);
        drop(result);
        assert_eq!(fits.usage().used, 0);
        token.cancel();
        assert_eq!(
            reduce(ReduceOp::Sum, &input, None, &fits, &token)
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
        assert_eq!(fits.usage().used, 0);
    }

    #[test]
    fn half_accumulation_does_not_round_each_contiguous_addition() {
        let input = dense(
            vec![3],
            Buffer::F16(vec![Float16::MAX, Float16::ONE, -Float16::MAX]),
        );
        let (budget, token) = env();
        let sum = reduce(ReduceOp::Sum, &input, None, &budget, &token).unwrap();
        assert_eq!(sum.buffer, Buffer::F16(vec![Float16::ONE]));
        let ones = dense(vec![3], Buffer::F16(vec![Float16::ONE; 3]));
        let dot = matmul(&input, &ones, DType::Float16, &budget, &token).unwrap();
        assert_eq!(dot.buffer, Buffer::F16(vec![Float16::ONE]));
        let unchanged = reduce(ReduceOp::Mean, &input, Some(&[]), &budget, &token).unwrap();
        assert_eq!(unchanged.buffer, input.buffer);
    }

    #[test]
    fn reductions_reserve_casts_and_release_scratch() {
        assert_eq!(
            checked_reduction_bytes(usize::MAX, std::mem::size_of::<i64>())
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded
        );
        let input = dense(vec![2, 2], Buffer::I32(vec![1, 2, 3, 4]));
        let token = CancellationToken::new();
        for op in [ReduceOp::Sum, ReduceOp::Mean] {
            // Rank metadata (66), output data and shape (24), and cast (32).
            for limit in [65, 89, 121] {
                let tight = MemoryBudget::new(Device::Cpu, limit);
                assert_eq!(
                    reduce(op, &input, Some(&[0]), &tight, &token)
                        .unwrap_err()
                        .code,
                    ErrorCode::MemoryBudgetExceeded
                );
                assert_eq!(tight.usage().used, 0);
            }
            let fits = MemoryBudget::new(Device::Cpu, 122);
            let result = reduce(op, &input, Some(&[0]), &fits, &token).unwrap();
            assert_eq!(fits.usage().peak, 122);
            assert_eq!(fits.usage().used, 24);
            drop(result);
            assert_eq!(fits.usage().used, 0);
        }
        // Min/max borrow the input, with one byte per output for
        // initialization.
        let tight = MemoryBudget::new(Device::Cpu, 83);
        assert_eq!(
            reduce(ReduceOp::Min, &input, Some(&[0]), &tight, &token)
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded
        );
        assert_eq!(tight.usage().used, 0);
        let fits = MemoryBudget::new(Device::Cpu, 84);
        let result = reduce(ReduceOp::Max, &input, Some(&[0]), &fits, &token).unwrap();
        assert_eq!(fits.usage().peak, 84);
        assert_eq!(fits.usage().used, 16);
        drop(result);
        assert_eq!(fits.usage().used, 0);

        let large = dense(vec![100_000], Buffer::I32(vec![1; 100_000]));
        let tiny = MemoryBudget::new(Device::Cpu, 8);
        assert_eq!(
            reduce(ReduceOp::Sum, &large, None, &tiny, &token)
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded
        );
        assert_eq!(tiny.usage().used, 0);
    }

    #[test]
    fn complex_reduction_borrows_without_input_sized_scratch() {
        let input = dense(vec![100_000], Buffer::C64(vec![[1.0, 2.0]; 100_000]));
        let budget = MemoryBudget::new(Device::Cpu, 41);
        let result = reduce(
            ReduceOp::Sum,
            &input,
            None,
            &budget,
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(result.buffer, Buffer::C64(vec![[100_000.0, 200_000.0]]));
        assert_eq!(budget.usage().peak, 41);
        drop(result);
        assert_eq!(budget.usage().used, 0);
    }

    #[test]
    fn float32_pairwise_sum_and_mean_preserve_precision_and_budget() {
        let values = vec![0.1_f32; 1_000_000];
        let input = dense(vec![1_000_000], Buffer::F32(values.clone()));
        // One rank of metadata (33 bytes) and one float32 result (4 bytes).
        let budget = MemoryBudget::new(Device::Cpu, 37);
        let token = CancellationToken::new();
        let sum = reduce(ReduceOp::Sum, &input, None, &budget, &token).unwrap();
        let Buffer::F32(sums) = &sum.buffer else {
            panic!("Expected float32 sum")
        };
        assert_eq!(sums[0], 100_000.007_812_5); // NumPy 2.5.4, float32 sum.
        assert!((sums[0] - 100_000.0).abs() < 1.0);
        assert_ne!(sums[0], 100_958.343_75);
        assert_eq!(budget.usage().peak, 37);
        drop(sum);
        let mean = reduce(ReduceOp::Mean, &input, None, &budget, &token).unwrap();
        let Buffer::F32(means) = &mean.buffer else {
            panic!("Expected float32 mean")
        };
        assert_eq!(means[0], 0.100_000_008_940_696_72); // NumPy 2.5.4, float32 mean.
        assert!((means[0] - 0.1).abs() < 0.000_001);
        assert_ne!(means[0], 0.100_958_347_320_556_64);
        drop(mean);
        assert_eq!(budget.usage().used, 0);

        let rows = dense(
            vec![2, 1_000_000],
            Buffer::F32([values.clone(), values].concat()),
        );
        // Two ranks of metadata (66 bytes), result shape (8), and data (8).
        let budget = MemoryBudget::new(Device::Cpu, 82);
        for op in [ReduceOp::Sum, ReduceOp::Mean] {
            let result = reduce(op, &rows, Some(&[1]), &budget, &token).unwrap();
            assert_eq!(result.shape, vec![2]);
            let Buffer::F32(actual) = &result.buffer else {
                panic!("Expected float32 rows")
            };
            let target = if op == ReduceOp::Sum { 100_000.0 } else { 0.1 };
            let tolerance = if op == ReduceOp::Sum { 1.0 } else { 0.000_001 };
            assert!(
                actual
                    .iter()
                    .all(|&value| (value - target).abs() < tolerance)
            );
            let reference = if op == ReduceOp::Sum {
                100_000.007_812_5
            } else {
                0.100_000_008_940_696_72
            };
            assert_eq!(actual, &vec![reference; 2]);
            assert_eq!(budget.usage().peak, 82);
            drop(result);
        }
    }

    #[test]
    fn float32_pairwise_matches_numpy_at_block_boundaries() {
        // NumPy 2.5.4 float32 sums for [1e8, 1, ..., 1, -1e8].
        let (budget, token) = env();
        for (len, expected) in [
            (7, 0.0),
            (8, 0.0),
            (9, 0.0),
            (127, 112.0),
            (128, 112.0),
            (129, 120.0),
            (256, 240.0),
            (257, 240.0),
        ] {
            let mut values = vec![1.0_f32; len];
            values[0] = 100_000_000.0;
            values[len - 1] = -100_000_000.0;
            let input = dense(vec![len], Buffer::F32(values));
            assert_eq!(
                reduce(ReduceOp::Sum, &input, None, &budget, &token)
                    .unwrap()
                    .buffer,
                Buffer::F32(vec![expected])
            );
        }
        token.cancel();
        let input = dense(vec![129], Buffer::F32(vec![1.0; 129]));
        assert_eq!(
            reduce(ReduceOp::Sum, &input, None, &budget, &token)
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
    }

    #[test]
    fn float32_mean_keeps_float32_overflow_and_axis_shape() {
        let (budget, token) = env();
        let max = f32::MAX;
        let pair = dense(vec![2], Buffer::F32(vec![max, max]));
        for op in [ReduceOp::Sum, ReduceOp::Mean] {
            let result = reduce(op, &pair, None, &budget, &token).unwrap();
            assert_eq!(result.shape, Vec::<usize>::new());
            assert_eq!(result.buffer, Buffer::F32(vec![f32::INFINITY]));
        }
        let rows = dense(vec![2, 2], Buffer::F32(vec![max, max, 2.0, 4.0]));
        let mean = reduce(ReduceOp::Mean, &rows, Some(&[1]), &budget, &token).unwrap();
        assert_eq!(
            (mean.shape, mean.buffer),
            (vec![2], Buffer::F32(vec![f32::INFINITY, 3.0]))
        );
        let doubles = dense(vec![2], Buffer::F64(vec![max as f64, max as f64]));
        assert_eq!(
            reduce(ReduceOp::Mean, &doubles, None, &budget, &token)
                .unwrap()
                .buffer,
            Buffer::F64(vec![max as f64])
        );
        let ints = dense(vec![2], Buffer::I32(vec![2, 4]));
        assert_eq!(
            reduce(ReduceOp::Mean, &ints, None, &budget, &token)
                .unwrap()
                .buffer,
            Buffer::F64(vec![3.0])
        );
    }

    #[test]
    fn matmul_reserves_only_required_casts_and_releases_them() {
        let a = dense(vec![2], Buffer::I32(vec![1, 2]));
        let b = dense(vec![2], Buffer::C128(vec![[1.0, 2.0], [3.0, 4.0]]));
        let token = CancellationToken::new();
        let tight = MemoryBudget::new(Device::Cpu, 47);
        assert_eq!(
            matmul(&a, &b, DType::Complex128, &tight, &token)
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded
        );
        assert_eq!(tight.usage().used, 0);
        let fits = MemoryBudget::new(Device::Cpu, 48);
        let result = matmul(&a, &b, DType::Complex128, &fits, &token).unwrap();
        assert_eq!(result.buffer, Buffer::C128(vec![[7.0, 10.0]]));
        assert_eq!(fits.usage().peak, 48);
        assert_eq!(fits.usage().used, 16);
        drop(result);
        assert_eq!(fits.usage().used, 0);
        let borrowed = MemoryBudget::new(Device::Cpu, 16);
        let result = matmul(&b, &b, DType::Complex128, &borrowed, &token).unwrap();
        assert_eq!(borrowed.usage().peak, 16);
        drop(result);
        assert_eq!(borrowed.usage().used, 0);
    }

    #[test]
    fn cancelled_reductions_matmul_and_casts_release_allocations() {
        let input = dense(vec![2], Buffer::C64(vec![[1.0, 2.0]; 2]));
        let (budget, token) = env();
        token.cancel();
        for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Min, ReduceOp::Max] {
            assert_eq!(
                reduce(op, &input, None, &budget, &token).unwrap_err().code,
                ErrorCode::Cancelled
            );
            assert_eq!(budget.usage().used, 0);
        }
        assert_eq!(
            matmul(&input, &input, DType::Complex128, &budget, &token)
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
        assert_eq!(budget.usage().used, 0);
        assert_eq!(
            input
                .buffer
                .cast_cancellable(DType::Complex128, &token)
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
    }

    #[test]
    fn reduction_axis_validation_honors_cancellation() {
        let rank = 100_000;
        let input = dense(vec![1; rank], Buffer::I32(vec![1]));
        let budget = MemoryBudget::new(Device::Cpu, u64::MAX);
        let token = CancellationToken::new();
        token.cancel();
        let axes: Vec<i64> = (0..rank as i64).collect();
        assert_eq!(
            reduce(ReduceOp::Sum, &input, Some(&axes), &budget, &token)
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
        assert_eq!(budget.usage().used, 0);
        assert_eq!(
            reduce_axes(3, Some(&[0, -3])).unwrap_err().code,
            ErrorCode::IndexInvalid
        );
    }

    #[test]
    fn cancellation_interrupts_a_running_complex_matmul() {
        let input = dense(vec![512, 512], Buffer::C64(vec![[1.0, 2.0]; 512 * 512]));
        let (budget, token) = env();
        let cancel = token.clone();
        let thread = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(1));
            cancel.cancel();
        });
        let result = matmul(&input, &input, DType::Complex64, &budget, &token);
        thread.join().unwrap();
        assert_eq!(result.unwrap_err().code, ErrorCode::Cancelled);
        assert_eq!(budget.usage().used, 0);
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
        let zero_budget = MemoryBudget::new(Device::Cpu, 0);
        let empty_vector = dense(vec![0], Buffer::F64(vec![]));
        for op in [ReduceOp::Min, ReduceOp::Max] {
            assert_eq!(
                reduce(op, &empty_vector, None, &zero_budget, &t)
                    .unwrap_err()
                    .code,
                ErrorCode::ShapeMismatch
            );
            assert_eq!(zero_budget.usage().used, 0);
        }
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
    fn batched_matmul_broadcasts_without_materializing_repeated_operands() {
        let a = dense(vec![2, 1, 2, 2], Buffer::I32((1..=8).collect()));
        let b = dense(vec![3, 2, 1], Buffer::I32(vec![1, 2, 3, 4, 5, 6]));
        // Exactly the output bytes fit. Any expanded operand or per-batch
        // temporary would exceed this budget.
        let budget = MemoryBudget::new(Device::Cpu, 48);
        let out = matmul(&a, &b, DType::Int32, &budget, &CancellationToken::new()).unwrap();
        assert_eq!(out.shape, vec![2, 3, 2, 1]);
        assert_eq!(
            out.buffer,
            Buffer::I32(vec![5, 11, 11, 25, 17, 39, 17, 23, 39, 53, 61, 83])
        );
        assert_eq!(budget.usage().peak, 48);
        drop(out);
        assert_eq!(budget.usage().used, 0);
        let tight = MemoryBudget::new(Device::Cpu, 47);
        assert_eq!(
            matmul(&a, &b, DType::Int32, &tight, &CancellationToken::new())
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded
        );
        assert_eq!(tight.usage().used, 0);
    }

    #[test]
    fn batched_matmul_vectors_booleans_half_and_complex() {
        let (budget, token) = env();
        let a = dense(vec![2, 2, 2], Buffer::I32((1..=8).collect()));
        let v = dense(vec![2], Buffer::I32(vec![1, 2]));
        let out = matmul(&a, &v, DType::Int32, &budget, &token).unwrap();
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.buffer, Buffer::I32(vec![5, 11, 17, 23]));
        let out = matmul(&v, &a, DType::Int32, &budget, &token).unwrap();
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.buffer, Buffer::I32(vec![7, 10, 19, 22]));
        let a = dense(vec![2, 1, 2], Buffer::Bool(vec![true, false, false, true]));
        let b = dense(vec![2], Buffer::Bool(vec![false, true]));
        assert_eq!(
            matmul(&a, &b, DType::Bool, &budget, &token).unwrap().buffer,
            Buffer::Bool(vec![false, true])
        );
        let a = dense(
            vec![2, 1, 3],
            Buffer::F16(vec![
                Float16::MAX,
                Float16::ONE,
                -Float16::MAX,
                Float16::ONE,
                Float16::ONE,
                Float16::ONE,
            ]),
        );
        let b = dense(vec![3], Buffer::F16(vec![Float16::ONE; 3]));
        assert_eq!(
            matmul(&a, &b, DType::Float16, &budget, &token)
                .unwrap()
                .buffer,
            Buffer::F16(vec![Float16::ONE, Float16::from_f32(3.0)])
        );
        let a = dense(vec![2, 1, 1], Buffer::C128(vec![[1.0, 2.0], [3.0, -1.0]]));
        let b = dense(vec![1], Buffer::C128(vec![[2.0, 1.0]]));
        assert_eq!(
            matmul(&a, &b, DType::Complex128, &budget, &token)
                .unwrap()
                .buffer,
            Buffer::C128(vec![[0.0, 5.0], [7.0, 1.0]])
        );
    }

    #[test]
    fn batched_matmul_empty_axes_errors_and_cancellation() {
        let (budget, token) = env();
        let a = dense(vec![2, 3, 0], Buffer::I32(vec![]));
        let b = dense(vec![1, 0, 4], Buffer::I32(vec![]));
        let out = matmul(&a, &b, DType::Int32, &budget, &token).unwrap();
        assert_eq!(out.shape, vec![2, 3, 4]);
        assert_eq!(out.buffer, Buffer::I32(vec![0; 24]));
        let a = dense(vec![0, 3, 2], Buffer::I32(vec![]));
        let b = dense(vec![1, 2, 4], Buffer::I32(vec![1; 8]));
        assert_eq!(
            matmul(&a, &b, DType::Int32, &budget, &token).unwrap().shape,
            vec![0, 3, 4]
        );
        for (a, b, code) in [
            (vec![], vec![1], ErrorCode::ShapeMismatch),
            (vec![1], vec![], ErrorCode::ShapeMismatch),
            (vec![2, 3, 4], vec![2, 5, 6], ErrorCode::ShapeMismatch),
            (vec![2, 3, 4], vec![3, 4, 6], ErrorCode::ShapeBroadcast),
        ] {
            assert_eq!(crate::shape::matmul_shape(&a, &b).unwrap_err().code, code);
        }
        let overflow_left = dense(vec![usize::MAX, 0], Buffer::I32(vec![]));
        let overflow_right = dense(vec![0, 2], Buffer::I32(vec![]));
        assert_eq!(
            matmul(
                &overflow_left,
                &overflow_right,
                DType::Int32,
                &budget,
                &token
            )
            .unwrap_err()
            .code,
            ErrorCode::MemoryBudgetExceeded
        );
        token.cancel();
        assert_eq!(
            matmul(&a, &b, DType::Int32, &budget, &token)
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
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
    fn boolean_matmul_stops_at_each_first_true_product() {
        let token = CancellationToken::new();
        for first_true in [0, 3, 7] {
            let mut lhs = vec![false; 8];
            lhs[first_true] = true;
            let rhs = vec![true; 8];
            let mut evaluated = 0;
            let result = boolean_matmul(&lhs, &rhs, 1, 8, 1, &token, || evaluated += 1).unwrap();
            assert_eq!(result, vec![true]);
            assert_eq!(evaluated, first_true + 1);
        }
        let mut evaluated = 0;
        let result =
            boolean_matmul(&[false; 8], &[true; 8], 1, 8, 1, &token, || evaluated += 1).unwrap();
        assert_eq!(result, vec![false]);
        assert_eq!(evaluated, 8);
    }

    #[test]
    fn all_true_boolean_matmul_evaluates_one_product_per_output() {
        let (budget, token) = env();
        let lhs = dense(vec![128, 4096], Buffer::Bool(vec![true; 128 * 4096]));
        let rhs = dense(vec![4096, 128], Buffer::Bool(vec![true; 4096 * 128]));
        let mut evaluated = 0;
        let Buffer::Bool(lhs_values) = &lhs.buffer else {
            unreachable!()
        };
        let Buffer::Bool(rhs_values) = &rhs.buffer else {
            unreachable!()
        };
        let values = boolean_matmul(lhs_values, rhs_values, 128, 4096, 128, &token, || {
            evaluated += 1
        })
        .unwrap();
        assert_eq!(values, vec![true; 128 * 128]);
        assert_eq!(evaluated, 128 * 128);

        let result = matmul(&lhs, &rhs, DType::Bool, &budget, &token).unwrap();
        assert_eq!(result.shape, vec![128, 128]);
        assert_eq!(result.dtype(), DType::Bool);
        assert_eq!(result.buffer, Buffer::Bool(values));
    }

    #[test]
    fn boolean_matmul_preserves_shapes_and_strict_result_budget() {
        let token = CancellationToken::new();
        let vector = dense(vec![2], Buffer::Bool(vec![true, false]));
        let matrix = dense(vec![2, 2], Buffer::Bool(vec![true, false, false, true]));
        for (lhs, rhs, shape, values) in [
            (&vector, &vector, vec![], vec![true]),
            (&vector, &matrix, vec![2], vec![true, false]),
            (&matrix, &vector, vec![2], vec![true, false]),
            (&matrix, &matrix, vec![2, 2], vec![true, false, false, true]),
        ] {
            let budget = MemoryBudget::new(Device::Cpu, values.len() as u64);
            let result = matmul(lhs, rhs, DType::Bool, &budget, &token).unwrap();
            assert_eq!(result.shape, shape);
            assert_eq!(result.dtype(), DType::Bool);
            assert_eq!(result.buffer, Buffer::Bool(values));
            assert_eq!(budget.usage().peak, budget.usage().used);
            drop(result);
            assert_eq!(budget.usage().used, 0);
        }
        let tight = MemoryBudget::new(Device::Cpu, 3);
        assert_eq!(
            matmul(&matrix, &matrix, DType::Bool, &tight, &token)
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded
        );
        assert_eq!(tight.usage().used, 0);

        let empty_inner = dense(vec![2, 0], Buffer::Bool(vec![]));
        let empty_inner_rhs = dense(vec![0, 3], Buffer::Bool(vec![]));
        let budget = MemoryBudget::new(Device::Cpu, 6);
        let result = matmul(&empty_inner, &empty_inner_rhs, DType::Bool, &budget, &token).unwrap();
        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(result.buffer, Buffer::Bool(vec![false; 6]));
        let empty_rows = dense(vec![0, 2], Buffer::Bool(vec![]));
        let result = matmul(&empty_rows, &matrix, DType::Bool, &budget, &token).unwrap();
        assert_eq!(result.shape, vec![0, 2]);
        assert_eq!(result.buffer, Buffer::Bool(vec![]));
    }

    #[test]
    fn boolean_matmul_cancels_during_an_unsuccessful_dot_product() {
        let token = CancellationToken::new();
        let budget = MemoryBudget::new(Device::Cpu, 1);
        let k = CHUNK * 3;
        let lhs = dense(vec![1, k], Buffer::Bool(vec![false; k]));
        let rhs = dense(vec![k, 1], Buffer::Bool(vec![true; k]));
        token.cancel();
        assert_eq!(
            matmul(&lhs, &rhs, DType::Bool, &budget, &token)
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
        assert_eq!(budget.usage().used, 0);

        let token = CancellationToken::new();
        let cancel = token.clone();
        let mut evaluated = 0;
        let Buffer::Bool(lhs_values) = &lhs.buffer else {
            unreachable!()
        };
        let Buffer::Bool(rhs_values) = &rhs.buffer else {
            unreachable!()
        };
        let result = boolean_matmul(lhs_values, rhs_values, 1, k, 1, &token, || {
            evaluated += 1;
            if evaluated == CHUNK + 1 {
                cancel.cancel();
            }
        });
        assert_eq!(result.unwrap_err().code, ErrorCode::Cancelled);
        assert!(evaluated <= 2 * CHUNK);
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
/// Operation-specific loops are needed because datetime subtraction changes
/// kind.
pub fn special_binary_dtype(op: BinaryOp, a: DType, b: DType) -> Result<DType> {
    use BinaryOp::*;
    use DType::*;

    use crate::dtype::unsupported;
    match (a, b, op) {
        (Str(x), Str(y), Add) => Ok(Str(x
            .checked_add(y)
            .filter(|n| *n <= usize::MAX / 4)
            .ok_or_else(unsupported)?)),
        (Datetime64(x), Timedelta64(y), Add | Sub) | (Timedelta64(y), Datetime64(x), Add) => {
            Ok(Datetime64(x.common(y)?))
        }
        (Datetime64(x), Datetime64(y), Sub) => Ok(Timedelta64(x.common(y)?)),
        (Timedelta64(x), Timedelta64(y), Add | Sub | Mod) => Ok(Timedelta64(x.common(y)?)),
        (Timedelta64(x), Timedelta64(y), Div) => {
            x.common(y)?;
            Ok(Float64)
        }
        (Timedelta64(_), n, Mul | Div) if n.kind() == DTypeKind::Int => Ok(a),
        (n, Timedelta64(_), Mul) if n.kind() == DTypeKind::Int => Ok(b),
        _ => Err(unsupported()),
    }
}
fn special_binary(
    op: BinaryOp,
    lhs: &DenseArray,
    rhs: &DenseArray,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    use crate::{
        buffer::Scalar,
        dtype::unsupported,
        temporal::{NAT, convert},
    };
    token.check()?;
    let dtype = special_binary_dtype(op, lhs.dtype(), rhs.dtype())?;
    let shape = broadcast_shapes(&lhs.shape, &rhs.shape)?;
    let size = shape.iter().product();
    let reservation = reserve_for(budget, dtype, size)?;
    let pairs = BroadcastPairs::new(
        &shape,
        broadcast_strides(&lhs.shape, &shape),
        broadcast_strides(&rhs.shape, &shape),
    );
    if let DType::Str(width) = dtype {
        let (
            Buffer::Unicode {
                width: aw,
                values: av,
            },
            Buffer::Unicode {
                width: bw,
                values: bv,
            },
        ) = (&lhs.buffer, &rhs.buffer)
        else {
            unreachable!()
        };
        let mut values = vec![0; size * width];
        for (n, (i, j)) in pairs.enumerate() {
            if n % CHUNK == 0 {
                token.check()?;
            }
            let a = &av[i * aw..(i + 1) * aw];
            let b = &bv[j * bw..(j + 1) * bw];
            let an = a.iter().rposition(|&c| c != 0).map_or(0, |x| x + 1);
            let bn = b.iter().rposition(|&c| c != 0).map_or(0, |x| x + 1);
            values[n * width..n * width + an].copy_from_slice(&a[..an]);
            values[n * width + an..n * width + an + bn].copy_from_slice(&b[..bn]);
        }
        return DenseArray::new(shape, Buffer::Unicode { width, values }, reservation);
    }
    let unit = match (lhs.dtype().temporal_unit(), rhs.dtype().temporal_unit()) {
        (Some(a), Some(b)) => a.common(b)?,
        (Some(a), None) | (None, Some(a)) => a,
        _ => unreachable!(),
    };
    for dtype in [lhs.dtype(), rhs.dtype()] {
        if let Some(from) = dtype.temporal_unit() {
            convert(NAT, from, unit, matches!(dtype, DType::Datetime64(_)))?;
        }
    }
    let tick = |buffer: &Buffer, i| -> Result<i64> {
        match buffer.get(i) {
            Scalar::Temporal(t, d) => convert(
                t,
                d.temporal_unit().unwrap(),
                unit,
                matches!(d, DType::Datetime64(_)),
            ),
            Scalar::Int(t) => Ok(t),
            // NumPy's temporal integer loop casts uint64 operands to int64
            // by preserving their bits. Do not reject the operand before the
            // operation: zero multiplication and small quotients remain valid.
            // checked_tick below still rejects result overflow and NaT collisions.
            Scalar::UInt(t) => Ok(t as i64),
            _ => Err(unsupported()),
        }
    };
    let mut out = Vec::with_capacity(if dtype == DType::Float64 { 0 } else { size });
    let mut floats = Vec::with_capacity(if dtype == DType::Float64 { size } else { 0 });
    for (n, (i, j)) in pairs.enumerate() {
        if n % CHUNK == 0 {
            token.check()?;
        }
        let (a, b) = (tick(&lhs.buffer, i)?, tick(&rhs.buffer, j)?);
        let nat = (lhs.dtype().temporal_unit().is_some() && a == NAT)
            || (rhs.dtype().temporal_unit().is_some() && b == NAT);
        if dtype == DType::Float64 {
            floats.push(if nat { f64::NAN } else { a as f64 / b as f64 });
            continue;
        }
        out.push(if nat {
            NAT
        } else {
            match op {
                BinaryOp::Add => crate::temporal::checked_tick(a.checked_add(b))?,
                BinaryOp::Sub => crate::temporal::checked_tick(a.checked_sub(b))?,
                BinaryOp::Mul => crate::temporal::checked_tick(a.checked_mul(b))?,
                BinaryOp::Div => {
                    if b == 0 {
                        NAT
                    } else {
                        a.wrapping_div(b)
                    }
                }
                BinaryOp::Mod => {
                    if b == 0 {
                        NAT
                    } else {
                        let r = a.wrapping_rem(b);
                        if r != 0 && (r < 0) != (b < 0) {
                            r.wrapping_add(b)
                        } else {
                            r
                        }
                    }
                }
                _ => return Err(unsupported()),
            }
        });
    }
    DenseArray::new(
        shape,
        if dtype == DType::Float64 {
            Buffer::F64(floats)
        } else {
            Buffer::Temporal { dtype, ticks: out }
        },
        reservation,
    )
}

pub fn validate_special_reduce(op: ReduceOp, dtype: DType, axes: Option<&[i64]>) -> Result<()> {
    let temporal = dtype.temporal_unit().is_some();
    if matches!(dtype, DType::Str(_))
        || (matches!(dtype, DType::Datetime64(_)) && matches!(op, ReduceOp::Sum | ReduceOp::Mean))
        || (temporal && matches!(op, ReduceOp::Prod | ReduceOp::ArgMin | ReduceOp::ArgMax))
    {
        return Err(crate::dtype::unsupported());
    }
    if op.is_arg() && axes.is_some_and(|list| list.len() != 1) {
        return Err(Error::new(
            ErrorCode::IndexInvalid,
            format!(
                "The '{}' reduction selects at most one axis. Pass one integer axis or omit it.",
                op.id()
            ),
        ));
    }
    Ok(())
}
