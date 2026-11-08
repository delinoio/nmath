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
    /// Cholesky factor; `upper` returns the conjugate transpose.
    Cholesky {
        upper: bool,
    },
    /// QR factor `Q`; `complete` selects NumPy mode `"complete"`.
    QrQ {
        complete: bool,
    },
    /// QR factor `R`; `complete` selects NumPy mode `"complete"`.
    QrR {
        complete: bool,
    },
    /// Ascending eigenvalues of a Hermitian matrix read from one triangle.
    EighValues {
        upper: bool,
    },
    /// Eigenvectors (columns) matching `EighValues`.
    EighVectors {
        upper: bool,
    },
    /// Left singular vectors; `full` selects `full_matrices=True`.
    SvdU {
        full: bool,
    },
    /// Descending singular values.
    SvdS,
    /// Conjugate-transposed right singular vectors.
    SvdVh {
        full: bool,
    },
    /// `slogdet` sign.
    SlogdetSign,
    /// `slogdet` natural log of the absolute determinant.
    SlogdetLogabs,
    /// Vector norm over the final axis or matrix norm over the final two.
    Norm(NormKind),
    /// Eigenvalues of a general matrix, always complex.
    EigValues,
    /// Right eigenvectors (columns) matching `EigValues`, always complex.
    EigVectors,
}

/// NumPy `norm` orders. Vector kinds reduce the final axis; matrix kinds
/// reduce the final two axes (rows, then columns).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    VectorInf,
    VectorNegInf,
    VectorZero,
    VectorOne,
    VectorTwo,
    /// Any other finite order, stored as `f64` bits.
    VectorP(u64),
    Fro,
    Nuc,
    MatrixOne,
    MatrixNegOne,
    MatrixTwo,
    MatrixNegTwo,
    MatrixInf,
    MatrixNegInf,
}

impl NormKind {
    fn is_vector(self) -> bool {
        matches!(
            self,
            Self::VectorInf
                | Self::VectorNegInf
                | Self::VectorZero
                | Self::VectorOne
                | Self::VectorTwo
                | Self::VectorP(_)
        )
    }
}

fn empty_reduction() -> Error {
    Error::new(
        ErrorCode::ShapeMismatch,
        "The norm order takes a maximum or minimum over an empty axis.",
    )
}

fn infer_norm(kind: NormKind, a: &[usize], ad: DType) -> Result<(Vec<usize>, DType)> {
    let dtype = real_dtype(result_dtype(ad, None)?);
    let r = a.len();
    if kind.is_vector() {
        if r < 1 {
            return Err(invalid("A vector norm requires at least one dimension."));
        }
        if a[r - 1] == 0 && matches!(kind, NormKind::VectorInf | NormKind::VectorNegInf) {
            return Err(empty_reduction());
        }
        return Ok((a[..r - 1].to_vec(), dtype));
    }
    if r < 2 {
        return Err(invalid("A matrix norm requires at least two dimensions."));
    }
    let (m, n) = (a[r - 2], a[r - 1]);
    let empty = match kind {
        NormKind::MatrixOne | NormKind::MatrixNegOne => n == 0,
        NormKind::MatrixInf | NormKind::MatrixNegInf => m == 0,
        NormKind::MatrixTwo | NormKind::MatrixNegTwo => m.min(n) == 0,
        _ => false,
    };
    if empty {
        return Err(empty_reduction());
    }
    Ok((a[..r - 2].to_vec(), dtype))
}

fn vector_norm(kind: NormKind, x: &[C]) -> f64 {
    let abs = x.iter().map(|z| z.norm());
    match kind {
        NormKind::VectorInf => abs.fold(
            f64::NEG_INFINITY,
            |a, b| if b > a || b.is_nan() { b } else { a },
        ),
        NormKind::VectorNegInf => abs.fold(
            f64::INFINITY,
            |a, b| if b < a || b.is_nan() { b } else { a },
        ),
        NormKind::VectorZero => x.iter().filter(|z| **z != C::new(0.0, 0.0)).count() as f64,
        NormKind::VectorOne => abs.sum(),
        NormKind::VectorTwo => x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt(),
        NormKind::VectorP(bits) => {
            let p = f64::from_bits(bits);
            abs.map(|v| v.powf(p)).sum::<f64>().powf(1.0 / p)
        }
        _ => unreachable!("matrix norm"),
    }
}

fn matrix_norm(
    kind: NormKind,
    x: &[C],
    m: usize,
    n: usize,
    token: &CancellationToken,
) -> Result<f64> {
    let pick = |values: Vec<f64>, max: bool| {
        values.into_iter().fold(
            if max {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            },
            |a, b| {
                if b.is_nan() || (max && b > a) || (!max && b < a) {
                    b
                } else {
                    a
                }
            },
        )
    };
    Ok(match kind {
        NormKind::Fro => x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt(),
        NormKind::MatrixOne | NormKind::MatrixNegOne => {
            let sums = (0..n)
                .map(|j| (0..m).map(|i| x[i * n + j].norm()).sum())
                .collect();
            pick(sums, kind == NormKind::MatrixOne)
        }
        NormKind::MatrixInf | NormKind::MatrixNegInf => {
            let sums = (0..m)
                .map(|i| (0..n).map(|j| x[i * n + j].norm()).sum())
                .collect();
            pick(sums, kind == NormKind::MatrixInf)
        }
        NormKind::Nuc | NormKind::MatrixTwo | NormKind::MatrixNegTwo => {
            let (_, s, _) = crate::decomp::svd(x, m, n, false, token)?;
            match kind {
                NormKind::Nuc => s.iter().sum(),
                NormKind::MatrixTwo => pick(s, true),
                _ => pick(s, false),
            }
        }
        _ => unreachable!("vector norm"),
    })
}

fn execute_norm(
    kind: NormKind,
    a: &DenseArray,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    let (shape, dtype) = infer_norm(kind, &a.shape, a.buffer.dtype())?;
    let count = size(&shape)?;
    let charge = budget.reserve(bytes(count, dtype.item_size())?)?;
    let mut output = zeros(dtype, count)?;
    let r = a.shape.len();
    let (m, n) = if kind.is_vector() {
        (1, a.shape[r - 1])
    } else {
        (a.shape[r - 2], a.shape[r - 1])
    };
    let block = m.checked_mul(n).ok_or_else(overflow)?;
    let side = m.max(n);
    let _workspace = budget.reserve(bytes(
        block
            .checked_add(
                side.checked_mul(side)
                    .ok_or_else(overflow)?
                    .checked_mul(2)
                    .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?,
        std::mem::size_of::<C>(),
    )?)?;
    let mut x = vec![C::new(0., 0.); block];
    for out in 0..count {
        token.check()?;
        for (i, v) in x.iter_mut().enumerate() {
            if i % 1024 == 0 {
                token.check()?;
            }
            *v = read(&a.buffer, out * block + i);
        }
        let value = if kind.is_vector() {
            vector_norm(kind, &x)
        } else {
            matrix_norm(kind, &x, m, n, token)?
        };
        write(&mut output, out, C::new(value, 0.0));
    }
    token.check()?;
    DenseArray::new(shape, output, charge)
}
impl LinalgOp {
    pub fn id(self) -> &'static str {
        match self {
            Self::Solve => "solve",
            Self::Inv => "inv",
            Self::Det => "det",
            Self::Cholesky { .. } => "cholesky",
            Self::QrQ { .. } | Self::QrR { .. } => "qr",
            Self::EighValues { .. } => "eigvalsh",
            Self::EighVectors { .. } => "eigh",
            Self::SvdU { .. } | Self::SvdVh { .. } => "svd",
            Self::SvdS => "svdvals",
            Self::SlogdetSign | Self::SlogdetLogabs => "slogdet",
            Self::Norm(_) => "norm",
            Self::EigValues => "eigvals",
            Self::EigVectors => "eig",
        }
    }

    fn is_decomposition(self) -> bool {
        !matches!(self, Self::Solve | Self::Inv | Self::Det | Self::Norm(_))
    }

    fn square(self) -> bool {
        !matches!(
            self,
            Self::QrQ { .. }
                | Self::QrR { .. }
                | Self::SvdU { .. }
                | Self::SvdS
                | Self::SvdVh { .. }
        )
    }
}

/// NumPy returns real eigenvalues for real input only when every eigenvalue
/// is real, which depends on the data. Lazy nodes need a static dtype, so
/// `eig` always uses the complex dtype of the input precision.
fn complex_dtype(dtype: DType) -> DType {
    match dtype {
        DType::Float32 => DType::Complex64,
        DType::Float64 => DType::Complex128,
        other => other,
    }
}

fn real_dtype(dtype: DType) -> DType {
    match dtype {
        DType::Complex64 => DType::Float32,
        DType::Complex128 => DType::Float64,
        other => other,
    }
}

#[derive(Clone, Copy)]
struct SignedLog {
    sign: f64,
    mantissa: f64,
    exponent: f64,
}

impl SignedLog {
    fn zero() -> Self {
        Self {
            sign: 0.0,
            mantissa: 0.0,
            exponent: 0.0,
        }
    }

    fn from_value(value: f64) -> Self {
        if value == 0.0 {
            Self::zero()
        } else if !value.is_finite() {
            Self {
                sign: value.signum(),
                mantissa: value,
                exponent: 0.0,
            }
        } else {
            let bits = value.abs().to_bits();
            let raw_exponent = (bits >> 52) & 0x7ff;
            let (fraction, exponent) = if raw_exponent == 0 {
                let scaled = value.abs() * 2_f64.powi(1074);
                let scaled_bits = scaled.to_bits();
                (
                    scaled_bits & ((1_u64 << 52) - 1),
                    ((scaled_bits >> 52) & 0x7ff) as f64 - 1023.0 - 1074.0,
                )
            } else {
                (bits & ((1_u64 << 52) - 1), raw_exponent as f64 - 1023.0)
            };
            Self {
                sign: value.signum(),
                mantissa: 1.0 + fraction as f64 / (1_u64 << 52) as f64,
                exponent,
            }
        }
    }

    fn normalized(sign: f64, mut mantissa: f64, mut exponent: f64) -> Self {
        if mantissa == 0.0 {
            return Self::zero();
        }
        if !mantissa.is_finite() {
            return Self {
                sign,
                mantissa,
                exponent,
            };
        }
        while mantissa >= 2.0 {
            mantissa *= 0.5;
            exponent += 1.0;
        }
        while mantissa < 1.0 {
            mantissa *= 2.0;
            exponent -= 1.0;
        }
        Self {
            sign,
            mantissa,
            exponent,
        }
    }

    fn neg(self) -> Self {
        Self {
            sign: -self.sign,
            mantissa: self.mantissa,
            exponent: self.exponent,
        }
    }

    fn add(self, other: Self) -> Self {
        if self.sign == 0.0 {
            return other;
        }
        if other.sign == 0.0 {
            return self;
        }
        if self.sign == other.sign {
            let (large, small) = if self.exponent >= other.exponent {
                (self, other)
            } else {
                (other, self)
            };
            let difference = small.exponent - large.exponent;
            if difference < -1074.0 {
                return large;
            }
            return Self::normalized(
                large.sign,
                large.mantissa + small.mantissa * 2_f64.powf(difference),
                large.exponent,
            );
        }
        let (large, small) = if self.exponent > other.exponent
            || (self.exponent == other.exponent && self.mantissa >= other.mantissa)
        {
            (self, other)
        } else {
            (other, self)
        };
        let difference = small.exponent - large.exponent;
        if difference < -1074.0 {
            return large;
        }
        let mantissa = large.mantissa - small.mantissa * 2_f64.powf(difference);
        Self::normalized(large.sign, mantissa, large.exponent)
    }

    fn mul(self, other: Self) -> Self {
        if self.sign == 0.0 || other.sign == 0.0 {
            Self::zero()
        } else {
            Self::normalized(
                self.sign * other.sign,
                self.mantissa * other.mantissa,
                self.exponent + other.exponent,
            )
        }
    }

    fn value(self) -> f64 {
        if self.sign == 0.0 {
            0.0
        } else {
            self.sign * (self.mantissa.ln() + self.exponent * std::f64::consts::LN_2).exp()
        }
    }
}

#[derive(Clone, Copy)]
struct LogComplex {
    re: SignedLog,
    im: SignedLog,
}

impl LogComplex {
    fn one() -> Self {
        Self {
            re: SignedLog::from_value(1.0),
            im: SignedLog::zero(),
        }
    }

    fn neg(self) -> Self {
        Self {
            re: self.re.neg(),
            im: self.im.neg(),
        }
    }

    fn mul_assign(&mut self, rhs: C) {
        let re = self.re;
        let im = self.im;
        let rhs_re = SignedLog::from_value(rhs.re);
        let rhs_im = SignedLog::from_value(rhs.im);
        self.re = re.mul(rhs_re).add(im.mul(rhs_im).neg());
        self.im = re.mul(rhs_im).add(im.mul(rhs_re));
    }

    fn value(self) -> C {
        C::new(self.re.value(), self.im.value())
    }
}

/// Result shape and dtype of a factorization of `a` (..., m, n).
fn infer_decomposition(op: LinalgOp, a: &[usize], ad: DType) -> Result<(Vec<usize>, DType)> {
    if a.len() < 2 {
        return Err(invalid(
            "linalg requires at least two dimensions; the final two axes form the matrix.",
        ));
    }
    let (m, n) = (a[a.len() - 2], a[a.len() - 1]);
    if op.square() && m != n {
        return Err(invalid(
            "linalg requires square matrices on the final two axes.",
        ));
    }
    let k = m.min(n);
    let dtype = result_dtype(ad, None)?;
    let mut shape = a[..a.len() - 2].to_vec();
    let dtype = match op {
        LinalgOp::Cholesky { .. } | LinalgOp::EighVectors { .. } => {
            shape.extend([n, n]);
            dtype
        }
        LinalgOp::QrQ { complete } => {
            shape.extend([m, if complete { m } else { k }]);
            dtype
        }
        LinalgOp::QrR { complete } => {
            shape.extend([if complete { m } else { k }, n]);
            dtype
        }
        LinalgOp::EighValues { .. } => {
            shape.push(n);
            real_dtype(dtype)
        }
        LinalgOp::SvdU { full } => {
            shape.extend([m, if full { m } else { k }]);
            dtype
        }
        LinalgOp::SvdS => {
            shape.push(k);
            real_dtype(dtype)
        }
        LinalgOp::SvdVh { full } => {
            shape.extend([if full { n } else { k }, n]);
            dtype
        }
        LinalgOp::SlogdetSign => dtype,
        LinalgOp::EigValues => {
            shape.push(n);
            complex_dtype(dtype)
        }
        LinalgOp::EigVectors => {
            shape.extend([n, n]);
            complex_dtype(dtype)
        }
        LinalgOp::SlogdetLogabs => real_dtype(dtype),
        LinalgOp::Solve | LinalgOp::Inv | LinalgOp::Det | LinalgOp::Norm(_) => {
            unreachable!("not a factorization")
        }
    };
    size(&shape)?;
    Ok((shape, dtype))
}

/// Runs one factorization per batch matrix and writes the selected output.
fn execute_decomposition(
    op: LinalgOp,
    a: &DenseArray,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    let (shape, dtype) = infer_decomposition(op, &a.shape, a.buffer.dtype())?;
    let count = size(&shape)?;
    let charge = budget.reserve(bytes(count, dtype.item_size())?)?;
    let mut output = zeros(dtype, count)?;
    let (m, n) = (a.shape[a.shape.len() - 2], a.shape[a.shape.len() - 1]);
    let matrix = m.checked_mul(n).ok_or_else(overflow)?;
    let batches = size(&a.shape[..a.shape.len() - 2])?;
    if batches == 0 || count == 0 {
        return DenseArray::new(shape, output, charge);
    }
    let per_batch = count / batches;
    // Input copy, two m×m or n×n factors and the matrix-sized result.
    let side = m.max(n);
    let square = side.checked_mul(side).ok_or_else(overflow)?;
    let workspace = matrix
        .checked_add(square.checked_mul(3).ok_or_else(overflow)?)
        .ok_or_else(overflow)?;
    let _workspace = budget.reserve(bytes(workspace, std::mem::size_of::<C>())?)?;
    let mut input = vec![C::new(0., 0.); matrix];
    for batch in 0..batches {
        token.check()?;
        for (i, v) in input.iter_mut().enumerate() {
            if i % 1024 == 0 {
                token.check()?;
            }
            *v = read(&a.buffer, batch * matrix + i);
        }
        let base = batch * per_batch;
        let values: Vec<C> = match op {
            LinalgOp::Cholesky { upper } => {
                let l = crate::decomp::cholesky(&input, n, token)?;
                if upper {
                    (0..n * n).map(|i| l[(i % n) * n + i / n].conj()).collect()
                } else {
                    l
                }
            }
            LinalgOp::QrQ { complete } | LinalgOp::QrR { complete } => {
                let (q, r) = crate::decomp::qr(&input, m, n, token)?;
                let k = if complete { m } else { m.min(n) };
                if matches!(op, LinalgOp::QrQ { .. }) {
                    (0..m * k).map(|i| q[(i / k) * m + i % k]).collect()
                } else {
                    (0..k * n).map(|i| r[i]).collect()
                }
            }
            LinalgOp::EighValues { upper } | LinalgOp::EighVectors { upper } => {
                let (w, v) = crate::decomp::eigh(&input, n, upper, token)?;
                if matches!(op, LinalgOp::EighValues { .. }) {
                    w.into_iter().map(|x| C::new(x, 0.0)).collect()
                } else {
                    v
                }
            }
            LinalgOp::SvdU { .. } | LinalgOp::SvdS | LinalgOp::SvdVh { .. } => {
                let full = matches!(
                    op,
                    LinalgOp::SvdU { full: true } | LinalgOp::SvdVh { full: true }
                );
                let (u, s, vh) = crate::decomp::svd(&input, m, n, full, token)?;
                match op {
                    LinalgOp::SvdU { .. } => u,
                    LinalgOp::SvdS => s.into_iter().map(|x| C::new(x, 0.0)).collect(),
                    _ => vh,
                }
            }
            LinalgOp::EigValues | LinalgOp::EigVectors => {
                let (w, v) = crate::decomp::eig(&input, n, token)?;
                if op == LinalgOp::EigValues { w } else { v }
            }
            LinalgOp::SlogdetSign | LinalgOp::SlogdetLogabs => {
                let (sign, log) = crate::decomp::slogdet(&input, n, token)?;
                vec![if op == LinalgOp::SlogdetSign {
                    sign
                } else {
                    C::new(log, 0.0)
                }]
            }
            LinalgOp::Solve | LinalgOp::Inv | LinalgOp::Det | LinalgOp::Norm(_) => {
                unreachable!("not a factorization")
            }
        };
        debug_assert_eq!(values.len(), per_batch);
        for (i, v) in values.into_iter().enumerate() {
            write(&mut output, base + i, v);
        }
    }
    token.check()?;
    DenseArray::new(shape, output, charge)
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
    if let LinalgOp::Norm(kind) = op {
        if b.is_some() {
            return Err(invalid("Only solve accepts a right-hand side."));
        }
        return infer_norm(kind, a, ad);
    }
    if op.is_decomposition() {
        if b.is_some() {
            return Err(invalid("Only solve accepts a right-hand side."));
        }
        return infer_decomposition(op, a, ad);
    }
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
        _ => {
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
    if let LinalgOp::Norm(kind) = op {
        return execute_norm(kind, a, budget, token);
    }
    if op.is_decomposition() {
        return execute_decomposition(op, a, budget, token);
    }
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
        _ => shape.len() - if vector { 1 } else { 2 },
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
        let mut determinant_product = LogComplex::one();
        let mut singular_determinant = false;
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
                    singular_determinant = true;
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
                if op == LinalgOp::Det {
                    determinant_product = determinant_product.neg();
                }
            }
            let diagonal = lu[k * n + k];
            if op == LinalgOp::Det {
                // Accumulate real and imaginary components in signed log form.
                // A shared normalized phase can underflow a tiny component (for
                // example, 1e-100 / 1e308) before reconstruction can recover
                // it. Component-wise logs avoid both that loss
                // and an overflowing intermediate complex
                // magnitude.
                determinant_product.mul_assign(diagonal);
            }
            for i in k + 1..n {
                token.check()?;
                let factor = lu[i * n + k].fdiv(diagonal);
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
            let determinant = if singular_determinant {
                C::new(0., 0.)
            } else {
                determinant_product.value()
            };
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
                rhs[i * cols + j] = v.fdiv(lu[i * n + i]);
            }
        }
        for (i, &v) in rhs.iter().enumerate() {
            write(&mut output, batch_id * rhs_len + i, v);
        }
    }
    token.check()?;
    DenseArray::new(shape, output, charge)
}
