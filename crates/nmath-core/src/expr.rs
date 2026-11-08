//! Immutable lazy expression graphs with construction-time shape and dtype
//! inference.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    array::NdArray,
    buffer::{Buffer, DenseArray, Scalar, checked_byte_len},
    dtype::DType,
    error::{Error, ErrorCode, Result},
    kernels::{self, BinaryOp, ReduceOp, ScanOp, ShiftOp, UnaryOp, VarianceOp},
    memory::{Device, MemoryBudget, Reservation},
    random::{Generator, Pcg64, RandomOperation},
    shape::{Selector, broadcast_shapes, index_shape},
    sync::CancellationToken,
};

#[derive(Debug)]
pub enum ExprKind {
    Input(NdArray),
    Scalar(Scalar),
    Unary(UnaryOp, Expr),
    Binary {
        op: BinaryOp,
        lhs: Expr,
        rhs: Expr,
        compute_dtype: DType,
    },
    /// NumPy `where(condition, x, y)`; both branches are cast to the node
    /// dtype.
    Where(Expr, Expr, Expr),
    /// Elementwise NumPy math ufunc.
    Math(crate::ufunc::MathOp, Expr),
    Reduce(ReduceOp, Expr, Option<Vec<i64>>),
    Scan(ScanOp, Expr, Option<i64>),
    Variance(VarianceOp, Expr, Option<i64>, f64),
    Shift(ShiftOp, Expr, Vec<usize>),
    /// NumPy `concatenate` along a normalized axis; every part is cast to the
    /// node dtype.
    Concat(Vec<Expr>, usize),
    MatMul(Expr, Expr),
    Linalg(crate::linalg::LinalgOp, Expr, Option<Expr>),
    Fft(crate::fft::FftOptions, Expr),
    Polynomial(crate::polynomial::PolynomialOptions, Expr, Expr),
    PolynomialDerivative(crate::polynomial::DerivativeOptions, Expr),
    Index(Expr, Vec<Selector>),
    Cast(Expr, DType),
    Transpose(Expr, Vec<usize>),
    Reshape(Expr, crate::layout::CopyPolicy),
    /// `Generator.random(shape)` in float64.
    Random(Generator, Vec<usize>, RandomOperation),
}

#[derive(Debug)]
struct Node {
    kind: ExprKind,
    dtype: DType,
    shape: Vec<usize>,
    context_id: Option<u64>,
}

/// A lazy expression. Cloning shares the node.
#[derive(Clone, Debug)]
pub struct Expr(Arc<Node>);

fn join_context(a: Option<u64>, b: Option<u64>) -> Result<Option<u64>> {
    match (a, b) {
        (Some(x), Some(y)) if x != y => Err(Error::new(
            ErrorCode::ContextMismatch,
            format!("Values from context {x} and context {y} cannot be combined directly."),
        )),
        (Some(x), _) | (_, Some(x)) => Ok(Some(x)),
        (None, None) => Ok(None),
    }
}

impl Expr {
    fn node(kind: ExprKind, dtype: DType, shape: Vec<usize>, context_id: Option<u64>) -> Expr {
        Expr(Arc::new(Node {
            kind,
            dtype,
            shape,
            context_id,
        }))
    }

    pub fn input(array: &NdArray) -> Expr {
        Expr::node(
            ExprKind::Input(array.clone()),
            array.dtype(),
            array.shape().to_vec(),
            Some(array.context_id()),
        )
    }

    pub fn scalar(value: Scalar) -> Expr {
        let dtype = value.default_dtype();
        Expr::node(ExprKind::Scalar(value), dtype, vec![], None)
    }

    pub fn random(context_id: u64, generator: &Generator, shape: Vec<usize>) -> Expr {
        Expr::node(
            ExprKind::Random(
                generator.clone(),
                shape.clone(),
                RandomOperation::Random(DType::Float64),
            ),
            DType::Float64,
            shape,
            Some(context_id),
        )
    }

    pub fn random_operation(
        context_id: u64,
        generator: &Generator,
        shape: Vec<usize>,
        operation: RandomOperation,
    ) -> Result<Expr> {
        operation.validate()?;
        Ok(Expr::node(
            ExprKind::Random(generator.clone(), shape.clone(), operation.clone()),
            operation.dtype(),
            shape,
            Some(context_id),
        ))
    }

    pub fn dtype(&self) -> DType {
        self.0.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    pub fn context_id(&self) -> Option<u64> {
        self.0.context_id
    }

    pub fn kind(&self) -> &ExprKind {
        &self.0.kind
    }

    fn weak_scalar(&self) -> Option<Scalar> {
        match &self.0.kind {
            ExprKind::Scalar(s) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn binary(op: BinaryOp, lhs: &Expr, rhs: &Expr) -> Result<Expr> {
        let context_id = join_context(lhs.context_id(), rhs.context_id())?;
        let shape = broadcast_shapes(lhs.shape(), rhs.shape())?;
        if lhs.dtype().kind() == crate::dtype::DTypeKind::Other
            || rhs.dtype().kind() == crate::dtype::DTypeKind::Other
        {
            let dtype = kernels::special_binary_dtype(op, lhs.dtype(), rhs.dtype())?;
            return Ok(Expr::node(
                ExprKind::Binary {
                    op,
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    compute_dtype: dtype,
                },
                dtype,
                shape,
                context_id,
            ));
        }
        let compute_dtype = match (lhs.weak_scalar(), rhs.weak_scalar()) {
            (Some(_), Some(_)) | (None, None) if op.is_math() => {
                crate::ufunc::binary_loop_dtype(op, lhs.dtype(), rhs.dtype())?
            }
            (Some(_), Some(_)) | (None, None) => DType::promote(lhs.dtype(), rhs.dtype())?,
            (Some(s), None) => weak_scalar_dtype(rhs.dtype(), s)?,
            (None, Some(s)) => weak_scalar_dtype(lhs.dtype(), s)?,
        };
        let dtype = op.result_dtype(compute_dtype)?;
        Ok(Expr::node(
            ExprKind::Binary {
                op,
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                compute_dtype,
            },
            dtype,
            shape,
            context_id,
        ))
    }

    /// NumPy `where`: branch dtypes promote with NEP 50 weak scalars and all
    /// three operands broadcast together.
    pub fn select(condition: &Expr, x: &Expr, y: &Expr) -> Result<Expr> {
        let context_id = join_context(
            join_context(condition.context_id(), x.context_id())?,
            y.context_id(),
        )?;
        kernels::validate_select(condition.dtype(), x.dtype(), y.dtype())?;
        let shape = broadcast_shapes(&broadcast_shapes(condition.shape(), x.shape())?, y.shape())?;
        let dtype = match (x.weak_scalar(), y.weak_scalar()) {
            (Some(_), Some(_)) | (None, None) => DType::promote(x.dtype(), y.dtype())?,
            (Some(s), None) => weak_scalar_dtype(y.dtype(), s)?,
            (None, Some(s)) => weak_scalar_dtype(x.dtype(), s)?,
        };
        Ok(Expr::node(
            ExprKind::Where(condition.clone(), x.clone(), y.clone()),
            dtype,
            shape,
            context_id,
        ))
    }

    pub fn math(op: crate::ufunc::MathOp, input: &Expr) -> Result<Expr> {
        let dtype = op.result_dtype(input.dtype())?;
        Ok(Expr::node(
            ExprKind::Math(op, input.clone()),
            dtype,
            input.shape().to_vec(),
            input.context_id(),
        ))
    }

    pub fn unary(op: UnaryOp, input: &Expr) -> Result<Expr> {
        let dtype = input.dtype().ensure_computable()?;
        if dtype == DType::Bool || matches!(dtype, DType::Str(_) | DType::Datetime64(_)) {
            return Err(Error::new(
                ErrorCode::DTypeUnsupported,
                "Unary arithmetic is not supported for bool, string or datetime operands. Use a \
                 supported numeric or duration dtype.",
            ));
        }
        Ok(Expr::node(
            ExprKind::Unary(op, input.clone()),
            dtype,
            input.shape().to_vec(),
            input.context_id(),
        ))
    }

    pub fn reduce(op: ReduceOp, input: &Expr, axes: Option<Vec<i64>>) -> Result<Expr> {
        kernels::validate_special_reduce(op, input.dtype(), axes.as_deref())?;
        let reduced = kernels::reduce_axes(input.shape().len(), axes.as_deref())?;
        let shape = kernels::reduce_shape(input.shape(), &reduced);
        let dtype = op.result_dtype(input.dtype().ensure_computable()?);
        Ok(Expr::node(
            ExprKind::Reduce(op, input.clone(), axes),
            dtype,
            shape,
            input.context_id(),
        ))
    }

    pub fn shift(op: ShiftOp, input: &Expr, axes: Option<Vec<i64>>) -> Result<Expr> {
        let rank = input.shape().len() as i64;
        let mut normalized =
            Vec::with_capacity(axes.as_ref().map_or(input.shape().len(), Vec::len));
        for raw in axes.unwrap_or_else(|| (0..rank).collect()) {
            let axis = if raw < 0 { raw + rank } else { raw };
            if axis < 0 || axis >= rank {
                return Err(Error::new(
                    ErrorCode::IndexOutOfBounds,
                    format!("Axis {raw} is out of bounds for an array of dimension {rank}."),
                ));
            }
            let axis = axis as usize;
            if normalized.contains(&axis) {
                return Err(Error::new(
                    ErrorCode::IndexInvalid,
                    format!("Axis {raw} is repeated."),
                ));
            }
            normalized.push(axis);
        }
        Ok(Expr::node(
            ExprKind::Shift(op, input.clone(), normalized),
            input.dtype(),
            input.shape().to_vec(),
            input.context_id(),
        ))
    }

    /// NumPy `concatenate(parts, axis)`: the parts promote to one dtype, keep
    /// one rank and match every dimension except `axis`.
    pub fn concat(parts: &[Expr], axis: i64) -> Result<Expr> {
        let first = parts.first().ok_or_else(|| {
            Error::new(
                ErrorCode::ShapeMismatch,
                "Concatenation needs at least one array.",
            )
        })?;
        let rank = first.shape().len();
        if rank == 0 {
            return Err(Error::new(
                ErrorCode::ShapeMismatch,
                "Zero-dimensional arrays cannot be concatenated.",
            ));
        }
        let normalized = if axis < 0 { axis + rank as i64 } else { axis };
        if normalized < 0 || normalized >= rank as i64 {
            return Err(Error::new(
                ErrorCode::IndexOutOfBounds,
                format!("Axis {axis} is out of bounds for an array of dimension {rank}."),
            ));
        }
        let axis = normalized as usize;
        let mut shape = first.shape().to_vec();
        let mut context_id = first.context_id();
        for (i, part) in parts.iter().enumerate().skip(1) {
            context_id = join_context(context_id, part.context_id())?;
            if part.shape().len() != rank {
                return Err(Error::new(
                    ErrorCode::ShapeMismatch,
                    format!(
                        "All input arrays must have the same number of dimensions, but the array \
                         at index 0 has {rank} dimension(s) and the array at index {i} has {} \
                         dimension(s).",
                        part.shape().len()
                    ),
                ));
            }
            for (dim, (&expected, &actual)) in first.shape().iter().zip(part.shape()).enumerate() {
                if dim != axis && expected != actual {
                    return Err(Error::new(
                        ErrorCode::ShapeMismatch,
                        format!(
                            "All input array dimensions except for the concatenation axis must \
                             match exactly, but along dimension {dim}, the array at index 0 has \
                             size {expected} and the array at index {i} has size {actual}."
                        ),
                    ));
                }
            }
            shape[axis] = shape[axis]
                .checked_add(part.shape()[axis])
                .ok_or_else(crate::numerical::overflow)?;
        }
        // Public shape transport uses unsigned 32-bit dimensions.
        if shape[axis] > u32::MAX as usize {
            return Err(Error::new(
                ErrorCode::ShapeMismatch,
                "The joined dimension exceeds the supported range.",
            ));
        }
        let dtypes: Vec<DType> = parts.iter().map(Expr::dtype).collect();
        let dtype = DType::promote_all(&dtypes)?;
        crate::numerical::size(&shape)?;
        Ok(Expr::node(
            ExprKind::Concat(parts.to_vec(), axis),
            dtype,
            shape,
            context_id,
        ))
    }

    pub fn variance(op: VarianceOp, input: &Expr, axis: Option<i64>, ddof: f64) -> Result<Expr> {
        kernels::validate_variance(input.dtype().ensure_computable()?)?;
        if !ddof.is_finite() {
            return Err(Error::new(
                ErrorCode::IndexInvalid,
                "ddof must be a finite number.",
            ));
        }
        let rank = input.shape().len();
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
                Some(normalized)
            }
        };
        let reduced = kernels::reduce_axes(rank, axis.map(|a| vec![a]).as_deref())?;
        let shape = kernels::reduce_shape(input.shape(), &reduced);
        let dtype = op.result_dtype(input.dtype().ensure_computable()?);
        Ok(Expr::node(
            ExprKind::Variance(op, input.clone(), axis, ddof),
            dtype,
            shape,
            input.context_id(),
        ))
    }

    pub fn scan(op: ScanOp, input: &Expr, axis: Option<i64>) -> Result<Expr> {
        kernels::validate_scan(input.dtype().ensure_computable()?)?;
        let rank = input.shape().len();
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
                Some(normalized)
            }
        };
        let shape = match axis {
            None => vec![crate::numerical::size(input.shape())?],
            Some(_) => input.shape().to_vec(),
        };
        let dtype = op.result_dtype(input.dtype().ensure_computable()?);
        Ok(Expr::node(
            ExprKind::Scan(op, input.clone(), axis),
            dtype,
            shape,
            input.context_id(),
        ))
    }

    pub fn matmul(lhs: &Expr, rhs: &Expr) -> Result<Expr> {
        let context_id = join_context(lhs.context_id(), rhs.context_id())?;
        if lhs.dtype().kind() == crate::dtype::DTypeKind::Other
            || rhs.dtype().kind() == crate::dtype::DTypeKind::Other
        {
            return Err(crate::dtype::unsupported());
        }
        let dtype = DType::promote(lhs.dtype(), rhs.dtype())?;
        let shape = crate::shape::matmul_shape(lhs.shape(), rhs.shape())?;
        Ok(Expr::node(
            ExprKind::MatMul(lhs.clone(), rhs.clone()),
            dtype,
            shape,
            context_id,
        ))
    }

    pub fn linalg(op: crate::linalg::LinalgOp, a: &Expr, b: Option<&Expr>) -> Result<Expr> {
        let context = join_context(a.context_id(), b.and_then(Expr::context_id))?;
        let (shape, dtype) =
            crate::linalg::infer(op, a.shape(), a.dtype(), b.map(|v| (v.shape(), v.dtype())))?;
        Ok(Expr::node(
            ExprKind::Linalg(op, a.clone(), b.cloned()),
            dtype,
            shape,
            context,
        ))
    }

    pub fn fft(options: crate::fft::FftOptions, input: &Expr) -> Result<Expr> {
        let (shape, dtype, _, _) = options.infer(input.shape(), input.dtype())?;
        Ok(Expr::node(
            ExprKind::Fft(options, input.clone()),
            dtype,
            shape,
            input.context_id(),
        ))
    }

    pub fn polynomial(
        options: crate::polynomial::PolynomialOptions,
        coefficients: &Expr,
        points: &Expr,
    ) -> Result<Expr> {
        let context = join_context(coefficients.context_id(), points.context_id())?;
        let (shape, dtype) = options.infer(
            coefficients.shape(),
            coefficients.dtype(),
            points.shape(),
            points.dtype(),
        )?;
        Ok(Expr::node(
            ExprKind::Polynomial(options, coefficients.clone(), points.clone()),
            dtype,
            shape,
            context,
        ))
    }

    pub fn polynomial_derivative(
        options: crate::polynomial::DerivativeOptions,
        coefficients: &Expr,
    ) -> Result<Expr> {
        let (shape, dtype, _) = options.infer(coefficients.shape(), coefficients.dtype())?;
        // The legacy zero-order derivative aliases its input, whereas the
        // ascending API always copies (and converts integer coefficients).
        if options.order == crate::polynomial::PolynomialOrder::Descending && options.count == 0 {
            return Ok(coefficients.clone());
        }
        Ok(Expr::node(
            ExprKind::PolynomialDerivative(options, coefficients.clone()),
            dtype,
            shape,
            coefficients.context_id(),
        ))
    }

    pub fn index(input: &Expr, selectors: Vec<Selector>) -> Result<Expr> {
        let shape = index_shape(input.shape(), &selectors)?;
        Ok(Expr::node(
            ExprKind::Index(input.clone(), selectors),
            input.dtype(),
            shape,
            input.context_id(),
        ))
    }

    pub fn transpose(input: &Expr, axes: Option<&[i64]>) -> Result<Expr> {
        let axes = crate::layout::permutation(input.shape().len(), axes)?;
        let shape = axes.iter().map(|&axis| input.shape()[axis]).collect();
        Ok(Expr::node(
            ExprKind::Transpose(input.clone(), axes),
            input.dtype(),
            shape,
            input.context_id(),
        ))
    }

    pub fn reshape(input: &Expr, shape: &[i64], copy: crate::layout::CopyPolicy) -> Result<Expr> {
        let size = crate::numerical::size(input.shape())?;
        let shape = crate::layout::reshape_shape(size, shape)?;
        if copy == crate::layout::CopyPolicy::Never
            && crate::layout::reshape_strides(input.shape(), &input.layout_strides()?, &shape)
                .is_none()
        {
            return Err(Error::new(
                ErrorCode::ShapeMismatch,
                "The requested C-order reshape requires a copy.",
            ));
        }
        Ok(Expr::node(
            ExprKind::Reshape(input.clone(), copy),
            input.dtype(),
            shape,
            input.context_id(),
        ))
    }

    /// NumPy's C-order `ravel`: a view when the input layout is C-contiguous
    /// and an independent copy otherwise, so the flat result never aliases
    /// stepped, reversed or transposed storage.
    pub fn ravel(input: &Expr) -> Result<Expr> {
        let expected = crate::layout::checked_strides(input.shape())?;
        let contiguous = input
            .shape()
            .iter()
            .zip(input.layout_strides()?)
            .zip(expected)
            .all(|((&dim, stride), expected)| dim <= 1 || stride == expected);
        let copy = if contiguous {
            crate::layout::CopyPolicy::IfNeeded
        } else {
            crate::layout::CopyPolicy::Always
        };
        Expr::reshape(input, &[-1], copy)
    }

    // Infer layout without reading data. Arithmetic produces C-order storage;
    // pure shape/index transforms retain the layout of their input. This lets
    // copy:false reject incompatible views even over a computed expression.
    fn layout_strides(&self) -> Result<Vec<isize>> {
        match self.kind() {
            ExprKind::Input(a) => Ok(a.strides().to_vec()),
            ExprKind::Transpose(e, axes) => {
                let strides = e.layout_strides()?;
                Ok(axes.iter().map(|&axis| strides[axis]).collect())
            }
            ExprKind::Reshape(e, copy) if *copy != crate::layout::CopyPolicy::Always => Ok(
                crate::layout::reshape_strides(e.shape(), &e.layout_strides()?, self.shape())
                    .unwrap_or(crate::layout::checked_strides(self.shape())?),
            ),
            ExprKind::Index(e, selectors) if selectors.iter().all(Selector::is_basic) => {
                let strides = e.layout_strides()?;
                let mut out = Vec::new();
                for (axis, (&len, &stride)) in e.shape().iter().zip(&strides).enumerate() {
                    match selectors.get(axis) {
                        Some(Selector::Index(_)) => {}
                        Some(Selector::Slice { start, stop, step }) => {
                            let (_, step, n) =
                                crate::shape::resolve_slice(*start, *stop, *step, len)?;
                            out.push(if n <= 1 {
                                0
                            } else {
                                stride.checked_mul(step).ok_or_else(|| {
                                    Error::new(
                                        ErrorCode::ShapeMismatch,
                                        "The slice strides are too large.",
                                    )
                                })?
                            });
                        }
                        _ => out.push(stride),
                    }
                }
                Ok(out)
            }
            _ => crate::layout::checked_strides(self.shape()),
        }
    }

    pub fn cast(input: &Expr, dtype: DType) -> Result<Expr> {
        dtype.ensure_computable()?;
        Ok(Expr::node(
            ExprKind::Cast(input.clone(), dtype),
            dtype,
            input.shape().to_vec(),
            input.context_id(),
        ))
    }

    /// Short operation kind for measurements: the root node's operator ID.
    pub fn operation_id(&self) -> &'static str {
        match &self.0.kind {
            ExprKind::Input(_) => "input",
            ExprKind::Scalar(_) => "scalar",
            ExprKind::Unary(op, _) => op.id(),
            ExprKind::Binary { op, .. } => op.id(),
            ExprKind::Where(..) => "where",
            ExprKind::Math(op, _) => op.id(),
            ExprKind::Reduce(op, _, _) => op.id(),
            ExprKind::Scan(op, _, _) => op.id(),
            ExprKind::Variance(op, ..) => op.id(),
            ExprKind::Shift(op, ..) => op.id(),
            ExprKind::Concat(..) => "concatenate",
            ExprKind::MatMul(..) => "matmul",
            ExprKind::Linalg(op, ..) => op.id(),
            ExprKind::Fft(options, ..) => options.op.id(),
            ExprKind::Polynomial(options, ..) => options.order.id(),
            ExprKind::PolynomialDerivative(options, ..) => options.id(),
            ExprKind::Index(..) => "index",
            ExprKind::Cast(..) => "cast",
            ExprKind::Transpose(..) => "transpose",
            ExprKind::Reshape(..) => "reshape",
            ExprKind::Random(..) => "random",
        }
    }

    fn children(&self) -> Vec<&Expr> {
        match &self.0.kind {
            ExprKind::Input(_) | ExprKind::Scalar(_) | ExprKind::Random(..) => vec![],
            ExprKind::Linalg(_, a, b) => std::iter::once(a).chain(b.as_ref()).collect(),
            ExprKind::Fft(_, e) | ExprKind::PolynomialDerivative(_, e) => vec![e],
            ExprKind::Unary(_, e)
            | ExprKind::Math(_, e)
            | ExprKind::Reduce(_, e, _)
            | ExprKind::Scan(_, e, _)
            | ExprKind::Variance(_, e, _, _)
            | ExprKind::Shift(_, e, _)
            | ExprKind::Index(e, _)
            | ExprKind::Cast(e, _)
            | ExprKind::Transpose(e, _)
            | ExprKind::Reshape(e, _) => vec![e],
            ExprKind::Where(c, x, y) => vec![c, x, y],
            ExprKind::Concat(parts, _) => parts.iter().collect(),
            ExprKind::Binary { lhs, rhs, .. }
            | ExprKind::MatMul(lhs, rhs)
            | ExprKind::Polynomial(_, lhs, rhs) => vec![lhs, rhs],
        }
    }

    fn visit(&self, f: &mut impl FnMut(&Expr)) {
        fn walk(expr: &Expr, seen: &mut HashSet<*const Node>, f: &mut impl FnMut(&Expr)) {
            if !seen.insert(Arc::as_ptr(&expr.0)) {
                return;
            }
            f(expr);
            for child in expr.children() {
                walk(child, seen, f);
            }
        }
        walk(self, &mut HashSet::new(), f);
    }

    pub fn inputs(&self) -> Vec<NdArray> {
        let mut out: Vec<NdArray> = Vec::new();
        self.visit(&mut |e| {
            if let ExprKind::Input(a) = &e.0.kind
                && !out.iter().any(|b| same_view(a, b))
            {
                out.push(a.clone());
            }
        });
        out
    }

    pub fn generators(&self) -> Vec<Generator> {
        let mut out: Vec<Generator> = Vec::new();
        self.visit(&mut |e| {
            if let ExprKind::Random(g, _, _) = &e.0.kind
                && !out.iter().any(|h| h.ptr_eq(g))
            {
                out.push(g.clone());
            }
        });
        out
    }

    /// Checks the GPU subset statically. Errors here reject `compute()` before
    /// admission.
    pub fn check_gpu_support(&self) -> Result<()> {
        crate::gpu::validate(self)
    }
}

/// NEP 50 promotion of an array dtype with a weak scalar. An integer scalar
/// that the resulting integer dtype cannot represent fails instead of wrapping,
/// as NumPy raises `OverflowError` for out-of-bounds Python integers.
fn weak_scalar_dtype(array: DType, scalar: Scalar) -> Result<DType> {
    let dtype = DType::promote_weak(array, scalar.kind())?;
    let value: i128 = match scalar {
        Scalar::Int(v) => v.into(),
        Scalar::UInt(v) => v.into(),
        Scalar::Bool(_) | Scalar::Float(_) | Scalar::Complex(_) => return Ok(dtype),
        _ => return Err(crate::dtype::unsupported()),
    };
    let (min, max): (i128, i128) = match dtype {
        DType::Int8 => (i8::MIN.into(), i8::MAX.into()),
        DType::Int16 => (i16::MIN.into(), i16::MAX.into()),
        DType::Int32 => (i32::MIN.into(), i32::MAX.into()),
        DType::Int64 => (i64::MIN.into(), i64::MAX.into()),
        DType::UInt8 => (0, u8::MAX.into()),
        DType::UInt16 => (0, u16::MAX.into()),
        DType::UInt32 => (0, u32::MAX.into()),
        DType::UInt64 => (0, u64::MAX.into()),
        _ => return Ok(dtype),
    };
    if value < min || value > max {
        return Err(Error::new(
            ErrorCode::DTypeUnsupported,
            format!(
                "The integer scalar {value} is out of bounds for dtype '{}'. Cast the array to a \
                 wider dtype first.",
                dtype.id()
            ),
        ));
    }
    Ok(dtype)
}

fn same_view(a: &NdArray, b: &NdArray) -> bool {
    a.shares_storage(b) && a.shape() == b.shape() && std::ptr::eq(a, b)
}

/// The value produced by a job.
#[derive(Debug)]
pub enum Value {
    Dense(DenseArray),
    /// A basic-indexing view of a live input, as NumPy returns.
    View(NdArray),
}

/// Per-job state: the frozen inputs, budget, cancellation and pending generator
/// states.
pub struct JobEnv {
    pub generator_turns: Vec<crate::sync::FifoGuard>,
    pub device: Device,
    pub budget: MemoryBudget,
    pub gpu_budget: Option<MemoryBudget>,
    pub token: CancellationToken,
    snapshots: HashMap<*const NdArray, DenseArray>,
    rng: Vec<(Generator, Pcg64)>,
    /// Incoming edge counts of the expression being evaluated.
    uses: HashMap<*const Node, usize>,
    /// Results of nodes with several incoming edges and their unread uses.
    shared: HashMap<*const Node, (DenseArray, usize)>,
}

// SAFETY: the raw pointer keys are only compared for identity, never
// dereferenced.
unsafe impl Send for JobEnv {}

impl JobEnv {
    pub fn new(device: Device, budget: MemoryBudget, token: CancellationToken) -> Self {
        Self {
            generator_turns: Vec::new(),
            device,
            budget,
            gpu_budget: None,
            token,
            snapshots: HashMap::new(),
            rng: Vec::new(),
            uses: HashMap::new(),
            shared: HashMap::new(),
        }
    }

    /// Freezes every input and reads generator state. Call once when the job
    /// starts.
    pub fn freeze(&mut self, expr: &Expr) -> Result<()> {
        let mut result = Ok(());
        expr.visit(&mut |e| {
            if result.is_err() {
                return;
            }
            if let ExprKind::Input(a) = &e.0.kind {
                let key = a as *const NdArray;
                if !self.snapshots.contains_key(&key) {
                    match a.snapshot_cancellable(&self.budget, &self.token) {
                        Ok(s) => {
                            self.snapshots.insert(key, s);
                            #[cfg(test)]
                            if let Some(hook) = freeze_test_hook::take(a.storage_id()) {
                                hook();
                            }
                        }
                        Err(err) => result = Err(err),
                    }
                }
            }
        });
        result?;
        for g in expr.generators() {
            let state = g.current();
            self.rng.push((g, state));
        }
        Ok(())
    }

    /// Generator states to commit after success.
    pub fn take_generator_states(&mut self) -> Vec<(Generator, Pcg64)> {
        std::mem::take(&mut self.rng)
    }

    fn snapshot(&self, a: &NdArray) -> Result<&DenseArray> {
        self.snapshots.get(&(a as *const NdArray)).ok_or_else(|| {
            Error::new(
                ErrorCode::IndexInvalid,
                "An input was not frozen at job start.",
            )
        })
    }

    fn copy_dense(&self, d: &DenseArray) -> Result<DenseArray> {
        let reservation = self.budget.reserve(d.buffer.byte_len())?;
        DenseArray::new(d.shape.clone(), d.buffer.clone(), reservation)
    }

    /// Evaluates `expr` on the explicitly requested device.
    pub fn eval(&mut self, expr: &Expr) -> Result<Value> {
        if self.device == Device::Gpu {
            let gpu = self.gpu_budget.as_ref().ok_or_else(|| {
                Error::new(
                    ErrorCode::ContextInvalidOptions,
                    "The GPU budget is missing.",
                )
            })?;
            return crate::gpu::execute(expr, &self.budget, gpu, &self.token, |array| {
                self.snapshot(array)
            })
            .map(Value::Dense);
        }
        if !matches!(expr.kind(), ExprKind::Input(_))
            && let Some(view) = Self::live_view(expr)
        {
            return Ok(Value::View(view));
        }
        self.begin(expr);
        let result = if matches!(
            expr.kind(),
            ExprKind::Transpose(..) | ExprKind::Reshape(..) | ExprKind::Index(..)
        ) {
            self.owned_layout(expr).map(Value::View)
        } else {
            self.eval_node(expr).map(Value::Dense)
        };
        self.end();
        result
    }

    /// Counts the incoming edges of every node under `root`. A node reached
    /// through several edges is then evaluated once per evaluation, so a
    /// random node draws once and repeated subexpressions do no extra work.
    fn begin(&mut self, root: &Expr) {
        self.shared.clear();
        let mut uses = HashMap::new();
        root.visit(&mut |e| {
            for child in e.children() {
                *uses.entry(Arc::as_ptr(&child.0)).or_insert(0usize) += 1;
            }
        });
        self.uses = uses;
    }

    fn end(&mut self) {
        self.uses.clear();
        self.shared.clear();
    }

    fn is_shared(&self, expr: &Expr) -> bool {
        self.uses
            .get(&Arc::as_ptr(&expr.0))
            .is_some_and(|&uses| uses > 1)
    }

    /// Resolve metadata-only chains without reading live data. Returned views
    /// deliberately retain live storage; calculations below use frozen inputs.
    fn live_view(expr: &Expr) -> Option<NdArray> {
        match expr.kind() {
            ExprKind::Input(a) => Some(a.clone()),
            ExprKind::Index(e, selectors) if selectors.iter().all(Selector::is_basic) => {
                Self::live_view(e)?.view(selectors).ok()
            }
            ExprKind::Transpose(e, axes) => Some(Self::live_view(e)?.transpose(axes)),
            ExprKind::Reshape(e, copy) if *copy != crate::layout::CopyPolicy::Always => {
                Self::live_view(e)?.reshape_view(expr.shape())
            }
            _ => None,
        }
    }

    /// Build an owned result from snapshots, retaining strided layouts through
    /// shape transforms over computed inputs. No live storage is read here.
    fn owned_layout(&mut self, expr: &Expr) -> Result<NdArray> {
        if self.is_shared(expr) {
            return NdArray::from_dense(expr.context_id().unwrap_or(0), self.eval_node(expr)?);
        }
        self.owned_layout_unshared(expr)
    }

    fn owned_layout_unshared(&mut self, expr: &Expr) -> Result<NdArray> {
        self.token.check()?;
        match expr.kind() {
            ExprKind::Transpose(e, axes) => Ok(self.owned_layout(e)?.transpose(axes)),
            ExprKind::Reshape(e, copy) => {
                let input = self.owned_layout(e)?;
                let contiguous = input
                    .shape()
                    .iter()
                    .zip(input.strides())
                    .zip(crate::layout::checked_strides(input.shape())?)
                    .all(|((&dim, &stride), expected)| dim <= 1 || stride == expected);
                if (*copy != crate::layout::CopyPolicy::Always || contiguous)
                    && let Some(view) = input.reshape_view(expr.shape())
                {
                    return Ok(view);
                }
                let mut dense = input.snapshot_cancellable(&self.budget, &self.token)?;
                self.token.check()?;
                dense.shape = expr.shape().to_vec();
                NdArray::from_dense(expr.context_id().unwrap_or(0), dense)
            }
            ExprKind::Index(e, selectors) if selectors.iter().all(Selector::is_basic) => {
                self.owned_layout(e)?.view(selectors)
            }
            _ => NdArray::from_dense(expr.context_id().unwrap_or(0), self.eval_node(expr)?),
        }
    }

    /// Evaluates `expr` on the CPU into a new dense array, materializing
    /// views.
    pub(crate) fn eval_dense(&mut self, expr: &Expr) -> Result<DenseArray> {
        self.begin(expr);
        let result = self.eval_node(expr);
        self.end();
        result
    }

    /// Evaluates a node of the current expression, reusing the result of a
    /// shared node. The last reader takes the cached array; earlier readers
    /// receive budgeted copies.
    fn eval_node(&mut self, expr: &Expr) -> Result<DenseArray> {
        if !self.is_shared(expr) {
            return self.compute_dense(expr);
        }
        let key = Arc::as_ptr(&expr.0);
        if !self.shared.contains_key(&key) {
            let value = self.compute_dense(expr)?;
            self.shared.insert(key, (value, self.uses[&key]));
        }
        let unread = &mut self.shared.get_mut(&key).expect("cached above").1;
        *unread -= 1;
        if *unread == 0 {
            return Ok(self.shared.remove(&key).expect("cached above").0);
        }
        self.copy_dense(&self.shared[&key].0)
    }

    fn compute_dense(&mut self, expr: &Expr) -> Result<DenseArray> {
        self.token.check()?;
        let budget = self.budget.clone();
        let token = self.token.clone();
        match &expr.0.kind {
            ExprKind::Input(a) => {
                let snap = self.snapshot(a)?;
                self.copy_dense(snap)
            }
            ExprKind::Scalar(s) => {
                let dtype = s.default_dtype();
                let reservation = budget.reserve(dtype.item_size() as u64)?;
                DenseArray::new(vec![], Buffer::filled(dtype, 1, s.clone())?, reservation)
            }
            ExprKind::Unary(op, e) => {
                let v = self.eval_node(e)?;
                kernels::unary(*op, &v, &budget, &token)
            }
            ExprKind::Binary {
                op,
                lhs,
                rhs,
                compute_dtype,
            } => {
                let a = self.eval_node(lhs)?;
                let b = self.eval_node(rhs)?;
                kernels::binary(*op, &a, &b, *compute_dtype, &budget, &token)
            }
            ExprKind::Where(c, x, y) => {
                let c = self.eval_node(c)?;
                let x = self.eval_node(x)?;
                let y = self.eval_node(y)?;
                kernels::select(&c, &x, &y, expr.dtype(), &budget, &token)
            }
            ExprKind::Math(op, e) => {
                let v = self.eval_node(e)?;
                crate::ufunc::execute(*op, &v, &budget, &token)
            }
            ExprKind::Reduce(op, e, axes) => {
                let v = self.eval_node(e)?;
                kernels::reduce(*op, &v, axes.as_deref(), &budget, &token)
            }
            ExprKind::Scan(op, e, axis) => {
                let v = self.eval_node(e)?;
                kernels::scan(*op, &v, *axis, &budget, &token)
            }
            ExprKind::Variance(op, e, axis, ddof) => {
                let v = self.eval_node(e)?;
                kernels::variance(*op, &v, *axis, *ddof, &budget, &token)
            }
            ExprKind::Shift(op, e, axes) => {
                let v = self.eval_node(e)?;
                kernels::shift(*op, &v, axes, &budget, &token)
            }
            ExprKind::Concat(parts, axis) => {
                let dtype = expr.dtype();
                let mut values = Vec::with_capacity(parts.len());
                for part in parts {
                    let value = self.eval_node(part)?;
                    values.push(if value.dtype() == dtype {
                        value
                    } else {
                        let reservation =
                            budget.reserve(checked_byte_len(value.size(), dtype)? as u64)?;
                        DenseArray::new(
                            value.shape.clone(),
                            value.buffer.cast_cancellable(dtype, &token)?,
                            reservation,
                        )?
                    });
                }
                kernels::concat(&values, *axis, expr.shape(), dtype, &budget, &token)
            }
            ExprKind::MatMul(l, r) => {
                let a = self.eval_node(l)?;
                let b = self.eval_node(r)?;
                kernels::matmul(&a, &b, expr.dtype(), &budget, &token)
            }
            ExprKind::Linalg(op, a, b) => {
                let a = self.eval_node(a)?;
                let b = b.as_ref().map(|v| self.eval_node(v)).transpose()?;
                crate::linalg::execute(*op, &a, b.as_ref(), &budget, &token)
            }
            ExprKind::Polynomial(options, coefficients, points) => {
                let coefficients = self.eval_node(coefficients)?;
                let points = self.eval_node(points)?;
                crate::polynomial::execute(*options, &coefficients, &points, &budget, &token)
            }
            ExprKind::PolynomialDerivative(options, coefficients) => {
                let coefficients = self.eval_node(coefficients)?;
                crate::polynomial::derivative(*options, &coefficients, &budget, &token)
            }
            ExprKind::Fft(options, input) => {
                let input = self.eval_node(input)?;
                crate::fft::execute(*options, &input, &budget, &token)
            }
            ExprKind::Index(e, selectors) => {
                let v = self.eval_node(e)?;
                let tmp = NdArray::from_dense(
                    0,
                    DenseArray::new(v.shape.clone(), v.buffer.clone(), Reservation::none())?,
                )?;
                tmp.take(selectors, &budget)
            }
            ExprKind::Transpose(..) | ExprKind::Reshape(..) => {
                let array = self.owned_layout_unshared(expr)?;
                let dense = array.snapshot_cancellable(&budget, &token)?;
                token.check()?;
                Ok(dense)
            }
            ExprKind::Cast(e, dtype) => {
                let v = self.eval_node(e)?;
                let reservation = budget.reserve(checked_byte_len(v.size(), *dtype)? as u64)?;
                DenseArray::new(
                    v.shape.clone(),
                    v.buffer.cast_cancellable(*dtype, &token)?,
                    reservation,
                )
            }
            ExprKind::Random(g, shape, operation) => {
                let slot = self
                    .rng
                    .iter_mut()
                    .find(|(h, _)| h.ptr_eq(g))
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::IndexInvalid,
                            "A generator was not frozen at job start.",
                        )
                    })?;
                operation.generate(&mut slot.1, shape, &budget, &token)
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod freeze_test_hook {
    use std::{
        collections::HashMap,
        sync::{Mutex, OnceLock},
    };

    type Hook = Box<dyn FnOnce() + Send>;
    static HOOKS: OnceLock<Mutex<HashMap<u64, Hook>>> = OnceLock::new();

    fn hooks() -> &'static Mutex<HashMap<u64, Hook>> {
        HOOKS.get_or_init(|| Mutex::new(HashMap::new()))
    }

    pub fn install(storage_id: u64, hook: Hook) {
        assert!(hooks().lock().unwrap().insert(storage_id, hook).is_none());
    }

    pub fn take(storage_id: u64) -> Option<Hook> {
        hooks().lock().unwrap().remove(&storage_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(ctx: u64, shape: Vec<usize>, buffer: Buffer) -> NdArray {
        NdArray::from_dense(
            ctx,
            DenseArray::new(shape, buffer, Reservation::none()).unwrap(),
        )
        .unwrap()
    }

    fn eval(e: &Expr) -> DenseArray {
        let mut env = JobEnv::new(
            Device::Cpu,
            MemoryBudget::new(Device::Cpu, u64::MAX),
            CancellationToken::new(),
        );
        env.freeze(e).unwrap();
        match env.eval(e).unwrap() {
            Value::Dense(d) => d,
            Value::View(v) => {
                DenseArray::new(v.shape().to_vec(), v.to_buffer(), Reservation::none()).unwrap()
            }
        }
    }

    #[test]
    fn shared_metadata_walk_visits_each_node_once() {
        let mut expr = Expr::scalar(Scalar::Int(1));
        for _ in 0..30 {
            expr = Expr::binary(BinaryOp::Add, &expr, &expr).unwrap();
        }
        let mut visits = 0;
        expr.visit(&mut |_| visits += 1);
        assert_eq!(visits, 31);
        assert!(expr.inputs().is_empty());
        assert!(expr.generators().is_empty());

        let mut env = JobEnv::new(
            Device::Cpu,
            MemoryBudget::new(Device::Cpu, 1024),
            CancellationToken::new(),
        );
        env.freeze(&expr).unwrap();
        assert!(env.snapshots.is_empty());
    }

    #[test]
    fn shared_nodes_evaluate_once() {
        let mut expr = Expr::scalar(Scalar::Int(1));
        for _ in 0..40 {
            expr = Expr::binary(BinaryOp::Add, &expr, &expr).unwrap();
        }
        assert_eq!(eval(&expr).buffer, Buffer::I64(vec![1 << 40]));

        // Both slices of one random node read the same draw.
        let generator = Generator::new(7);
        let random = Expr::random(1, &generator, vec![5]);
        let slice = |start, stop| Selector::Slice {
            start,
            stop,
            step: None,
        };
        let later = Expr::index(&random, vec![slice(Some(1), None)]).unwrap();
        let earlier = Expr::index(&random, vec![slice(None, Some(-1))]).unwrap();
        let diff = Expr::binary(BinaryOp::Sub, &later, &earlier).unwrap();
        let Buffer::F64(draw) = eval(&random).buffer else {
            panic!("random returns float64");
        };
        let expected: Vec<f64> = draw.windows(2).map(|w| w[1] - w[0]).collect();
        assert_eq!(eval(&diff).buffer, Buffer::F64(expected));
        let same = Expr::binary(BinaryOp::Sub, &random, &random).unwrap();
        assert_eq!(eval(&same).buffer, Buffer::F64(vec![0.0; 5]));
        let moved = Expr::transpose(&random, Some(&[0])).unwrap();
        let both = Expr::binary(BinaryOp::Sub, &moved, &random).unwrap();
        assert_eq!(eval(&both).buffer, Buffer::F64(vec![0.0; 5]));
    }

    #[test]
    fn concat_promotes_and_interleaves_blocks() {
        let a = Expr::input(&arr(1, vec![2, 2], Buffer::I32(vec![1, 2, 3, 4])));
        let b = Expr::input(&arr(1, vec![2, 1], Buffer::F32(vec![0.5, 1.5])));
        let joined = Expr::concat(&[a.clone(), b.clone()], -1).unwrap();
        assert_eq!(joined.shape(), &[2, 3]);
        assert_eq!(joined.dtype(), DType::Float64);
        assert_eq!(
            eval(&joined).buffer,
            Buffer::F64(vec![1.0, 2.0, 0.5, 3.0, 4.0, 1.5])
        );
        let rows = Expr::concat(&[a.clone(), a.clone()], 0).unwrap();
        assert_eq!(rows.shape(), &[4, 2]);
        assert_eq!(
            eval(&rows).buffer,
            Buffer::I32(vec![1, 2, 3, 4, 1, 2, 3, 4])
        );
        let empty = Expr::input(&arr(1, vec![0, 2], Buffer::I32(vec![])));
        let with_empty = Expr::concat(&[empty, a.clone()], 0).unwrap();
        assert_eq!(eval(&with_empty).buffer, Buffer::I32(vec![1, 2, 3, 4]));

        let text = |width, values: &[&str]| {
            let mut storage = vec![0u32; width * values.len()];
            for (row, value) in storage.chunks_exact_mut(width).zip(values) {
                for (slot, c) in row.iter_mut().zip(value.chars()) {
                    *slot = c as u32;
                }
            }
            Expr::input(&arr(
                1,
                vec![values.len()],
                Buffer::Unicode {
                    width,
                    values: storage,
                },
            ))
        };
        let words = Expr::concat(&[text(1, &["a"]), text(3, &["bcd", "e"])], 0).unwrap();
        assert_eq!(words.dtype(), DType::Str(3));
        let joined = eval(&words);
        assert_eq!(joined.buffer.get(0), Scalar::Text("a".into()));
        assert_eq!(joined.buffer.get(1), Scalar::Text("bcd".into()));
        assert_eq!(joined.buffer.get(2), Scalar::Text("e".into()));
    }

    #[test]
    fn concat_rejects_invalid_parts() {
        let a = Expr::input(&arr(1, vec![2, 2], Buffer::I32(vec![1, 2, 3, 4])));
        let code = |result: Result<Expr>| result.unwrap_err().code;
        assert_eq!(code(Expr::concat(&[], 0)), ErrorCode::ShapeMismatch);
        let scalar = Expr::input(&arr(1, vec![], Buffer::I32(vec![1])));
        assert_eq!(code(Expr::concat(&[scalar], 0)), ErrorCode::ShapeMismatch);
        assert_eq!(
            code(Expr::concat(&[a.clone()], 2)),
            ErrorCode::IndexOutOfBounds
        );
        assert_eq!(
            code(Expr::concat(&[a.clone()], -3)),
            ErrorCode::IndexOutOfBounds
        );
        let vector = Expr::input(&arr(1, vec![2], Buffer::I32(vec![1, 2])));
        assert_eq!(
            code(Expr::concat(&[a.clone(), vector], 0)),
            ErrorCode::ShapeMismatch
        );
        let wide = Expr::input(&arr(1, vec![1, 3], Buffer::I32(vec![1, 2, 3])));
        assert_eq!(
            code(Expr::concat(&[a.clone(), wide], 0)),
            ErrorCode::ShapeMismatch
        );
        let other = Expr::input(&arr(2, vec![2, 2], Buffer::I32(vec![1, 2, 3, 4])));
        assert_eq!(
            code(Expr::concat(&[a.clone(), other], 0)),
            ErrorCode::ContextMismatch
        );
        assert!(Expr::concat(&[a], 0).unwrap().check_gpu_support().is_err());
        // An empty result needs no allocation, but its shape must still fit
        // the 32-bit public transport.
        let half = Expr::input(&arr(1, vec![0, 1 << 31], Buffer::I8(vec![])));
        assert_eq!(
            code(Expr::concat(&[half.clone(), half.clone()], 1)),
            ErrorCode::ShapeMismatch
        );
        let fits = Expr::input(&arr(1, vec![0, (1 << 31) - 1], Buffer::I8(vec![])));
        assert_eq!(
            Expr::concat(&[half, fits], 1).unwrap().shape(),
            &[0, u32::MAX as usize]
        );
    }

    #[test]
    fn shared_discovery_keeps_first_encounter_order() {
        let first = arr(1, vec![1], Buffer::F64(vec![1.0]));
        let second = arr(1, vec![1], Buffer::F64(vec![2.0]));
        let input_first = Expr::input(&first);
        let input_second = Expr::input(&second);
        let generator_first = Generator::new(1);
        let generator_second = Generator::new(2);
        let random_first = Expr::random(1, &generator_first, vec![1]);
        let random_second = Expr::random(1, &generator_second, vec![1]);
        let left = Expr::binary(BinaryOp::Add, &input_first, &random_first).unwrap();
        let right = Expr::binary(BinaryOp::Add, &input_second, &random_second).unwrap();
        let shared = Expr::binary(BinaryOp::Add, &left, &right).unwrap();
        let expr = Expr::binary(BinaryOp::Add, &shared, &shared).unwrap();

        let inputs = expr.inputs();
        assert_eq!(inputs.len(), 2);
        assert!(inputs[0].shares_storage(&first));
        assert!(inputs[1].shares_storage(&second));
        let generators = expr.generators();
        assert_eq!(generators.len(), 2);
        assert!(generators[0].ptr_eq(&generator_first));
        assert!(generators[1].ptr_eq(&generator_second));

        let mut env = JobEnv::new(
            Device::Cpu,
            MemoryBudget::new(Device::Cpu, 1024),
            CancellationToken::new(),
        );
        env.freeze(&expr).unwrap();
        assert_eq!(env.snapshots.len(), 2);
        assert_eq!(env.rng.len(), 2);
        assert!(env.rng[0].0.ptr_eq(&generator_first));
        assert!(env.rng[1].0.ptr_eq(&generator_second));
    }

    #[test]
    fn shared_input_freeze_keeps_the_execution_snapshot() {
        let array = arr(1, vec![1], Buffer::F64(vec![3.0]));
        let input = Expr::input(&array);
        let expr = Expr::binary(BinaryOp::Add, &input, &input).unwrap();
        let budget = MemoryBudget::new(Device::Cpu, 1024);
        let token = CancellationToken::new();
        let mut frozen = JobEnv::new(Device::Cpu, budget.clone(), token.clone());
        frozen.freeze(&expr).unwrap();

        let turn = array.write_ticket().wait(&token).unwrap();
        let replacement =
            DenseArray::new(vec![1], Buffer::F64(vec![5.0]), Reservation::none()).unwrap();
        array.assign(&turn, &[], &replacement, &token).unwrap();
        drop(turn);

        assert_eq!(
            match frozen.eval(&expr).unwrap() {
                Value::Dense(value) => value.buffer,
                Value::View(_) => panic!("binary expression must produce a dense value"),
            },
            Buffer::F64(vec![6.0])
        );
        let mut later = JobEnv::new(Device::Cpu, budget, token);
        later.freeze(&expr).unwrap();
        assert_eq!(
            match later.eval(&expr).unwrap() {
                Value::Dense(value) => value.buffer,
                Value::View(_) => panic!("binary expression must produce a dense value"),
            },
            Buffer::F64(vec![10.0])
        );
    }

    #[test]
    fn unicode_cast_rejects_byte_overflow_before_budget_admission() {
        let input = Expr::input(&arr(
            1,
            vec![1024],
            Buffer::Unicode {
                width: 1,
                values: vec![0; 1024],
            },
        ));
        let cast = Expr::cast(&input, DType::Str(usize::MAX / 4096 + 1)).unwrap();
        let mut env = JobEnv::new(
            Device::Cpu,
            MemoryBudget::new(Device::Cpu, u64::MAX),
            CancellationToken::new(),
        );
        env.freeze(&cast).unwrap();
        assert_eq!(
            env.eval_dense(&cast).unwrap_err().code,
            ErrorCode::MemoryBudgetExceeded
        );
    }

    #[test]
    fn infers_dtype_and_shape_at_construction() {
        let a = Expr::input(&arr(1, vec![2, 1], Buffer::I32(vec![1, 2])));
        let b = Expr::input(&arr(1, vec![3], Buffer::F32(vec![0.5; 3])));
        let sum = Expr::binary(BinaryOp::Add, &a, &b).unwrap();
        assert_eq!(sum.dtype(), DType::Float64);
        assert_eq!(sum.shape(), &[2, 3]);
        let c = Expr::input(&arr(1, vec![4], Buffer::F32(vec![0.0; 4])));
        assert_eq!(
            Expr::binary(BinaryOp::Add, &b, &c).unwrap_err().code,
            ErrorCode::ShapeBroadcast
        );
    }

    #[test]
    fn weak_scalars_follow_nep50() {
        let a = Expr::input(&arr(1, vec![2], Buffer::F32(vec![1.0, 2.0])));
        let e = Expr::binary(BinaryOp::Mul, &a, &Expr::scalar(Scalar::Float(2.0))).unwrap();
        assert_eq!(e.dtype(), DType::Float32);
        assert_eq!(eval(&e).buffer, Buffer::F32(vec![2.0, 4.0]));
    }

    #[test]
    fn weak_integer_scalars_must_fit_the_array_dtype() {
        // NumPy 2.5.4: np.array([1], np.uint32) + (-1) and
        // np.array([1], np.int32) + 2**32 raise OverflowError.
        let u = Expr::input(&arr(1, vec![1], Buffer::U32(vec![1])));
        let i = Expr::input(&arr(1, vec![1], Buffer::I32(vec![1])));
        let add = |a: &Expr, s: Scalar| Expr::binary(BinaryOp::Add, a, &Expr::scalar(s));
        assert_eq!(
            add(&u, Scalar::Int(-1)).unwrap_err().code,
            ErrorCode::DTypeUnsupported
        );
        assert_eq!(
            Expr::binary(BinaryOp::Add, &Expr::scalar(Scalar::Int(1 << 32)), &i)
                .unwrap_err()
                .code,
            ErrorCode::DTypeUnsupported
        );
        let b = Expr::input(&arr(1, vec![1], Buffer::Bool(vec![true])));
        assert!(add(&b, Scalar::UInt(u64::MAX)).is_err());
        // In-range scalars keep the array dtype; float arrays accept any int.
        assert_eq!(
            add(&u, Scalar::Int(u32::MAX as i64)).unwrap().dtype(),
            DType::UInt32
        );
        assert_eq!(
            add(&i, Scalar::Int(i32::MIN as i64)).unwrap().dtype(),
            DType::Int32
        );
        let f = Expr::input(&arr(1, vec![1], Buffer::F32(vec![1.0])));
        assert_eq!(
            add(&f, Scalar::UInt(u64::MAX)).unwrap().dtype(),
            DType::Float32
        );
    }

    #[test]
    fn rejects_cross_context_combination() {
        let a = Expr::input(&arr(1, vec![1], Buffer::F64(vec![1.0])));
        let b = Expr::input(&arr(2, vec![1], Buffer::F64(vec![1.0])));
        assert_eq!(
            Expr::binary(BinaryOp::Add, &a, &b).unwrap_err().code,
            ErrorCode::ContextMismatch
        );
    }

    #[test]
    fn evaluates_chained_expression() {
        let a = Expr::input(&arr(1, vec![2, 2], Buffer::F64(vec![1.0, 2.0, 3.0, 4.0])));
        let e = Expr::binary(
            BinaryOp::Pow,
            &Expr::unary(UnaryOp::Neg, &a).unwrap(),
            &Expr::scalar(Scalar::Int(2)),
        )
        .unwrap();
        let m = Expr::reduce(ReduceOp::Sum, &Expr::matmul(&e, &a).unwrap(), Some(vec![1])).unwrap();
        // [[1,4],[9,16]] @ [[1,2],[3,4]] = [[13,18],[57,82]] -> row sums [31,
        // 139]
        assert_eq!(eval(&m).buffer, Buffer::F64(vec![31.0, 139.0]));
    }

    #[test]
    fn indexing_reads_and_masks() {
        let a = arr(1, vec![4], Buffer::I64(vec![10, 20, 30, 40]));
        let e = Expr::index(
            &Expr::input(&a),
            vec![Selector::Mask(vec![true, false, false, true])],
        )
        .unwrap();
        assert_eq!(eval(&e).buffer, Buffer::I64(vec![10, 40]));
        let doubled = Expr::binary(
            BinaryOp::Mul,
            &Expr::input(&a),
            &Expr::scalar(Scalar::Int(2)),
        )
        .unwrap();
        let picked = Expr::index(&doubled, vec![Selector::Indices(vec![-1, 0])]).unwrap();
        assert_eq!(eval(&picked).buffer, Buffer::I64(vec![80, 20]));
    }

    #[test]
    fn gpu_subset_is_checked_statically() {
        let f32a = Expr::input(&arr(1, vec![2], Buffer::F32(vec![1.0, 2.0])));
        let ok = Expr::binary(BinaryOp::Add, &f32a, &Expr::scalar(Scalar::Float(1.0))).unwrap();
        assert!(ok.check_gpu_support().is_ok());
        let f64a = Expr::input(&arr(1, vec![2], Buffer::F64(vec![1.0, 2.0])));
        assert_eq!(
            f64a.check_gpu_support().unwrap_err().code,
            ErrorCode::GpuUnsupportedOperation
        );
    }
}
