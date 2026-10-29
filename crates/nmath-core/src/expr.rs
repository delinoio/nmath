//! Immutable lazy expression graphs with construction-time shape and dtype
//! inference.

use std::{collections::HashMap, sync::Arc};

use crate::{
    array::NdArray,
    buffer::{Buffer, DenseArray, Scalar},
    dtype::DType,
    error::{Error, ErrorCode, Result},
    kernels::{self, BinaryOp, ReduceOp, UnaryOp},
    memory::{Device, MemoryBudget, Reservation},
    random::{Generator, Pcg64},
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
    Reduce(ReduceOp, Expr, Option<Vec<i64>>),
    MatMul(Expr, Expr),
    Index(Expr, Vec<Selector>),
    Cast(Expr, DType),
    /// `Generator.random(shape)` in float64.
    Random(Generator, Vec<usize>),
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
        Expr::node(ExprKind::Scalar(value), value.default_dtype(), vec![], None)
    }

    pub fn random(context_id: u64, generator: &Generator, shape: Vec<usize>) -> Expr {
        Expr::node(
            ExprKind::Random(generator.clone(), shape.clone()),
            DType::Float64,
            shape,
            Some(context_id),
        )
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
        match self.0.kind {
            ExprKind::Scalar(s) => Some(s),
            _ => None,
        }
    }

    pub fn binary(op: BinaryOp, lhs: &Expr, rhs: &Expr) -> Result<Expr> {
        let context_id = join_context(lhs.context_id(), rhs.context_id())?;
        let shape = broadcast_shapes(lhs.shape(), rhs.shape())?;
        let compute_dtype = match (lhs.weak_scalar(), rhs.weak_scalar()) {
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

    pub fn unary(op: UnaryOp, input: &Expr) -> Result<Expr> {
        let dtype = input.dtype().ensure_computable()?;
        if op == UnaryOp::Neg && dtype == DType::Bool {
            return Err(Error::new(
                ErrorCode::DTypeUnsupported,
                "The negative operator is not supported for bool operands.",
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

    pub fn matmul(lhs: &Expr, rhs: &Expr) -> Result<Expr> {
        let context_id = join_context(lhs.context_id(), rhs.context_id())?;
        let dtype = DType::promote(lhs.dtype(), rhs.dtype())?;
        let (a, b) = (lhs.shape(), rhs.shape());
        let k_a = *a.last().ok_or_else(|| scalar_matmul_error())?;
        let k_b = match b.len() {
            0 => return Err(scalar_matmul_error()),
            1 => b[0],
            _ => b[b.len() - 2],
        };
        if a.len() > 2 || b.len() > 2 {
            return Err(Error::new(
                ErrorCode::ShapeMismatch,
                "matmul in this build supports 1-D and 2-D operands.",
            ));
        }
        if k_a != k_b {
            return Err(Error::new(
                ErrorCode::ShapeMismatch,
                format!("matmul: the core dimensions of shapes {a:?} and {b:?} do not match."),
            ));
        }
        let mut shape = Vec::new();
        if a.len() == 2 {
            shape.push(a[0]);
        }
        if b.len() == 2 {
            shape.push(b[1]);
        }
        Ok(Expr::node(
            ExprKind::MatMul(lhs.clone(), rhs.clone()),
            dtype,
            shape,
            context_id,
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
            ExprKind::Reduce(op, _, _) => op.id(),
            ExprKind::MatMul(..) => "matmul",
            ExprKind::Index(..) => "index",
            ExprKind::Cast(..) => "cast",
            ExprKind::Random(..) => "random",
        }
    }

    fn children(&self) -> Vec<&Expr> {
        match &self.0.kind {
            ExprKind::Input(_) | ExprKind::Scalar(_) | ExprKind::Random(..) => vec![],
            ExprKind::Unary(_, e)
            | ExprKind::Reduce(_, e, _)
            | ExprKind::Index(e, _)
            | ExprKind::Cast(e, _) => vec![e],
            ExprKind::Binary { lhs, rhs, .. } | ExprKind::MatMul(lhs, rhs) => vec![lhs, rhs],
        }
    }

    fn visit(&self, f: &mut impl FnMut(&Expr)) {
        f(self);
        for child in self.children() {
            child.visit(f);
        }
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
            if let ExprKind::Random(g, _) = &e.0.kind
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

fn scalar_matmul_error() -> Error {
    Error::new(
        ErrorCode::ShapeMismatch,
        "matmul does not accept scalar operands.",
    )
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
    pub device: Device,
    pub budget: MemoryBudget,
    pub gpu_budget: Option<MemoryBudget>,
    pub token: CancellationToken,
    snapshots: HashMap<*const NdArray, DenseArray>,
    rng: Vec<(Generator, Pcg64)>,
}

// SAFETY: the raw pointer keys are only compared for identity, never
// dereferenced.
unsafe impl Send for JobEnv {}

impl JobEnv {
    pub fn new(device: Device, budget: MemoryBudget, token: CancellationToken) -> Self {
        Self {
            device,
            budget,
            gpu_budget: None,
            token,
            snapshots: HashMap::new(),
            rng: Vec::new(),
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
                    match a.snapshot(&self.budget) {
                        Ok(s) => {
                            self.snapshots.insert(key, s);
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
        if let ExprKind::Index(inner, selectors) = &expr.0.kind
            && let ExprKind::Input(array) = &inner.0.kind
            && selectors.iter().all(Selector::is_basic)
        {
            return array.view(selectors).map(Value::View);
        }
        self.eval_dense(expr).map(Value::Dense)
    }

    /// Evaluates `expr` on the CPU into a new dense array, materializing
    /// views.
    pub(crate) fn eval_dense(&mut self, expr: &Expr) -> Result<DenseArray> {
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
                DenseArray::new(vec![], Buffer::filled(dtype, 1, *s)?, reservation)
            }
            ExprKind::Unary(op, e) => {
                let v = self.eval_dense(e)?;
                kernels::unary(*op, &v, &budget, &token)
            }
            ExprKind::Binary {
                op,
                lhs,
                rhs,
                compute_dtype,
            } => {
                let a = self.eval_dense(lhs)?;
                let b = self.eval_dense(rhs)?;
                kernels::binary(*op, &a, &b, *compute_dtype, &budget, &token)
            }
            ExprKind::Reduce(op, e, axes) => {
                let v = self.eval_dense(e)?;
                kernels::reduce(*op, &v, axes.as_deref(), &budget, &token)
            }
            ExprKind::MatMul(l, r) => {
                let a = self.eval_dense(l)?;
                let b = self.eval_dense(r)?;
                kernels::matmul(&a, &b, expr.dtype(), &budget, &token)
            }
            ExprKind::Index(e, selectors) => {
                let v = self.eval_dense(e)?;
                let tmp = NdArray::from_dense(
                    0,
                    DenseArray::new(v.shape.clone(), v.buffer.clone(), Reservation::none())?,
                );
                tmp.take(selectors, &budget)
            }
            ExprKind::Cast(e, dtype) => {
                let v = self.eval_dense(e)?;
                let reservation = budget.reserve((v.size() * dtype.item_size()) as u64)?;
                DenseArray::new(
                    v.shape.clone(),
                    v.buffer.cast_cancellable(*dtype, &token)?,
                    reservation,
                )
            }
            ExprKind::Random(g, shape) => {
                let len: usize = shape.iter().product();
                let reservation = budget.reserve((len * 8) as u64)?;
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
                let values: Vec<f64> = (0..len).map(|_| slot.1.next_f64()).collect();
                DenseArray::new(shape.clone(), Buffer::F64(values), reservation)
            }
        }
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
