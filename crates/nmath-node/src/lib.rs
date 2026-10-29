//! Node.js binding for `nmath-core`. The TypeScript package wraps these
//! classes; they are not a public interface on their own.

use std::{path::PathBuf, sync::Once};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use nmath_core::{
    BinaryOp, Buffer as CoreBuffer, CancellationToken, Context, ContextOptions, DType, Device,
    Error as CoreError, ErrorCode, Expr, Generator, NdArray, ReduceOp, Scalar, Selector, UnaryOp,
    Value, context::JobOutput, io::WriteMode,
};

static LOGGING: Once = Once::new();

/// Enables `tracing` output on stderr when `NMATH_LOG` is set. ANSI color is on
/// by default and disabled by `NO_COLOR`.
fn init_logging() {
    LOGGING.call_once(|| {
        if let Ok(filter) = std::env::var("NMATH_LOG") {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(tracing_subscriber::EnvFilter::new(filter))
                .with_ansi(std::env::var_os("NO_COLOR").is_none())
                .with_writer(std::io::stderr)
                .try_init();
        }
    });
}

/// Encodes a core error so the TypeScript layer can rebuild an `NMathError`
/// with its code.
fn to_napi(err: CoreError) -> Error {
    Error::new(Status::GenericFailure, err.full_message())
}

fn core_err(code: ErrorCode, message: impl Into<String>) -> Error {
    to_napi(CoreError::new(code, message))
}

fn disposed() -> Error {
    core_err(ErrorCode::Disposed, "The object was disposed.")
}

fn parse_dtype(id: &str) -> Result<DType> {
    DType::from_id(id).map_err(to_napi)
}

fn to_u64(value: &BigInt, name: &str) -> Result<u64> {
    let (signed, v, lossless) = value.get_u64();
    if signed || !lossless {
        return Err(core_err(
            ErrorCode::ContextInvalidOptions,
            format!("{name} must be a non-negative 64-bit integer."),
        ));
    }
    Ok(v)
}

#[napi(object)]
pub struct NativeContextOptions {
    pub max_concurrent_jobs: u32,
    pub max_queue_length: u32,
    pub cpu_memory_budget: BigInt,
    pub gpu_memory_budget: BigInt,
}

#[napi(object)]
pub struct NativeSelector {
    /// `index`, `slice`, `indices` or `mask`.
    pub kind: String,
    pub index: Option<i64>,
    pub start: Option<i64>,
    pub stop: Option<i64>,
    pub step: Option<i64>,
    pub indices: Option<Vec<i64>>,
    pub mask: Option<Vec<bool>>,
}

fn to_selectors(selectors: Vec<NativeSelector>) -> Result<Vec<Selector>> {
    selectors
        .into_iter()
        .map(|s| {
            Ok(match s.kind.as_str() {
                "index" => Selector::Index(s.index.unwrap_or(0)),
                "slice" => Selector::Slice {
                    start: s.start,
                    stop: s.stop,
                    step: s.step,
                },
                "indices" => Selector::Indices(s.indices.unwrap_or_default()),
                "mask" => Selector::Mask(s.mask.unwrap_or_default()),
                other => {
                    return Err(core_err(
                        ErrorCode::IndexInvalid,
                        format!("Unknown selector kind '{other}'."),
                    ));
                }
            })
        })
        .collect()
}

/// A cancellation handle the TypeScript layer connects to an `AbortSignal`.
#[napi]
pub struct NativeCancellation {
    token: CancellationToken,
}

#[napi]
impl NativeCancellation {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            token: CancellationToken::new(),
        }
    }

    #[napi]
    pub fn cancel(&self) {
        self.token.cancel();
    }

    #[napi(getter)]
    pub fn cancelled(&self) -> bool {
        self.token.is_cancelled()
    }
}

impl Default for NativeCancellation {
    fn default() -> Self {
        Self::new()
    }
}

#[napi]
pub struct NativeArray {
    inner: Option<NdArray>,
}

impl NativeArray {
    fn get(&self) -> Result<&NdArray> {
        self.inner.as_ref().ok_or_else(disposed)
    }
}

fn buffer_from_bytes(dtype: DType, bytes: &[u8]) -> Result<CoreBuffer> {
    let size = dtype.item_size();
    if !bytes.len().is_multiple_of(size) {
        return Err(core_err(
            ErrorCode::ShapeMismatch,
            "The byte length is not a multiple of the item size.",
        ));
    }
    macro_rules! read {
        ($t:ty) => {
            bytes
                .chunks_exact(size)
                .map(|c| <$t>::from_ne_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>()
                .into()
        };
    }
    Ok(match dtype.ensure_computable().map_err(to_napi)? {
        DType::Bool => CoreBuffer::Bool(bytes.iter().map(|&b| b != 0).collect()),
        DType::Int8 => read!(i8),
        DType::Int16 => read!(i16),
        DType::Int32 => read!(i32),
        DType::Int64 => read!(i64),
        DType::UInt8 => read!(u8),
        DType::UInt16 => read!(u16),
        DType::UInt32 => read!(u32),
        DType::UInt64 => read!(u64),
        DType::Float16 => read!(nmath_core::buffer::Float16),
        DType::Float32 => read!(f32),
        DType::Float64 => read!(f64),
        DType::Complex64 => CoreBuffer::C64(
            bytes
                .chunks_exact(8)
                .map(|c| {
                    [
                        f32::from_ne_bytes(c[..4].try_into().unwrap()),
                        f32::from_ne_bytes(c[4..].try_into().unwrap()),
                    ]
                })
                .collect(),
        ),
        DType::Complex128 => CoreBuffer::C128(
            bytes
                .chunks_exact(16)
                .map(|c| {
                    [
                        f64::from_ne_bytes(c[..8].try_into().unwrap()),
                        f64::from_ne_bytes(c[8..].try_into().unwrap()),
                    ]
                })
                .collect(),
        ),
        _ => unreachable!(),
    })
}

fn buffer_to_bytes(buffer: &CoreBuffer) -> Vec<u8> {
    match buffer {
        CoreBuffer::Bool(v) => v.iter().map(|&b| b as u8).collect(),
        CoreBuffer::I8(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        CoreBuffer::I16(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        CoreBuffer::I32(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        CoreBuffer::I64(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        CoreBuffer::U8(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        CoreBuffer::U16(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        CoreBuffer::U32(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        CoreBuffer::U64(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        CoreBuffer::F16(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        CoreBuffer::F32(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        CoreBuffer::F64(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        CoreBuffer::C64(v) => v
            .iter()
            .flat_map(|z| z.iter().flat_map(|x| x.to_ne_bytes()))
            .collect(),
        CoreBuffer::C128(v) => v
            .iter()
            .flat_map(|z| z.iter().flat_map(|x| x.to_ne_bytes()))
            .collect(),
    }
}

#[napi(object)]
pub struct NativeComplex {
    pub real: f64,
    pub imag: f64,
}

#[napi]
impl NativeArray {
    #[napi(getter)]
    pub fn dtype(&self) -> Result<String> {
        Ok(self.get()?.dtype().id().to_string())
    }

    #[napi(getter)]
    pub fn shape(&self) -> Result<Vec<u32>> {
        Ok(self.get()?.shape().iter().map(|&d| d as u32).collect())
    }

    #[napi(getter)]
    pub fn context_id(&self) -> Result<i64> {
        Ok(self.get()?.context_id() as i64)
    }

    #[napi(getter)]
    pub fn disposed(&self) -> bool {
        self.inner.is_none()
    }

    /// Copies the visible elements in C order using native byte order.
    #[napi]
    pub fn to_bytes(&self) -> Result<Buffer> {
        Ok(buffer_to_bytes(&self.get()?.to_buffer()).into())
    }

    /// Reads one element. 64-bit integers are returned as bigint so they stay
    /// exact.
    #[napi]
    pub fn item(&self, index: Vec<i64>) -> Result<Either4<f64, BigInt, bool, NativeComplex>> {
        let array = self.get()?;
        let wide = matches!(array.dtype(), DType::Int64 | DType::UInt64);
        let value = array.get_scalar(&index).map_err(to_napi)?;
        Ok(match value {
            Scalar::Bool(b) => Either4::C(b),
            Scalar::Int(i) if wide => Either4::B(BigInt::from(i)),
            Scalar::UInt(u) if wide => Either4::B(BigInt::from(u)),
            // 32-bit integers are exact as JS numbers.
            Scalar::Int(i) => Either4::A(i as f64),
            Scalar::UInt(u) => Either4::A(u as f64),
            Scalar::Float(f) => Either4::A(f),
            Scalar::Complex([real, imag]) => Either4::D(NativeComplex { real, imag }),
        })
    }

    #[napi]
    pub fn shares_storage(&self, other: &NativeArray) -> Result<bool> {
        Ok(self.get()?.shares_storage(other.get()?))
    }

    /// Releases this handle. Storage stays alive while other views or running
    /// jobs use it.
    #[napi]
    pub fn dispose(&mut self) {
        self.inner = None;
    }
}

#[napi]
pub struct NativeGenerator {
    inner: Generator,
    context_id: u64,
}

#[napi]
pub struct NativeExpr {
    inner: Expr,
}

#[napi]
impl NativeExpr {
    #[napi(factory)]
    pub fn input(array: &NativeArray) -> Result<Self> {
        Ok(Self {
            inner: Expr::input(array.get()?),
        })
    }

    #[napi(factory)]
    pub fn complex_scalar(real: f64, imag: f64) -> Self {
        Self {
            inner: Expr::scalar(Scalar::Complex([real, imag])),
        }
    }

    #[napi(factory)]
    pub fn float_scalar(value: f64) -> Self {
        Self {
            inner: Expr::scalar(Scalar::Float(value)),
        }
    }

    #[napi(factory)]
    pub fn int_scalar(value: BigInt) -> Result<Self> {
        // `BigInt::get_u64` reports every negative value as lossy, so read the
        // sign and words directly. Negative magnitudes up to 2^63 are valid;
        // `-2^63` maps to `i64::MIN` exactly.
        let magnitude = match value.words.split_first() {
            Some((&low, high)) if high.iter().all(|&w| w == 0) => low,
            None => 0,
            Some(_) => {
                return Err(core_err(
                    ErrorCode::DTypeUnsupported,
                    "Integer scalars must fit in 64 bits.",
                ));
            }
        };
        let scalar = if value.sign_bit && magnitude != 0 {
            if magnitude > i64::MIN.unsigned_abs() {
                return Err(core_err(
                    ErrorCode::DTypeUnsupported,
                    "Negative integer scalars must be at least -2^63.",
                ));
            }
            Scalar::Int((magnitude as i64).wrapping_neg())
        } else if magnitude > i64::MAX as u64 {
            Scalar::UInt(magnitude)
        } else {
            Scalar::Int(magnitude as i64)
        };
        Ok(Self {
            inner: Expr::scalar(scalar),
        })
    }

    #[napi(factory)]
    pub fn bool_scalar(value: bool) -> Self {
        Self {
            inner: Expr::scalar(Scalar::Bool(value)),
        }
    }

    #[napi(factory)]
    pub fn binary(op: String, lhs: &NativeExpr, rhs: &NativeExpr) -> Result<Self> {
        let op = BinaryOp::from_id(&op).map_err(to_napi)?;
        Ok(Self {
            inner: Expr::binary(op, &lhs.inner, &rhs.inner).map_err(to_napi)?,
        })
    }

    #[napi(factory)]
    pub fn unary(op: String, input: &NativeExpr) -> Result<Self> {
        let op = match op.as_str() {
            "pos" => UnaryOp::Pos,
            "neg" => UnaryOp::Neg,
            other => {
                return Err(core_err(
                    ErrorCode::IndexInvalid,
                    format!("Unknown unary operator '{other}'."),
                ));
            }
        };
        Ok(Self {
            inner: Expr::unary(op, &input.inner).map_err(to_napi)?,
        })
    }

    #[napi(factory)]
    pub fn reduce(op: String, input: &NativeExpr, axes: Option<Vec<i64>>) -> Result<Self> {
        let op = match op.as_str() {
            "sum" => ReduceOp::Sum,
            "mean" => ReduceOp::Mean,
            "min" => ReduceOp::Min,
            "max" => ReduceOp::Max,
            other => {
                return Err(core_err(
                    ErrorCode::IndexInvalid,
                    format!("Unknown reduction '{other}'."),
                ));
            }
        };
        Ok(Self {
            inner: Expr::reduce(op, &input.inner, axes).map_err(to_napi)?,
        })
    }

    #[napi(factory)]
    pub fn matmul(lhs: &NativeExpr, rhs: &NativeExpr) -> Result<Self> {
        Ok(Self {
            inner: Expr::matmul(&lhs.inner, &rhs.inner).map_err(to_napi)?,
        })
    }

    #[napi(factory)]
    pub fn linalg(op: String, input: &NativeExpr, rhs: Option<&NativeExpr>) -> Result<Self> {
        use nmath_core::linalg::LinalgOp;
        let op = match op.as_str() {
            "solve" => LinalgOp::Solve,
            "inv" => LinalgOp::Inv,
            "det" => LinalgOp::Det,
            _ => {
                return Err(to_napi(nmath_core::Error::new(
                    nmath_core::ErrorCode::DTypeUnsupported,
                    "Unknown linalg operation.",
                )));
            }
        };
        Ok(Self {
            inner: Expr::linalg(op, &input.inner, rhs.map(|v| &v.inner)).map_err(to_napi)?,
        })
    }

    #[napi(factory)]
    pub fn fft(
        op: String,
        input: &NativeExpr,
        n: Option<u32>,
        axis: i64,
        norm: String,
    ) -> Result<Self> {
        use nmath_core::fft::{FftOp, FftOptions, Norm};
        let invalid = || {
            to_napi(nmath_core::Error::new(
                nmath_core::ErrorCode::DTypeUnsupported,
                "Unknown FFT operation or normalization.",
            ))
        };
        let op = match op.as_str() {
            "fft" => FftOp::Fft,
            "ifft" => FftOp::Ifft,
            "rfft" => FftOp::Rfft,
            "irfft" => FftOp::Irfft,
            _ => return Err(invalid()),
        };
        let norm = match norm.as_str() {
            "backward" => Norm::Backward,
            "forward" => Norm::Forward,
            "ortho" => Norm::Ortho,
            _ => return Err(invalid()),
        };
        Ok(Self {
            inner: Expr::fft(
                FftOptions {
                    op,
                    n: n.map(|v| v as usize),
                    axis,
                    norm,
                },
                &input.inner,
            )
            .map_err(to_napi)?,
        })
    }

    #[napi(factory)]
    pub fn index(input: &NativeExpr, selectors: Vec<NativeSelector>) -> Result<Self> {
        Ok(Self {
            inner: Expr::index(&input.inner, to_selectors(selectors)?).map_err(to_napi)?,
        })
    }

    #[napi(factory)]
    pub fn cast(input: &NativeExpr, dtype: String) -> Result<Self> {
        Ok(Self {
            inner: Expr::cast(&input.inner, parse_dtype(&dtype)?).map_err(to_napi)?,
        })
    }

    #[napi(factory)]
    pub fn random(generator: &NativeGenerator, shape: Vec<u32>) -> Self {
        Self {
            inner: Expr::random(
                generator.context_id,
                &generator.inner,
                shape.into_iter().map(|d| d as usize).collect(),
            ),
        }
    }

    #[napi(factory)]
    pub fn random_distribution(
        generator: &NativeGenerator,
        shape: Vec<u32>,
        operation: String,
        dtype: String,
        first: String,
        second: String,
        endpoint: bool,
    ) -> Result<Self> {
        use nmath_core::random::RandomOperation;
        let dtype = parse_dtype(&dtype)?;
        let invalid = || {
            napi::Error::from_reason(
                "NMATH_DTYPE_UNSUPPORTED: Invalid random distribution parameters.",
            )
        };
        let op = match operation.as_str() {
            "random" => RandomOperation::Random(dtype),
            "integers" => RandomOperation::Integers {
                low: first.parse().map_err(|_| invalid())?,
                high: second.parse().map_err(|_| invalid())?,
                endpoint,
                dtype,
            },
            "uniform" => RandomOperation::Uniform {
                low: first.parse().map_err(|_| invalid())?,
                high: second.parse().map_err(|_| invalid())?,
            },
            "standard_normal" => RandomOperation::StandardNormal(dtype),
            "normal" => RandomOperation::Normal {
                loc: first.parse().map_err(|_| invalid())?,
                scale: second.parse().map_err(|_| invalid())?,
            },
            "exponential" => RandomOperation::Exponential {
                scale: first.parse().map_err(|_| invalid())?,
            },
            _ => return Err(invalid()),
        };
        Ok(Self {
            inner: Expr::random_operation(
                generator.context_id,
                &generator.inner,
                shape.into_iter().map(|d| d as usize).collect(),
                op,
            )
            .map_err(to_napi)?,
        })
    }

    #[napi(getter)]
    pub fn dtype(&self) -> String {
        self.inner.dtype().id().to_string()
    }

    #[napi(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.inner.shape().iter().map(|&d| d as u32).collect()
    }

    #[napi(getter)]
    pub fn context_id(&self) -> Option<i64> {
        self.inner.context_id().map(|id| id as i64)
    }
}

#[napi(object)]
pub struct NativeExecution {
    pub id: i64,
    pub kind: String,
    pub operation: String,
    pub device: String,
    pub status: String,
    pub queue_time_ms: f64,
    pub execution_time_ms: f64,
}

#[napi(object)]
pub struct NativeMemoryUsage {
    pub limit: BigInt,
    pub used: BigInt,
    pub peak: BigInt,
    pub cumulative_reserved: BigInt,
    pub rejected: i64,
}

#[napi(object)]
pub struct NativeMetrics {
    pub executions: Vec<NativeExecution>,
    pub total_submitted: i64,
    pub total_succeeded: i64,
    pub total_failed: i64,
    pub total_cancelled: i64,
    pub total_rejected: i64,
    pub cpu_memory: NativeMemoryUsage,
    pub gpu_memory: NativeMemoryUsage,
}

fn memory(usage: nmath_core::memory::MemoryUsage) -> NativeMemoryUsage {
    NativeMemoryUsage {
        limit: BigInt::from(usage.limit),
        used: BigInt::from(usage.used),
        peak: BigInt::from(usage.peak),
        cumulative_reserved: BigInt::from(usage.cumulative_reserved),
        rejected: usage.rejected as i64,
    }
}

#[napi]
pub struct NativeContext {
    inner: Context,
}

type Resolver = Box<dyn FnOnce(Env) -> Result<NativeResult> + Send>;

/// Result of a job: an array, or nothing for writes.
pub enum NativeResult {
    Array(NativeArray),
    Unit,
}

impl ToNapiValue for NativeResult {
    unsafe fn to_napi_value(env: napi::sys::napi_env, val: Self) -> Result<napi::sys::napi_value> {
        // SAFETY: forwards to the generated conversions with the same
        // environment.
        unsafe {
            match val {
                NativeResult::Array(a) => NativeArray::to_napi_value(env, a),
                NativeResult::Unit => <()>::to_napi_value(env, ()),
            }
        }
    }
}

/// Retains archive metadata until conversion has created all JS entries.
struct NativeNpzResult(nmath_core::io_npz::NpzArrays);

impl ToNapiValue for NativeNpzResult {
    unsafe fn to_napi_value(
        env: napi::sys::napi_env,
        mut val: Self,
    ) -> Result<napi::sys::napi_value> {
        let entries = std::mem::take(&mut val.0.arrays)
            .into_iter()
            .map(|(name, array)| NativeNpzEntry {
                name,
                array: NativeArray { inner: Some(array) },
            })
            .collect::<Vec<_>>();
        // SAFETY: forwards to generated entry conversions in the same N-API
        // environment. The archive still owns its metadata reservation.
        let result = unsafe { Vec::<NativeNpzEntry>::to_napi_value(env, entries) };
        drop(val);
        result
    }
}

impl NativeContext {
    /// Admits a job and returns its Promise. Admission errors reject
    /// synchronously-created promises so callers always receive a Promise.
    fn promise<'env>(
        &self,
        env: &'env Env,
        submit: impl FnOnce(&Context, nmath_core::context::Done) -> nmath_core::Result<u64>,
    ) -> Result<Object<'env>> {
        let (deferred, promise) = env.create_deferred::<NativeResult, Resolver>()?;
        let slot = std::sync::Arc::new(std::sync::Mutex::new(Some(deferred)));
        let job_slot = slot.clone();
        let context_id = self.inner.id();
        let done: nmath_core::context::Done = Box::new(move |result| {
            let Some(deferred) = job_slot.lock().unwrap().take() else {
                return;
            };
            match result {
                Ok(JobOutput::Value(Value::Dense(dense))) => {
                    let ctx_array = NdArray::from_dense(context_id, dense);
                    deferred.resolve(Box::new(move |_| {
                        Ok(NativeResult::Array(NativeArray {
                            inner: Some(ctx_array),
                        }))
                    }));
                }
                Ok(JobOutput::Value(Value::View(view))) | Ok(JobOutput::Array(view)) => deferred
                    .resolve(Box::new(move |_| {
                        Ok(NativeResult::Array(NativeArray { inner: Some(view) }))
                    })),
                Ok(JobOutput::Unit) => deferred.resolve(Box::new(|_| Ok(NativeResult::Unit))),
                Err(err) => deferred.reject(to_napi(err)),
            }
        });
        if let Err(err) = submit(&self.inner, done)
            && let Some(deferred) = slot.lock().unwrap().take()
        {
            deferred.reject(to_napi(err));
        }
        Ok(promise)
    }
}

impl NativeContext {
    /// Advisory budget check before a native buffer is allocated. It cannot
    /// reserve (the core reserves only inside `Context::array`), so a
    /// concurrent reservation between this check and `Context::array` still
    /// fails there with the same code. Rejections here do not increment the
    /// core `rejected` counter. Remove this when the core exposes a
    /// reserve-then-fill array constructor.
    fn precheck_cpu_budget(&self, bytes: u64) -> Result<()> {
        if self.inner.is_closed() {
            return Err(core_err(ErrorCode::ContextClosed, "The context is closed."));
        }
        let usage = self.inner.metrics().cpu_memory;
        if usage.used.saturating_add(bytes) > usage.limit {
            tracing::debug!(
                event = "nmath.memory.rejected",
                device = "cpu",
                stage = "precheck",
                requested = bytes,
                used = usage.used,
                limit = usage.limit
            );
            return Err(core_err(
                ErrorCode::MemoryBudgetExceeded,
                format!(
                    "Reserving {bytes} bytes on cpu would exceed the budget ({} of {} bytes in \
                     use).",
                    usage.used, usage.limit
                ),
            ));
        }
        Ok(())
    }
}

fn token_of(signal: Option<&NativeCancellation>) -> CancellationToken {
    signal.map(|s| s.token.clone()).unwrap_or_default()
}

#[napi]
impl NativeContext {
    #[napi(constructor)]
    pub fn new(options: NativeContextOptions) -> Result<Self> {
        init_logging();
        let inner = Context::new(ContextOptions {
            max_concurrent_jobs: options.max_concurrent_jobs as usize,
            max_queue_length: options.max_queue_length as usize,
            cpu_memory_budget: to_u64(&options.cpu_memory_budget, "cpuMemoryBudget")?,
            gpu_memory_budget: to_u64(&options.gpu_memory_budget, "gpuMemoryBudget")?,
        })
        .map_err(to_napi)?;
        Ok(Self { inner })
    }

    #[napi(getter)]
    pub fn id(&self) -> i64 {
        self.inner.id() as i64
    }

    #[napi(getter)]
    pub fn closed(&self) -> bool {
        self.inner.is_closed()
    }

    /// Fails with `NMATH_MEMORY_BUDGET_EXCEEDED` when `bytes` more would not
    /// fit the CPU budget now. Creation helpers call it before they build
    /// input buffers.
    #[napi]
    pub fn check_cpu_budget(&self, bytes: BigInt) -> Result<()> {
        self.precheck_cpu_budget(to_u64(&bytes, "bytes")?)
    }

    #[napi]
    pub fn array(&self, shape: Vec<u32>, dtype: String, data: Buffer) -> Result<NativeArray> {
        let dtype = parse_dtype(&dtype)?;
        let item_size = dtype.ensure_computable().map_err(to_napi)?.item_size();
        let bytes = shape
            .iter()
            .try_fold(item_size, |acc, &d| acc.checked_mul(d as usize))
            .ok_or_else(|| core_err(ErrorCode::ShapeMismatch, "The shape is too large."))?;
        if data.len() != bytes {
            return Err(core_err(
                ErrorCode::ShapeMismatch,
                format!(
                    "The input has {} bytes, but shape {shape:?} with dtype {} needs {bytes}.",
                    data.len(),
                    dtype.id()
                ),
            ));
        }
        // Reject before copying the input into a native buffer. Context::array
        // still makes the authoritative reservation afterwards.
        self.precheck_cpu_budget(bytes as u64)?;
        let buffer = buffer_from_bytes(dtype, &data)?;
        let array = self
            .inner
            .array(shape.into_iter().map(|d| d as usize).collect(), buffer)
            .map_err(to_napi)?;
        Ok(NativeArray { inner: Some(array) })
    }

    #[napi]
    pub fn generator(&self, seed: BigInt) -> Result<NativeGenerator> {
        let (negative, low, lossless) = seed.get_u128();
        if negative || !lossless {
            return Err(core_err(
                ErrorCode::IndexInvalid,
                "A seed must be a non-negative integer below 2^128.",
            ));
        }
        Ok(NativeGenerator {
            inner: self.inner.generator(low),
            context_id: self.inner.id(),
        })
    }

    #[napi(ts_return_type = "Promise<NativeArray>")]
    pub fn compute<'env>(
        &self,
        env: &'env Env,
        expr: &NativeExpr,
        device: String,
        signal: Option<&NativeCancellation>,
    ) -> Result<Object<'env>> {
        let token = token_of(signal);
        let expr = expr.inner.clone();
        self.promise(env, move |ctx, done| {
            let device = Device::from_id(&device)?;
            ctx.compute(&expr, device, token, done)
        })
    }

    #[napi(ts_return_type = "Promise<void>")]
    pub fn assign<'env>(
        &self,
        env: &'env Env,
        target: &NativeArray,
        selectors: Vec<NativeSelector>,
        value: &NativeExpr,
        signal: Option<&NativeCancellation>,
    ) -> Result<Object<'env>> {
        let target = target.get()?.clone();
        let selectors = to_selectors(selectors)?;
        let value = value.inner.clone();
        let token = token_of(signal);
        self.promise(env, move |ctx, done| {
            ctx.assign(&target, selectors, &value, token, done)
        })
    }

    #[napi(ts_return_type = "Promise<NativeArray>")]
    pub fn copy_from<'env>(
        &self,
        env: &'env Env,
        source: &NativeArray,
        signal: Option<&NativeCancellation>,
    ) -> Result<Object<'env>> {
        let source = source.get()?.clone();
        let token = token_of(signal);
        self.promise(env, move |ctx, done| ctx.copy_from(&source, token, done))
    }

    #[napi(ts_return_type = "Promise<void>")]
    pub fn save_npy<'env>(
        &self,
        env: &'env Env,
        array: &NativeArray,
        path: String,
        overwrite: bool,
        signal: Option<&NativeCancellation>,
    ) -> Result<Object<'env>> {
        let array = array.get()?.clone();
        let token = token_of(signal);
        let mode = if overwrite {
            WriteMode::Overwrite
        } else {
            WriteMode::CreateNew
        };
        self.promise(env, move |ctx, done| {
            ctx.save_npy(&array, PathBuf::from(path), mode, token, done)
        })
    }

    #[napi(ts_return_type = "Promise<NativeArray>")]
    pub fn load_npy<'env>(
        &self,
        env: &'env Env,
        path: String,
        signal: Option<&NativeCancellation>,
    ) -> Result<Object<'env>> {
        let token = token_of(signal);
        self.promise(env, move |ctx, done| {
            ctx.load_npy(PathBuf::from(path), token, done)
        })
    }

    #[napi(ts_return_type = "Promise<void>")]
    pub fn save_npz<'env>(
        &self,
        env: &'env Env,
        names: Vec<String>,
        arrays: Vec<&NativeArray>,
        path: String,
        compressed: bool,
        overwrite: bool,
        signal: Option<&NativeCancellation>,
    ) -> Result<Object<'env>> {
        if names.len() != arrays.len() {
            return Err(core_err(
                ErrorCode::IoFormat,
                "NPZ names and arrays must have equal lengths.",
            ));
        }
        let arrays = names
            .into_iter()
            .zip(arrays)
            .map(|(name, array)| Ok((name, array.get()?.clone())))
            .collect::<Result<Vec<_>>>()?;
        let token = token_of(signal);
        let compression = if compressed {
            nmath_core::io_npz::NpzCompression::Deflated
        } else {
            nmath_core::io_npz::NpzCompression::Stored
        };
        let mode = if overwrite {
            WriteMode::Overwrite
        } else {
            WriteMode::CreateNew
        };
        self.promise(env, move |ctx, done| {
            ctx.save_npz(arrays, PathBuf::from(path), compression, mode, token, done)
        })
    }

    #[napi(ts_return_type = "Promise<Array<{ name: string; array: NativeArray }>>")]
    pub fn load_npz<'env>(
        &self,
        env: &'env Env,
        path: String,
        signal: Option<&NativeCancellation>,
    ) -> Result<Object<'env>> {
        type ArchiveResolver = Box<dyn FnOnce(Env) -> Result<NativeNpzResult> + Send>;
        let (deferred, promise) = env.create_deferred::<NativeNpzResult, ArchiveResolver>()?;
        let slot = std::sync::Arc::new(std::sync::Mutex::new(Some(deferred)));
        let job_slot = slot.clone();
        let done = Box::new(
            move |result: nmath_core::Result<nmath_core::io_npz::NpzArrays>| {
                let Some(deferred) = job_slot.lock().unwrap().take() else {
                    return;
                };
                match result {
                    Ok(archive) => {
                        deferred.resolve(Box::new(move |_| Ok(NativeNpzResult(archive))))
                    }
                    Err(error) => deferred.reject(to_napi(error)),
                }
            },
        );
        if let Err(error) = self
            .inner
            .load_npz(PathBuf::from(path), token_of(signal), done)
            && let Some(deferred) = slot.lock().unwrap().take()
        {
            deferred.reject(to_napi(error));
        }
        Ok(promise)
    }

    /// Closes the context on a helper thread and resolves when cleanup
    /// finishes.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn close<'env>(&self, env: &'env Env) -> Result<Object<'env>> {
        let (deferred, promise) =
            env.create_deferred::<(), Box<dyn FnOnce(Env) -> Result<()> + Send>>()?;
        let ctx = self.inner.clone();
        std::thread::spawn(move || {
            ctx.close();
            deferred.resolve(Box::new(|_| Ok(())));
        });
        Ok(promise)
    }

    #[napi]
    pub fn metrics(&self) -> NativeMetrics {
        let m = self.inner.metrics();
        NativeMetrics {
            executions: m
                .executions
                .iter()
                .map(|r| NativeExecution {
                    id: r.id as i64,
                    kind: r.kind.id().to_string(),
                    operation: r.operation.to_string(),
                    device: r.device.id().to_string(),
                    status: r.status.id().to_string(),
                    queue_time_ms: r.queue_time.as_secs_f64() * 1000.0,
                    execution_time_ms: r.execution_time.as_secs_f64() * 1000.0,
                })
                .collect(),
            total_submitted: m.total_submitted as i64,
            total_succeeded: m.total_succeeded as i64,
            total_failed: m.total_failed as i64,
            total_cancelled: m.total_cancelled as i64,
            total_rejected: m.total_rejected as i64,
            cpu_memory: memory(m.cpu_memory),
            gpu_memory: memory(m.gpu_memory),
        }
    }
}

mod table;

#[napi(object, object_from_js = false)]
pub struct NativeNpzEntry {
    pub name: String,
    pub array: NativeArray,
}

mod columnar;
mod io_text;
