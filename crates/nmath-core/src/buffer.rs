//! Typed contiguous buffers and scalar values.

use crate::{
    dtype::{DType, ScalarKind},
    error::{Error, ErrorCode, Result},
    memory::Reservation,
    sync::CancellationToken,
};

/// A JS-facing scalar. 64-bit integers are carried exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Complex([f64; 2]),
}

impl Scalar {
    pub fn kind(self) -> ScalarKind {
        match self {
            Scalar::Bool(_) => ScalarKind::Bool,
            Scalar::Int(_) | Scalar::UInt(_) => ScalarKind::Int,
            Scalar::Float(_) => ScalarKind::Float,
            Scalar::Complex(_) => ScalarKind::Complex,
        }
    }

    /// Dtype of a scalar that stands alone (both operands are scalars).
    pub fn default_dtype(self) -> DType {
        match self {
            Scalar::Bool(_) => DType::Bool,
            Scalar::Int(_) => DType::Int64,
            Scalar::UInt(_) => DType::UInt64,
            Scalar::Float(_) => DType::Float64,
            Scalar::Complex(_) => DType::Complex128,
        }
    }

    pub fn as_f64(self) -> f64 {
        match self {
            Scalar::Bool(b) => b as u8 as f64,
            Scalar::Int(i) => i as f64,
            Scalar::UInt(u) => u as f64,
            Scalar::Float(f) => f,
            Scalar::Complex(_) => panic!("complex conversion must be explicit"),
        }
    }

    pub fn as_complex(self) -> [f64; 2] {
        match self {
            Scalar::Complex(z) => z,
            other => [other.as_f64(), 0.0],
        }
    }

    fn as_i128(self) -> i128 {
        match self {
            Scalar::Bool(b) => b as i128,
            Scalar::Int(i) => i as i128,
            Scalar::UInt(u) => u as i128,
            Scalar::Float(f) => f as i128,
            Scalar::Complex(_) => panic!("complex conversion must be explicit"),
        }
    }
}

/// Contiguous typed storage for computable dtypes.
#[derive(Clone, Debug, PartialEq)]
pub enum Buffer {
    Bool(Vec<bool>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    C64(Vec<[f32; 2]>),
    C128(Vec<[f64; 2]>),
}

macro_rules! each_buffer {
    ($buf:expr, $v:ident => $body:expr) => {
        match $buf {
            Buffer::Bool($v) => $body,
            Buffer::I8($v) => $body,
            Buffer::I16($v) => $body,
            Buffer::I32($v) => $body,
            Buffer::I64($v) => $body,
            Buffer::U8($v) => $body,
            Buffer::U16($v) => $body,
            Buffer::U32($v) => $body,
            Buffer::U64($v) => $body,
            Buffer::F32($v) => $body,
            Buffer::F64($v) => $body,
            Buffer::C64($v) => $body,
            Buffer::C128($v) => $body,
        }
    };
}

impl Buffer {
    pub fn dtype(&self) -> DType {
        match self {
            Buffer::Bool(_) => DType::Bool,
            Buffer::I8(_) => DType::Int8,
            Buffer::I16(_) => DType::Int16,
            Buffer::I32(_) => DType::Int32,
            Buffer::I64(_) => DType::Int64,
            Buffer::U8(_) => DType::UInt8,
            Buffer::U16(_) => DType::UInt16,
            Buffer::U32(_) => DType::UInt32,
            Buffer::U64(_) => DType::UInt64,
            Buffer::F32(_) => DType::Float32,
            Buffer::F64(_) => DType::Float64,
            Buffer::C64(_) => DType::Complex64,
            Buffer::C128(_) => DType::Complex128,
        }
    }

    pub fn len(&self) -> usize {
        each_buffer!(self, v => v.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn byte_len(&self) -> u64 {
        (self.len() * self.dtype().item_size()) as u64
    }

    pub fn filled(dtype: DType, len: usize, value: Scalar) -> Result<Buffer> {
        if matches!(value, Scalar::Complex(_))
            && !matches!(dtype, DType::Complex64 | DType::Complex128)
        {
            return Err(Error::new(
                ErrorCode::DTypeUnsupported,
                "Complex values require a complex dtype. Use an explicit cast for a lossy \
                 conversion.",
            ));
        }
        Ok(match dtype.ensure_computable()? {
            DType::Bool => Buffer::Bool(vec![value.as_f64() != 0.0; len]),
            DType::Int8 => Buffer::I8(vec![value.as_i128() as i8; len]),
            DType::Int16 => Buffer::I16(vec![value.as_i128() as i16; len]),
            DType::Int32 => Buffer::I32(vec![value.as_i128() as i32; len]),
            DType::Int64 => Buffer::I64(vec![value.as_i128() as i64; len]),
            DType::UInt8 => Buffer::U8(vec![value.as_i128() as u8; len]),
            DType::UInt16 => Buffer::U16(vec![value.as_i128() as u16; len]),
            DType::UInt32 => Buffer::U32(vec![value.as_i128() as u32; len]),
            DType::UInt64 => Buffer::U64(vec![value.as_i128() as u64; len]),
            DType::Float32 => Buffer::F32(vec![value.as_f64() as f32; len]),
            DType::Float64 => Buffer::F64(vec![value.as_f64(); len]),
            DType::Complex64 => Buffer::C64(vec![value.as_complex().map(|x| x as f32); len]),
            DType::Complex128 => Buffer::C128(vec![value.as_complex(); len]),
            _ => unreachable!(),
        })
    }

    pub fn get(&self, i: usize) -> Scalar {
        match self {
            Buffer::Bool(v) => Scalar::Bool(v[i]),
            Buffer::I8(v) => Scalar::Int(v[i] as i64),
            Buffer::I16(v) => Scalar::Int(v[i] as i64),
            Buffer::I32(v) => Scalar::Int(v[i] as i64),
            Buffer::I64(v) => Scalar::Int(v[i]),
            Buffer::U8(v) => Scalar::UInt(v[i] as u64),
            Buffer::U16(v) => Scalar::UInt(v[i] as u64),
            Buffer::U32(v) => Scalar::UInt(v[i] as u64),
            Buffer::U64(v) => Scalar::UInt(v[i]),
            Buffer::F32(v) => Scalar::Float(v[i] as f64),
            Buffer::F64(v) => Scalar::Float(v[i]),
            Buffer::C64(v) => Scalar::Complex(v[i].map(f64::from)),
            Buffer::C128(v) => Scalar::Complex(v[i]),
        }
    }

    /// Copies one element from `src[j]` into `self[i]`; both buffers have the
    /// same dtype.
    pub fn set_from(&mut self, i: usize, src: &Buffer, j: usize) {
        match (self, src) {
            (Buffer::Bool(a), Buffer::Bool(b)) => a[i] = b[j],
            (Buffer::I8(a), Buffer::I8(b)) => a[i] = b[j],
            (Buffer::I16(a), Buffer::I16(b)) => a[i] = b[j],
            (Buffer::I32(a), Buffer::I32(b)) => a[i] = b[j],
            (Buffer::I64(a), Buffer::I64(b)) => a[i] = b[j],
            (Buffer::U8(a), Buffer::U8(b)) => a[i] = b[j],
            (Buffer::U16(a), Buffer::U16(b)) => a[i] = b[j],
            (Buffer::U32(a), Buffer::U32(b)) => a[i] = b[j],
            (Buffer::U64(a), Buffer::U64(b)) => a[i] = b[j],
            (Buffer::F32(a), Buffer::F32(b)) => a[i] = b[j],
            (Buffer::F64(a), Buffer::F64(b)) => a[i] = b[j],
            (Buffer::C64(a), Buffer::C64(b)) => a[i] = b[j],
            (Buffer::C128(a), Buffer::C128(b)) => a[i] = b[j],
            _ => panic!("set_from requires matching dtypes"),
        }
    }

    /// Gathers `indices` into a new buffer.
    pub fn gather(&self, indices: &[usize]) -> Buffer {
        each_buffer!(self, v => indices.iter().map(|&i| v[i]).collect::<Vec<_>>().into())
    }

    /// NumPy `astype` with `casting='unsafe'` semantics.
    pub fn cast(&self, dtype: DType) -> Result<Buffer> {
        self.cast_cancellable(dtype, &CancellationToken::new())
    }

    pub fn cast_cancellable(&self, dtype: DType, token: &CancellationToken) -> Result<Buffer> {
        token.check()?;
        dtype.ensure_computable()?;
        if self.dtype() == dtype {
            return Ok(each_buffer!(self, v => map_elements(v, token, |x| x)?.into()));
        }
        // NumPy complex-to-real astype explicitly discards the imaginary part.
        // Integer conversions saturate at the conversion width; small integer
        // targets use the 32-bit conversion before narrowing.
        if matches!(self.dtype(), DType::Complex64 | DType::Complex128) {
            macro_rules! complex_cast {
                ($v:expr) => {
                    match dtype {
                        DType::Int8 => {
                            Some(Buffer::I8(map_elements($v, token, |z| z[0] as i32 as i8)?))
                        }
                        DType::Int16 => Some(Buffer::I16(map_elements($v, token, |z| {
                            z[0] as i32 as i16
                        })?)),
                        DType::Int32 => {
                            Some(Buffer::I32(map_elements($v, token, |z| z[0] as i32)?))
                        }
                        DType::Int64 => {
                            Some(Buffer::I64(map_elements($v, token, |z| z[0] as i64)?))
                        }
                        DType::UInt8 => {
                            Some(Buffer::U8(map_elements($v, token, |z| z[0] as i32 as u8)?))
                        }
                        DType::UInt16 => Some(Buffer::U16(map_elements($v, token, |z| {
                            z[0] as i32 as u16
                        })?)),
                        DType::UInt32 => {
                            Some(Buffer::U32(map_elements($v, token, |z| z[0] as u32)?))
                        }
                        DType::UInt64 => {
                            Some(Buffer::U64(map_elements($v, token, |z| z[0] as u64)?))
                        }
                        _ => None,
                    }
                };
            }
            let converted = match self {
                Buffer::C64(v) => complex_cast!(v),
                Buffer::C128(v) => complex_cast!(v),
                _ => unreachable!(),
            };
            if let Some(buffer) = converted {
                return Ok(buffer);
            }
        }
        macro_rules! conv {
            ($v:expr) => {
                match dtype {
                    DType::Bool => Buffer::Bool(map_elements($v, token, |x| x.truth())?),
                    DType::Int8 => Buffer::I8(map_elements($v, token, |x| to_i128(x) as i8)?),
                    DType::Int16 => Buffer::I16(map_elements($v, token, |x| to_i128(x) as i16)?),
                    DType::Int32 => Buffer::I32(map_elements($v, token, |x| to_i128(x) as i32)?),
                    DType::Int64 => Buffer::I64(map_elements($v, token, |x| to_i128(x) as i64)?),
                    DType::UInt8 => Buffer::U8(map_elements($v, token, |x| to_i128(x) as u8)?),
                    DType::UInt16 => Buffer::U16(map_elements($v, token, |x| to_i128(x) as u16)?),
                    DType::UInt32 => Buffer::U32(map_elements($v, token, |x| to_i128(x) as u32)?),
                    DType::UInt64 => Buffer::U64(map_elements($v, token, |x| to_i128(x) as u64)?),
                    DType::Float32 => Buffer::F32(map_elements($v, token, |x| to_f64(x) as f32)?),
                    DType::Float64 => Buffer::F64(map_elements($v, token, |x| to_f64(x))?),
                    DType::Complex64 => {
                        Buffer::C64(map_elements($v, token, |x| x.complex().map(|v| v as f32))?)
                    }
                    DType::Complex128 => Buffer::C128(map_elements($v, token, |x| x.complex())?),
                    _ => unreachable!(),
                }
            };
        }
        Ok(each_buffer!(self, v => conv!(v)))
    }
}

/// Maps into one output allocation and checks cancellation between chunks.
pub(crate) fn map_elements<T: Copy, U>(
    values: &[T],
    token: &CancellationToken,
    mut f: impl FnMut(T) -> U,
) -> Result<Vec<U>> {
    let mut out = Vec::with_capacity(values.len());
    for chunk in values.chunks(1 << 14) {
        token.check()?;
        out.extend(chunk.iter().copied().map(&mut f));
    }
    token.check()?;
    Ok(out)
}

trait Elem: Copy {
    fn complex(self) -> [f64; 2] {
        [self.f64(), 0.0]
    }
    fn truth(self) -> bool {
        self.f64() != 0.0
    }
    fn f64(self) -> f64;
    fn i128(self) -> i128;
}
macro_rules! elem_num {
    ($($t:ty),*) => {$(
        impl Elem for $t {
            fn f64(self) -> f64 { self as f64 }
            fn i128(self) -> i128 { self as i128 }
        }
    )*};
}
elem_num!(i8, i16, u8, u16, i32, i64, u32, u64, f32, f64);
impl Elem for bool {
    fn f64(self) -> f64 {
        self as u8 as f64
    }

    fn i128(self) -> i128 {
        self as i128
    }
}
impl Elem for [f32; 2] {
    fn f64(self) -> f64 {
        self[0] as f64
    }

    fn i128(self) -> i128 {
        self[0] as i128
    }

    fn complex(self) -> [f64; 2] {
        self.map(f64::from)
    }

    fn truth(self) -> bool {
        self[0] != 0.0 || self[1] != 0.0
    }
}
impl Elem for [f64; 2] {
    fn f64(self) -> f64 {
        self[0]
    }

    fn i128(self) -> i128 {
        self[0] as i128
    }

    fn complex(self) -> [f64; 2] {
        self
    }

    fn truth(self) -> bool {
        self[0] != 0.0 || self[1] != 0.0
    }
}
fn to_f64<T: Elem>(x: T) -> f64 {
    x.f64()
}
fn to_i128<T: Elem>(x: T) -> i128 {
    x.i128()
}

macro_rules! from_vec {
    ($($t:ty => $variant:ident),*) => {$(
        impl From<Vec<$t>> for Buffer {
            fn from(v: Vec<$t>) -> Self { Buffer::$variant(v) }
        }
    )*};
}
from_vec!(bool => Bool, i8 => I8, i16 => I16, u8 => U8, u16 => U16, i32 => I32, i64 => I64, u32 => U32, u64 => U64, f32 => F32, f64 => F64, [f32; 2] => C64, [f64; 2] => C128);

/// An owned contiguous C-order array charged to a budget.
#[derive(Debug)]
pub struct DenseArray {
    pub shape: Vec<usize>,
    pub buffer: Buffer,
    pub reservation: Reservation,
}

impl DenseArray {
    pub fn new(shape: Vec<usize>, buffer: Buffer, reservation: Reservation) -> Result<Self> {
        let size: usize = shape.iter().product();
        if size != buffer.len() {
            return Err(Error::new(
                ErrorCode::ShapeMismatch,
                format!(
                    "A buffer with {} elements cannot have shape {:?}.",
                    buffer.len(),
                    shape
                ),
            ));
        }
        Ok(Self {
            shape,
            buffer,
            reservation,
        })
    }

    pub fn dtype(&self) -> DType {
        self.buffer.dtype()
    }

    pub fn size(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_mapping_observes_cancellation_between_chunks() {
        let token = CancellationToken::new();
        let values = vec![0.0; 1 << 16];
        let mut visited = 0;
        let result = map_elements(&values, &token, |x| {
            visited += 1;
            token.cancel();
            x
        });
        assert_eq!(result.unwrap_err().code, ErrorCode::Cancelled);
        assert_eq!(visited, 1 << 14);
    }

    #[test]
    fn casts_follow_unsafe_casting() {
        let b = Buffer::F64(vec![1.7, -2.5, 0.0]);
        assert_eq!(b.cast(DType::Int32).unwrap(), Buffer::I32(vec![1, -2, 0]));
        assert_eq!(
            b.cast(DType::Bool).unwrap(),
            Buffer::Bool(vec![true, true, false])
        );
        let big = Buffer::I64(vec![i64::MAX]);
        assert_eq!(big.cast(DType::Int64).unwrap(), big);
    }

    #[test]
    fn small_integer_storage_and_casts() {
        for (dtype, expected) in [
            (DType::Int8, Buffer::I8(vec![-1, 0, 1])),
            (DType::Int16, Buffer::I16(vec![-1, 0, 1])),
            (DType::UInt8, Buffer::U8(vec![255, 0, 1])),
            (DType::UInt16, Buffer::U16(vec![65535, 0, 1])),
        ] {
            // NumPy 2.5.4: np.array([-1, 0, 1], dtype='int64').astype(dtype).
            let cast = Buffer::I64(vec![-1, 0, 1]).cast(dtype).unwrap();
            assert_eq!(cast, expected);
            assert_eq!(cast.byte_len(), 3 * dtype.item_size() as u64);
            assert_eq!(cast.gather(&[2, 0]).get(0), cast.get(2));
            let mut filled = Buffer::filled(dtype, 3, Scalar::Int(0)).unwrap();
            filled.set_from(1, &cast, 0);
            assert_eq!(filled.get(1), cast.get(0));
        }
    }

    #[test]
    fn dense_array_validates_shape() {
        let err = DenseArray::new(vec![2, 2], Buffer::I32(vec![1, 2, 3]), Reservation::none());
        assert_eq!(err.unwrap_err().code, ErrorCode::ShapeMismatch);
    }
}
