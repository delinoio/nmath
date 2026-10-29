//! Typed contiguous buffers and scalar values.

use crate::{
    dtype::{DType, ScalarKind},
    error::{Error, ErrorCode, Result},
    memory::Reservation,
};

/// A JS-facing scalar. 64-bit integers are carried exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
}

impl Scalar {
    pub fn kind(self) -> ScalarKind {
        match self {
            Scalar::Bool(_) => ScalarKind::Bool,
            Scalar::Int(_) | Scalar::UInt(_) => ScalarKind::Int,
            Scalar::Float(_) => ScalarKind::Float,
        }
    }

    /// Dtype of a scalar that stands alone (both operands are scalars).
    pub fn default_dtype(self) -> DType {
        match self {
            Scalar::Bool(_) => DType::Bool,
            Scalar::Int(_) => DType::Int64,
            Scalar::UInt(_) => DType::UInt64,
            Scalar::Float(_) => DType::Float64,
        }
    }

    pub fn as_f64(self) -> f64 {
        match self {
            Scalar::Bool(b) => b as u8 as f64,
            Scalar::Int(i) => i as f64,
            Scalar::UInt(u) => u as f64,
            Scalar::Float(f) => f,
        }
    }

    fn as_i128(self) -> i128 {
        match self {
            Scalar::Bool(b) => b as i128,
            Scalar::Int(i) => i as i128,
            Scalar::UInt(u) => u as i128,
            Scalar::Float(f) => f as i128,
        }
    }
}

/// Contiguous typed storage for computable dtypes.
#[derive(Clone, Debug, PartialEq)]
pub enum Buffer {
    Bool(Vec<bool>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

macro_rules! each_buffer {
    ($buf:expr, $v:ident => $body:expr) => {
        match $buf {
            Buffer::Bool($v) => $body,
            Buffer::I32($v) => $body,
            Buffer::I64($v) => $body,
            Buffer::U32($v) => $body,
            Buffer::U64($v) => $body,
            Buffer::F32($v) => $body,
            Buffer::F64($v) => $body,
        }
    };
}

impl Buffer {
    pub fn dtype(&self) -> DType {
        match self {
            Buffer::Bool(_) => DType::Bool,
            Buffer::I32(_) => DType::Int32,
            Buffer::I64(_) => DType::Int64,
            Buffer::U32(_) => DType::UInt32,
            Buffer::U64(_) => DType::UInt64,
            Buffer::F32(_) => DType::Float32,
            Buffer::F64(_) => DType::Float64,
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
        Ok(match dtype.ensure_computable()? {
            DType::Bool => Buffer::Bool(vec![value.as_f64() != 0.0; len]),
            DType::Int32 => Buffer::I32(vec![value.as_i128() as i32; len]),
            DType::Int64 => Buffer::I64(vec![value.as_i128() as i64; len]),
            DType::UInt32 => Buffer::U32(vec![value.as_i128() as u32; len]),
            DType::UInt64 => Buffer::U64(vec![value.as_i128() as u64; len]),
            DType::Float32 => Buffer::F32(vec![value.as_f64() as f32; len]),
            DType::Float64 => Buffer::F64(vec![value.as_f64(); len]),
            _ => unreachable!(),
        })
    }

    pub fn get(&self, i: usize) -> Scalar {
        match self {
            Buffer::Bool(v) => Scalar::Bool(v[i]),
            Buffer::I32(v) => Scalar::Int(v[i] as i64),
            Buffer::I64(v) => Scalar::Int(v[i]),
            Buffer::U32(v) => Scalar::UInt(v[i] as u64),
            Buffer::U64(v) => Scalar::UInt(v[i]),
            Buffer::F32(v) => Scalar::Float(v[i] as f64),
            Buffer::F64(v) => Scalar::Float(v[i]),
        }
    }

    /// Copies one element from `src[j]` into `self[i]`; both buffers have the
    /// same dtype.
    pub fn set_from(&mut self, i: usize, src: &Buffer, j: usize) {
        match (self, src) {
            (Buffer::Bool(a), Buffer::Bool(b)) => a[i] = b[j],
            (Buffer::I32(a), Buffer::I32(b)) => a[i] = b[j],
            (Buffer::I64(a), Buffer::I64(b)) => a[i] = b[j],
            (Buffer::U32(a), Buffer::U32(b)) => a[i] = b[j],
            (Buffer::U64(a), Buffer::U64(b)) => a[i] = b[j],
            (Buffer::F32(a), Buffer::F32(b)) => a[i] = b[j],
            (Buffer::F64(a), Buffer::F64(b)) => a[i] = b[j],
            _ => panic!("set_from requires matching dtypes"),
        }
    }

    /// Gathers `indices` into a new buffer.
    pub fn gather(&self, indices: &[usize]) -> Buffer {
        each_buffer!(self, v => indices.iter().map(|&i| v[i]).collect::<Vec<_>>().into())
    }

    /// NumPy `astype` with `casting='unsafe'` semantics.
    pub fn cast(&self, dtype: DType) -> Result<Buffer> {
        dtype.ensure_computable()?;
        if self.dtype() == dtype {
            return Ok(self.clone());
        }
        macro_rules! conv {
            ($v:expr) => {
                match dtype {
                    DType::Bool => Buffer::Bool($v.iter().map(|x| to_f64(*x) != 0.0).collect()),
                    DType::Int32 => Buffer::I32($v.iter().map(|x| to_i128(*x) as i32).collect()),
                    DType::Int64 => Buffer::I64($v.iter().map(|x| to_i128(*x) as i64).collect()),
                    DType::UInt32 => Buffer::U32($v.iter().map(|x| to_i128(*x) as u32).collect()),
                    DType::UInt64 => Buffer::U64($v.iter().map(|x| to_i128(*x) as u64).collect()),
                    DType::Float32 => Buffer::F32($v.iter().map(|x| to_f64(*x) as f32).collect()),
                    DType::Float64 => Buffer::F64($v.iter().map(|x| to_f64(*x)).collect()),
                    _ => unreachable!(),
                }
            };
        }
        Ok(each_buffer!(self, v => conv!(v)))
    }
}

trait Elem: Copy {
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
elem_num!(i32, i64, u32, u64, f32, f64);
impl Elem for bool {
    fn f64(self) -> f64 {
        self as u8 as f64
    }

    fn i128(self) -> i128 {
        self as i128
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
from_vec!(bool => Bool, i32 => I32, i64 => I64, u32 => U32, u64 => U64, f32 => F32, f64 => F64);

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
    fn dense_array_validates_shape() {
        let err = DenseArray::new(vec![2, 2], Buffer::I32(vec![1, 2, 3]), Reservation::none());
        assert_eq!(err.unwrap_err().code, ErrorCode::ShapeMismatch);
    }
}
