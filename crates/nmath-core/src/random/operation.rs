// SPDX-License-Identifier: Apache-2.0
use super::{Pcg64, distributions::IntegerBuffer};
use crate::{
    buffer::{Buffer, DenseArray},
    dtype::DType,
    error::{Error, ErrorCode, Result},
    memory::MemoryBudget,
    sync::CancellationToken,
};

/// Scalar-parameter Generator distributions. Shape is owned by the expression.
#[derive(Clone, Debug)]
pub enum RandomOperation {
    Random(DType),
    Integers {
        low: i128,
        high: i128,
        endpoint: bool,
        dtype: DType,
    },
    Uniform {
        low: f64,
        high: f64,
    },
    StandardNormal(DType),
    Normal {
        loc: f64,
        scale: f64,
    },
    Exponential {
        scale: f64,
    },
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorCode::DTypeUnsupported, message)
}

impl RandomOperation {
    pub fn dtype(&self) -> DType {
        match self {
            Self::Random(dtype) | Self::StandardNormal(dtype) | Self::Integers { dtype, .. } => {
                *dtype
            }
            _ => DType::Float64,
        }
    }

    pub fn validate(&self) -> Result<()> {
        match *self {
            Self::Random(DType::Float32 | DType::Float64)
            | Self::StandardNormal(DType::Float32 | DType::Float64) => Ok(()),
            Self::Random(_) | Self::StandardNormal(_) => {
                Err(invalid("Random requires float32 or float64."))
            }
            Self::Integers {
                low,
                high,
                endpoint,
                dtype,
            } => {
                let (min, max, _) = integer_limits(dtype)?;
                let last = if endpoint {
                    Some(high)
                } else {
                    high.checked_sub(1)
                };
                if low < min || last.is_none_or(|last| last > max || last < low) {
                    return Err(invalid(
                        "Integer bounds must be ordered and fit the selected dtype.",
                    ));
                }
                Ok(())
            }
            Self::Uniform { low, high } => {
                if !(high - low).is_finite() || (high - low).is_sign_negative() {
                    Err(invalid(
                        "Uniform bounds must have a finite nonnegative difference.",
                    ))
                } else {
                    Ok(())
                }
            }
            Self::Normal { scale, .. } | Self::Exponential { scale } => {
                // NumPy accepts NaN and positive infinity, but rejects negative
                // zero.
                if scale.is_sign_negative() && !scale.is_nan() {
                    Err(invalid("Scale must be nonnegative."))
                } else {
                    Ok(())
                }
            }
        }
    }

    pub fn generate(
        &self,
        rng: &mut Pcg64,
        shape: &[usize],
        budget: &MemoryBudget,
        token: &CancellationToken,
    ) -> Result<DenseArray> {
        self.validate()?;
        let len = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| invalid("Random shape exceeds addressable storage."))?;
        let bytes = len
            .checked_mul(self.dtype().item_size())
            .ok_or_else(|| invalid("Random output exceeds addressable storage."))?;
        let reservation = budget.reserve(bytes as u64)?;
        let mut local = *rng;
        let mut integers = IntegerBuffer::default();
        macro_rules! collect {
            ($variant:ident, $value:expr) => {{
                let mut values = Vec::with_capacity(len);
                for i in 0..len {
                    if i % 1024 == 0 {
                        token.check()?;
                    }
                    values.push($value);
                }
                Buffer::$variant(values)
            }};
        }
        let buffer = match *self {
            Self::Random(DType::Float32) => collect!(F32, local.next_f32()),
            Self::Random(_) => collect!(F64, local.next_f64()),
            Self::Uniform { low, high } => collect!(F64, affine(high - low, local.next_f64(), low)),
            Self::StandardNormal(DType::Float32) => collect!(F32, local.standard_normal_f32()),
            Self::StandardNormal(_) => collect!(F64, local.standard_normal()),
            Self::Normal { loc, scale } => {
                collect!(F64, affine(scale, local.standard_normal(), loc))
            }
            Self::Exponential { scale } => collect!(F64, scale * local.standard_exponential()),
            Self::Integers {
                low,
                high,
                endpoint,
                dtype,
            } => {
                let (_, _, width) = integer_limits(dtype)?;
                let range = (high - low - i128::from(!endpoint)) as u64;
                macro_rules! ints {
                    ($variant:ident, $ty:ty) => {
                        collect!(
                            $variant,
                            (low + integers.bounded(&mut local, range, width) as i128) as $ty
                        )
                    };
                }
                match dtype {
                    DType::Bool => collect!(
                        Bool,
                        (low + integers.bounded(&mut local, range, width) as i128) != 0
                    ),
                    DType::Int8 => ints!(I8, i8),
                    DType::Int16 => ints!(I16, i16),
                    DType::Int32 => ints!(I32, i32),
                    DType::Int64 => ints!(I64, i64),
                    DType::UInt8 => ints!(U8, u8),
                    DType::UInt16 => ints!(U16, u16),
                    DType::UInt32 => ints!(U32, u32),
                    DType::UInt64 => ints!(U64, u64),
                    _ => unreachable!(),
                }
            }
        };
        token.check()?;
        let result = DenseArray::new(shape.to_vec(), buffer, reservation)?;
        *rng = local;
        Ok(result)
    }
}

fn integer_limits(dtype: DType) -> Result<(i128, i128, u32)> {
    Ok(match dtype {
        DType::Bool => (0, 1, 1),
        DType::Int8 => (i8::MIN as i128, i8::MAX as i128, 8),
        DType::Int16 => (i16::MIN as i128, i16::MAX as i128, 16),
        DType::Int32 => (i32::MIN as i128, i32::MAX as i128, 32),
        DType::Int64 => (i64::MIN as i128, i64::MAX as i128, 64),
        DType::UInt8 => (0, u8::MAX as i128, 8),
        DType::UInt16 => (0, u16::MAX as i128, 16),
        DType::UInt32 => (0, u32::MAX as i128, 32),
        DType::UInt64 => (0, u64::MAX as i128, 64),
        _ => return Err(invalid("Integers requires a boolean or integer dtype.")),
    })
}

// Preserve the contracted multiply-add used by the pinned NumPy oracle.
// Cross-platform libm behavior still requires platform-specific acceptance.
fn affine(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}
