//! NumPy elementwise math ufuncs on the CPU.
//!
//! Result dtypes follow the NumPy 2.5.4 loop tables. Floating-point
//! transcendental functions evaluate in float64 and round once to the result
//! dtype. Functions whose NumPy loops are exact or use a fixed formula
//! (`deg2rad`, `rad2deg`, `spacing`, `square`, `reciprocal` and the rounding
//! functions) use the result dtype's arithmetic so results match NumPy
//! exactly.

use crate::{
    buffer::{Buffer, DenseArray, Float16, checked_byte_len, map_elements},
    dtype::{DType, DTypeKind},
    error::{Error, ErrorCode, Result},
    memory::{MemoryBudget, Reservation},
    sync::CancellationToken,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MathOp {
    Sqrt,
    Cbrt,
    Exp,
    Exp2,
    Expm1,
    Log,
    Log2,
    Log10,
    Log1p,
    Sin,
    Cos,
    Tan,
    Arcsin,
    Arccos,
    Arctan,
    Sinh,
    Cosh,
    Tanh,
    Arcsinh,
    Arccosh,
    Arctanh,
    Deg2rad,
    Rad2deg,
    Fabs,
    Rint,
    Spacing,
    Absolute,
    Square,
    Reciprocal,
    Sign,
    Floor,
    Ceil,
    Trunc,
    Conjugate,
    Invert,
    Isnan,
    Isinf,
    Isfinite,
    Signbit,
    LogicalNot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    /// Real-valued floating-point result from a bool, integer or float input.
    Float,
    /// Same-dtype result computed with the input's arithmetic.
    Exact,
    /// Bool result.
    Predicate,
}

impl MathOp {
    pub const ALL: [MathOp; 40] = [
        MathOp::Sqrt,
        MathOp::Cbrt,
        MathOp::Exp,
        MathOp::Exp2,
        MathOp::Expm1,
        MathOp::Log,
        MathOp::Log2,
        MathOp::Log10,
        MathOp::Log1p,
        MathOp::Sin,
        MathOp::Cos,
        MathOp::Tan,
        MathOp::Arcsin,
        MathOp::Arccos,
        MathOp::Arctan,
        MathOp::Sinh,
        MathOp::Cosh,
        MathOp::Tanh,
        MathOp::Arcsinh,
        MathOp::Arccosh,
        MathOp::Arctanh,
        MathOp::Deg2rad,
        MathOp::Rad2deg,
        MathOp::Fabs,
        MathOp::Rint,
        MathOp::Spacing,
        MathOp::Absolute,
        MathOp::Square,
        MathOp::Reciprocal,
        MathOp::Sign,
        MathOp::Floor,
        MathOp::Ceil,
        MathOp::Trunc,
        MathOp::Conjugate,
        MathOp::Invert,
        MathOp::Isnan,
        MathOp::Isinf,
        MathOp::Isfinite,
        MathOp::Signbit,
        MathOp::LogicalNot,
    ];

    /// The NumPy ufunc name.
    pub fn id(self) -> &'static str {
        match self {
            MathOp::Sqrt => "sqrt",
            MathOp::Cbrt => "cbrt",
            MathOp::Exp => "exp",
            MathOp::Exp2 => "exp2",
            MathOp::Expm1 => "expm1",
            MathOp::Log => "log",
            MathOp::Log2 => "log2",
            MathOp::Log10 => "log10",
            MathOp::Log1p => "log1p",
            MathOp::Sin => "sin",
            MathOp::Cos => "cos",
            MathOp::Tan => "tan",
            MathOp::Arcsin => "arcsin",
            MathOp::Arccos => "arccos",
            MathOp::Arctan => "arctan",
            MathOp::Sinh => "sinh",
            MathOp::Cosh => "cosh",
            MathOp::Tanh => "tanh",
            MathOp::Arcsinh => "arcsinh",
            MathOp::Arccosh => "arccosh",
            MathOp::Arctanh => "arctanh",
            MathOp::Deg2rad => "deg2rad",
            MathOp::Rad2deg => "rad2deg",
            MathOp::Fabs => "fabs",
            MathOp::Rint => "rint",
            MathOp::Spacing => "spacing",
            MathOp::Absolute => "absolute",
            MathOp::Square => "square",
            MathOp::Reciprocal => "reciprocal",
            MathOp::Sign => "sign",
            MathOp::Floor => "floor",
            MathOp::Ceil => "ceil",
            MathOp::Trunc => "trunc",
            MathOp::Conjugate => "conjugate",
            MathOp::Invert => "invert",
            MathOp::Isnan => "isnan",
            MathOp::Isinf => "isinf",
            MathOp::Isfinite => "isfinite",
            MathOp::Signbit => "signbit",
            MathOp::LogicalNot => "logical_not",
        }
    }

    pub fn from_id(id: &str) -> Result<Self> {
        MathOp::ALL
            .into_iter()
            .find(|op| op.id() == id)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::IndexInvalid,
                    format!("Unknown math function '{id}'."),
                )
            })
    }

    fn family(self) -> Family {
        match self {
            MathOp::Absolute
            | MathOp::Square
            | MathOp::Reciprocal
            | MathOp::Sign
            | MathOp::Floor
            | MathOp::Ceil
            | MathOp::Trunc
            | MathOp::Conjugate
            | MathOp::Invert => Family::Exact,
            MathOp::Isnan
            | MathOp::Isinf
            | MathOp::Isfinite
            | MathOp::Signbit
            | MathOp::LogicalNot => Family::Predicate,
            _ => Family::Float,
        }
    }

    /// Result dtype for `input`, or the reason the combination is rejected.
    pub fn result_dtype(self, input: DType) -> Result<DType> {
        let input = input.ensure_computable()?;
        let kind = input.kind();
        if kind == DTypeKind::Other {
            return Err(unsupported(format!(
                "{} does not support string or temporal arrays. Use a numeric or bool array.",
                self.id()
            )));
        }
        let complex = kind == DTypeKind::Complex;
        let complex_target = || {
            unsupported(format!(
                "{} does not support complex arrays yet. Compute the real and imaginary parts \
                 explicitly.",
                self.id()
            ))
        };
        match self {
            MathOp::Isnan | MathOp::Isinf | MathOp::Isfinite | MathOp::LogicalNot => {
                Ok(DType::Bool)
            }
            MathOp::Signbit if complex => Err(unsupported(
                "signbit is not defined for complex arrays.".to_string(),
            )),
            MathOp::Signbit => Ok(DType::Bool),
            MathOp::Invert if matches!(kind, DTypeKind::Bool | DTypeKind::Int) => Ok(input),
            MathOp::Invert => Err(unsupported(
                "invert requires a bool or integer array.".to_string(),
            )),
            MathOp::Conjugate if input == DType::Bool => Ok(DType::Int8),
            MathOp::Conjugate => Ok(input),
            MathOp::Floor | MathOp::Ceil | MathOp::Trunc if complex => Err(unsupported(format!(
                "{} is not defined for complex arrays.",
                self.id()
            ))),
            MathOp::Floor | MathOp::Ceil | MathOp::Trunc => Ok(input),
            _ if complex => Err(complex_target()),
            MathOp::Sign if input == DType::Bool => Err(unsupported(
                "sign is not defined for bool arrays. Cast to an integer dtype first.".to_string(),
            )),
            MathOp::Sign | MathOp::Absolute => Ok(input),
            MathOp::Square if input == DType::Bool => Ok(DType::Int8),
            MathOp::Square => Ok(input),
            // NumPy's integer reciprocal of zero is platform-dependent, so
            // only floating-point inputs are supported until that behavior is
            // decided explicitly.
            MathOp::Reciprocal if kind != DTypeKind::Float => Err(unsupported(
                "reciprocal supports only floating-point arrays. Cast to a float dtype first."
                    .to_string(),
            )),
            MathOp::Reciprocal => Ok(input),
            _ => Ok(float_dtype(input)),
        }
    }
}

fn unsupported(message: String) -> Error {
    Error::new(ErrorCode::DTypeUnsupported, message)
}

/// NumPy's smallest floating-point loop that can hold `input` exactly.
fn float_dtype(input: DType) -> DType {
    match input {
        DType::Bool | DType::Int8 | DType::UInt8 | DType::Float16 => DType::Float16,
        DType::Int16 | DType::UInt16 | DType::Float32 => DType::Float32,
        _ => DType::Float64,
    }
}

fn real(op: MathOp, x: f64) -> f64 {
    match op {
        MathOp::Sqrt => x.sqrt(),
        MathOp::Cbrt => x.cbrt(),
        MathOp::Exp => x.exp(),
        MathOp::Exp2 => x.exp2(),
        MathOp::Expm1 => x.exp_m1(),
        MathOp::Log => x.ln(),
        MathOp::Log2 => x.log2(),
        MathOp::Log10 => x.log10(),
        MathOp::Log1p => x.ln_1p(),
        MathOp::Sin => x.sin(),
        MathOp::Cos => x.cos(),
        MathOp::Tan => x.tan(),
        MathOp::Arcsin => x.asin(),
        MathOp::Arccos => x.acos(),
        MathOp::Arctan => x.atan(),
        MathOp::Sinh => x.sinh(),
        MathOp::Cosh => x.cosh(),
        MathOp::Tanh => x.tanh(),
        MathOp::Arcsinh => x.asinh(),
        MathOp::Arccosh => x.acosh(),
        MathOp::Arctanh => x.atanh(),
        MathOp::Fabs => x.abs(),
        MathOp::Rint => x.round_ties_even(),
        _ => unreachable!("handled by a dtype-specific path"),
    }
}

/// NumPy `spacing`: the distance to the next value away from zero, negative
/// for negative inputs. Negative zero counts as positive.
macro_rules! spacing {
    ($x:expr) => {{
        let x = $x;
        if !x.is_finite() {
            // NaN for both infinities and NaN.
            x - x
        } else {
            let a = x.abs();
            let s = a.next_up() - a;
            if x < 0.0 { -s } else { s }
        }
    }};
}

/// NumPy's half `spacing` differs from the float32/float64 loops: a negative
/// input returns the positive distance to the next value toward zero.
fn spacing16(x: Float16) -> Float16 {
    if !x.is_finite() {
        return Float16::NAN;
    }
    let bits = x.to_bits() & 0x7fff;
    let a = Float16::from_bits(bits);
    let other = if x.is_sign_negative() && bits != 0 {
        Float16::from_bits(bits - 1)
    } else {
        Float16::from_bits(bits + 1)
    };
    Float16::from_f32((other.to_f32() - a.to_f32()).abs())
}

fn float_family(
    op: MathOp,
    input: &Buffer,
    dtype: DType,
    token: &CancellationToken,
) -> Result<Buffer> {
    match op {
        MathOp::Deg2rad | MathOp::Rad2deg | MathOp::Spacing => {
            let native = input.cast_cancellable(dtype, token)?;
            let deg = op == MathOp::Deg2rad;
            Ok(match &native {
                Buffer::F16(v) => match op {
                    MathOp::Spacing => map_elements(v, token, spacing16)?.into(),
                    _ => map_elements(v, token, |x| {
                        let f = std::f32::consts::PI / 180.0;
                        let y = if deg {
                            x.to_f32() * f
                        } else {
                            x.to_f32() * (180.0 / std::f32::consts::PI)
                        };
                        Float16::from_f32(y)
                    })?
                    .into(),
                },
                Buffer::F32(v) => match op {
                    MathOp::Spacing => map_elements(v, token, |x| spacing!(x))?.into(),
                    _ if deg => {
                        map_elements(v, token, |x| x * (std::f32::consts::PI / 180.0))?.into()
                    }
                    _ => map_elements(v, token, |x| x * (180.0 / std::f32::consts::PI))?.into(),
                },
                Buffer::F64(v) => match op {
                    MathOp::Spacing => map_elements(v, token, |x| spacing!(x))?.into(),
                    _ if deg => {
                        map_elements(v, token, |x| x * (std::f64::consts::PI / 180.0))?.into()
                    }
                    _ => map_elements(v, token, |x| x * (180.0 / std::f64::consts::PI))?.into(),
                },
                _ => unreachable!("float results use a float buffer"),
            })
        }
        _ => {
            let wide = input.cast_cancellable(DType::Float64, token)?;
            let Buffer::F64(v) = &wide else {
                unreachable!("cast to float64")
            };
            Ok(match dtype {
                DType::Float16 => {
                    map_elements(v, token, |x| Float16::from_f64(real(op, x)))?.into()
                }
                DType::Float32 => map_elements(v, token, |x| real(op, x) as f32)?.into(),
                _ => map_elements(v, token, |x| real(op, x))?.into(),
            })
        }
    }
}

macro_rules! each_int {
    ($buf:expr, $v:ident => $body:expr, else $other:expr) => {
        match $buf {
            Buffer::I8($v) => $body,
            Buffer::I16($v) => $body,
            Buffer::I32($v) => $body,
            Buffer::I64($v) => $body,
            Buffer::U8($v) => $body,
            Buffer::U16($v) => $body,
            Buffer::U32($v) => $body,
            Buffer::U64($v) => $body,
            _ => $other,
        }
    };
}

fn sign_float<T: PartialOrd + Copy>(x: T, zero: T, one: T, minus: T) -> T {
    if x > zero {
        one
    } else if x < zero {
        minus
    } else if x == zero {
        zero
    } else {
        x
    }
}

fn exact_family(
    op: MathOp,
    input: &Buffer,
    dtype: DType,
    token: &CancellationToken,
) -> Result<Buffer> {
    let input = input.cast_cancellable(dtype, token)?;
    // NumPy's half loops compute in float32 and round once.
    let f16 = |v: &[Float16], f: fn(f32) -> f32| -> Result<Buffer> {
        Ok(map_elements(v, token, |x| Float16::from_f32(f(x.to_f32())))?.into())
    };
    Ok(match op {
        MathOp::Floor | MathOp::Ceil | MathOp::Trunc => {
            let (f32f, f64f): (fn(f32) -> f32, fn(f64) -> f64) = match op {
                MathOp::Floor => (f32::floor, f64::floor),
                MathOp::Ceil => (f32::ceil, f64::ceil),
                _ => (f32::trunc, f64::trunc),
            };
            match &input {
                Buffer::F16(v) => f16(v, f32f)?,
                Buffer::F32(v) => map_elements(v, token, f32f)?.into(),
                Buffer::F64(v) => map_elements(v, token, f64f)?.into(),
                // Integers and bools are already integral.
                _ => input,
            }
        }
        MathOp::Absolute => match &input {
            Buffer::Bool(_) => input,
            Buffer::F16(v) => {
                map_elements(v, token, |x| Float16::from_bits(x.to_bits() & 0x7fff))?.into()
            }
            Buffer::F32(v) => map_elements(v, token, f32::abs)?.into(),
            Buffer::F64(v) => map_elements(v, token, f64::abs)?.into(),
            Buffer::I8(v) => map_elements(v, token, i8::wrapping_abs)?.into(),
            Buffer::I16(v) => map_elements(v, token, i16::wrapping_abs)?.into(),
            Buffer::I32(v) => map_elements(v, token, i32::wrapping_abs)?.into(),
            Buffer::I64(v) => map_elements(v, token, i64::wrapping_abs)?.into(),
            _ => input,
        },
        MathOp::Square => match &input {
            Buffer::F16(v) => f16(v, |x| x * x)?,
            Buffer::F32(v) => map_elements(v, token, |x| x * x)?.into(),
            Buffer::F64(v) => map_elements(v, token, |x| x * x)?.into(),
            other => each_int!(other, v => map_elements(v, token, |x| x.wrapping_mul(x))?.into(),
                else unreachable!("bool squares use int8")),
        },
        MathOp::Reciprocal => match &input {
            Buffer::F16(v) => f16(v, |x| 1.0 / x)?,
            Buffer::F32(v) => map_elements(v, token, |x| 1.0 / x)?.into(),
            Buffer::F64(v) => map_elements(v, token, |x| 1.0 / x)?.into(),
            _ => unreachable!("reciprocal accepts only floats"),
        },
        MathOp::Sign => match &input {
            Buffer::F16(v) => f16(v, |x| sign_float(x, 0.0, 1.0, -1.0))?,
            Buffer::F32(v) => map_elements(v, token, |x| sign_float(x, 0.0, 1.0, -1.0))?.into(),
            Buffer::F64(v) => map_elements(v, token, |x| sign_float(x, 0.0, 1.0, -1.0))?.into(),
            Buffer::I8(v) => map_elements(v, token, i8::signum)?.into(),
            Buffer::I16(v) => map_elements(v, token, i16::signum)?.into(),
            Buffer::I32(v) => map_elements(v, token, i32::signum)?.into(),
            Buffer::I64(v) => map_elements(v, token, i64::signum)?.into(),
            other => each_int!(other, v => map_elements(v, token, |x| x.min(1))?.into(),
                else unreachable!("bool sign is rejected")),
        },
        MathOp::Invert => match &input {
            Buffer::Bool(v) => map_elements(v, token, |x| !x)?.into(),
            other => each_int!(other, v => map_elements(v, token, |x| !x)?.into(),
                else unreachable!("invert accepts only bool and integer arrays")),
        },
        MathOp::Conjugate => match &input {
            Buffer::C64(v) => map_elements(v, token, |[re, im]| [re, -im])?.into(),
            Buffer::C128(v) => map_elements(v, token, |[re, im]| [re, -im])?.into(),
            _ => input,
        },
        _ => unreachable!("not an exact-family function"),
    })
}

fn predicate(op: MathOp, input: &Buffer, token: &CancellationToken) -> Result<Vec<bool>> {
    macro_rules! real_float {
        ($v:expr, $zero:expr) => {
            map_elements($v, token, |x| match op {
                MathOp::Isnan => x.is_nan(),
                MathOp::Isinf => x.is_infinite(),
                MathOp::Isfinite => x.is_finite(),
                MathOp::Signbit => x.is_sign_negative(),
                _ => x == $zero,
            })
        };
    }
    macro_rules! complex {
        ($v:expr) => {
            map_elements($v, token, |[re, im]| match op {
                MathOp::Isnan => re.is_nan() || im.is_nan(),
                MathOp::Isinf => re.is_infinite() || im.is_infinite(),
                MathOp::Isfinite => re.is_finite() && im.is_finite(),
                _ => re == 0.0 && im == 0.0,
            })
        };
    }
    match input {
        Buffer::F16(v) => real_float!(v, Float16::ZERO),
        Buffer::F32(v) => real_float!(v, 0.0),
        Buffer::F64(v) => real_float!(v, 0.0),
        Buffer::C64(v) => complex!(v),
        Buffer::C128(v) => complex!(v),
        Buffer::Bool(v) => map_elements(v, token, |x| match op {
            MathOp::Isfinite => true,
            MathOp::LogicalNot => !x,
            _ => false,
        }),
        Buffer::I8(v) if op == MathOp::Signbit => map_elements(v, token, |x| x < 0),
        Buffer::I16(v) if op == MathOp::Signbit => map_elements(v, token, |x| x < 0),
        Buffer::I32(v) if op == MathOp::Signbit => map_elements(v, token, |x| x < 0),
        Buffer::I64(v) if op == MathOp::Signbit => map_elements(v, token, |x| x < 0),
        other => match op {
            MathOp::Isfinite => Ok(vec![true; other.len()]),
            MathOp::LogicalNot => {
                let Buffer::Bool(nonzero) = other.cast_cancellable(DType::Bool, token)? else {
                    unreachable!("cast to bool")
                };
                map_elements(&nonzero, token, |x| !x)
            }
            // isnan, isinf and unsigned signbit are false for integers.
            _ => Ok(vec![false; other.len()]),
        },
    }
}

/// Applies `op` elementwise. Temporary casts are charged to `budget` while
/// they live.
pub fn execute(
    op: MathOp,
    input: &DenseArray,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    token.check()?;
    let dtype = op.result_dtype(input.dtype())?;
    let reservation = budget.reserve(checked_byte_len(input.size(), dtype)? as u64)?;
    // Each path materializes at most one converted copy of the input, at most
    // eight bytes per element.
    let _scratch: Reservation =
        budget.reserve(checked_byte_len(input.size(), DType::Float64)? as u64)?;
    let buffer = match op.family() {
        Family::Float => float_family(op, &input.buffer, dtype, token)?,
        Family::Exact => exact_family(op, &input.buffer, dtype, token)?,
        Family::Predicate => predicate(op, &input.buffer, token)?.into(),
    };
    DenseArray::new(input.shape.clone(), buffer, reservation)
}
