//! Dtype identifiers and NumPy 2.x promotion rules.

use crate::error::{Error, ErrorCode, Result};

/// Every dtype ID in the project contract, including storage metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    Complex64,
    Complex128,
    Str(usize),
    Datetime64(TimeUnit),
    Timedelta64(TimeUnit),
}

/// Kind ordering used by NumPy promotion: bool < unsigned/signed integer <
/// float.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DTypeKind {
    Bool,
    Int,
    Float,
    Complex,
    Other,
}

impl DType {
    pub const ALL: [DType; 17] = [
        DType::Bool,
        DType::Int8,
        DType::Int16,
        DType::Int32,
        DType::Int64,
        DType::UInt8,
        DType::UInt16,
        DType::UInt32,
        DType::UInt64,
        DType::Float16,
        DType::Float32,
        DType::Float64,
        DType::Complex64,
        DType::Complex128,
        DType::Str(1),
        DType::Datetime64(TimeUnit::NANOSECOND),
        DType::Timedelta64(TimeUnit::NANOSECOND),
    ];

    pub fn id(self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::UInt8 => "uint8",
            DType::UInt16 => "uint16",
            DType::UInt32 => "uint32",
            DType::UInt64 => "uint64",
            DType::Float16 => "float16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Complex64 => "complex64",
            DType::Complex128 => "complex128",
            DType::Str(_) => "str",
            DType::Datetime64(_) => "datetime64",
            DType::Timedelta64(_) => "timedelta64",
        }
    }

    pub fn from_id(id: &str) -> Result<DType> {
        if let Some(width) = id.strip_prefix("str[").and_then(|s| s.strip_suffix(']')) {
            let width = width
                .parse::<usize>()
                .ok()
                .filter(|n| *n > 0 && *n <= usize::MAX / 4)
                .ok_or_else(unsupported)?;
            return Ok(DType::Str(width));
        }
        for (prefix, datetime) in [("datetime64[", true), ("timedelta64[", false)] {
            if let Some(unit) = id.strip_prefix(prefix).and_then(|s| s.strip_suffix(']')) {
                let unit = TimeUnit::parse(unit)?;
                return Ok(if datetime {
                    DType::Datetime64(unit)
                } else {
                    DType::Timedelta64(unit)
                });
            }
        }
        DType::ALL
            .into_iter()
            .find(|d| d.id() == id)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::DTypeUnsupported,
                    format!("Unknown dtype '{id}'."),
                )
            })
    }

    /// Valid CPU storage metadata. Operators validate their own supported
    /// loops.
    pub fn is_computable(self) -> bool {
        match self {
            Self::Str(n) => n > 0 && n <= usize::MAX / 4,
            Self::Datetime64(u) | Self::Timedelta64(u) => u.step > 0 && u.step <= i32::MAX as u32,
            _ => true,
        }
    }

    pub fn ensure_computable(self) -> Result<DType> {
        if self.is_computable() {
            Ok(self)
        } else {
            Err(Error::new(
                ErrorCode::DTypeUnsupported,
                format!("Dtype '{}' has no CPU kernel in this build.", self.id()),
            ))
        }
    }

    pub fn kind(self) -> DTypeKind {
        match self {
            DType::Bool => DTypeKind::Bool,
            DType::Int8
            | DType::Int16
            | DType::Int32
            | DType::Int64
            | DType::UInt8
            | DType::UInt16
            | DType::UInt32
            | DType::UInt64 => DTypeKind::Int,
            DType::Float16 | DType::Float32 | DType::Float64 => DTypeKind::Float,
            DType::Complex64 | DType::Complex128 => DTypeKind::Complex,
            _ => DTypeKind::Other,
        }
    }

    pub fn is_signed(self) -> bool {
        matches!(
            self,
            DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64
        )
    }

    /// Item size in bytes.
    pub fn item_size(self) -> usize {
        match self {
            DType::Bool | DType::Int8 | DType::UInt8 => 1,
            DType::Int16 | DType::UInt16 | DType::Float16 => 2,
            DType::Int32 | DType::UInt32 | DType::Float32 => 4,
            DType::Int64
            | DType::UInt64
            | DType::Float64
            | DType::Complex64
            | DType::Datetime64(_)
            | DType::Timedelta64(_) => 8,
            DType::Complex128 => 16,
            DType::Str(width) => width * 4,
        }
    }

    pub fn descriptor(self) -> String {
        match self {
            Self::Str(n) => format!("str[{n}]"),
            Self::Datetime64(u) | Self::Timedelta64(u) => format!("{}[{}]", self.id(), u.id()),
            _ => self.id().into(),
        }
    }

    pub fn temporal_unit(self) -> Option<TimeUnit> {
        match self {
            Self::Datetime64(u) | Self::Timedelta64(u) => Some(u),
            _ => None,
        }
    }

    fn bits(self) -> u32 {
        (self.item_size() * 8) as u32
    }

    /// NumPy `np.result_type` for two array dtypes (NEP 50).
    pub fn promote(a: DType, b: DType) -> Result<DType> {
        a.ensure_computable()?;
        b.ensure_computable()?;
        if a == b {
            return Ok(a);
        }
        if let (DType::Str(a), DType::Str(b)) = (a, b) {
            return Ok(DType::Str(a.max(b)));
        }
        if a.kind() == DTypeKind::Other || b.kind() == DTypeKind::Other {
            return Err(unsupported());
        }
        let (ka, kb) = (a.kind(), b.kind());
        if ka == DTypeKind::Bool {
            return Ok(b);
        }
        if kb == DTypeKind::Bool {
            return Ok(a);
        }
        if ka == DTypeKind::Complex || kb == DTypeKind::Complex {
            let other = if ka == DTypeKind::Complex { b } else { a };
            return Ok(
                if a == DType::Complex128
                    || b == DType::Complex128
                    || other == DType::Float64
                    || (other.kind() == DTypeKind::Int && other.bits() > 16)
                {
                    DType::Complex128
                } else {
                    DType::Complex64
                },
            );
        }
        match (ka, kb) {
            (DTypeKind::Float, DTypeKind::Float) => Ok(if a.bits() >= b.bits() { a } else { b }),
            (DTypeKind::Float, DTypeKind::Int) => Ok(float_for_int(a, b)),
            (DTypeKind::Int, DTypeKind::Float) => Ok(float_for_int(b, a)),
            (DTypeKind::Int, DTypeKind::Int) => Ok(promote_ints(a, b)),
            _ => unreachable!("computable dtypes are bool, int or float"),
        }
    }

    /// Result dtype of combining an array dtype with a weak JS number scalar
    /// (NEP 50).
    pub fn promote_weak(array: DType, scalar: ScalarKind) -> Result<DType> {
        array.ensure_computable()?;
        if array.kind() == DTypeKind::Other {
            return Err(unsupported());
        }
        Ok(match (array.kind(), scalar) {
            (DTypeKind::Bool | DTypeKind::Int, ScalarKind::Complex) => DType::Complex128,
            (DTypeKind::Float, ScalarKind::Complex) => {
                if matches!(array, DType::Float16 | DType::Float32) {
                    DType::Complex64
                } else {
                    DType::Complex128
                }
            }
            (DTypeKind::Bool, ScalarKind::Bool) => DType::Bool,
            (DTypeKind::Bool, ScalarKind::Int) => DType::Int64,
            (DTypeKind::Bool | DTypeKind::Int, ScalarKind::Float) => DType::Float64,
            _ => array,
        })
    }

    /// Dtype of true division (`/`) and `mean`.
    pub fn true_divide(self) -> DType {
        match self.kind() {
            DTypeKind::Float | DTypeKind::Complex => self,
            _ => DType::Float64,
        }
    }

    /// Dtype of `sum` on 64-bit platforms.
    pub fn sum_dtype(self) -> DType {
        match self {
            DType::Bool | DType::Int8 | DType::Int16 | DType::Int32 => DType::Int64,
            DType::UInt8 | DType::UInt16 | DType::UInt32 => DType::UInt64,
            other => other,
        }
    }
}

/// Kind of a weak scalar operand coming from a JS number, bigint or boolean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarKind {
    Other,
    Bool,
    Int,
    Float,
    Complex,
}

/// NumPy promotes an integer with a float to the smallest float that holds the
/// integer exactly enough by its rules: int8/int16/uint8/uint16 with float32
/// stay float32.
fn float_for_int(float: DType, int: DType) -> DType {
    if float == DType::Float16 && int.bits() <= 8 {
        DType::Float16
    } else if matches!(float, DType::Float16 | DType::Float32) && int.bits() <= 16 {
        DType::Float32
    } else {
        DType::Float64
    }
}

fn promote_ints(a: DType, b: DType) -> DType {
    let (sa, sb) = (a.is_signed(), b.is_signed());
    if sa == sb {
        return if a.bits() >= b.bits() { a } else { b };
    }
    let (signed, unsigned) = if sa { (a, b) } else { (b, a) };
    if signed.bits() > unsigned.bits() {
        return signed;
    }
    match unsigned.bits() {
        8 => DType::Int16,
        16 => DType::Int32,
        32 => DType::Int64,
        _ => DType::Float64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values were recorded from NumPy 2.5.4 `np.result_type`.
    #[test]
    fn promotion_matches_numpy() {
        let cases = [
            (DType::Int32, DType::Float32, DType::Float64),
            (DType::Int64, DType::UInt64, DType::Float64),
            (DType::UInt32, DType::Int32, DType::Int64),
            (DType::UInt32, DType::UInt64, DType::UInt64),
            (DType::Float32, DType::Float64, DType::Float64),
            (DType::Bool, DType::Int32, DType::Int32),
            (DType::Int32, DType::Int64, DType::Int64),
        ];
        for (a, b, want) in cases {
            assert_eq!(DType::promote(a, b).unwrap(), want, "{a:?} + {b:?}");
            assert_eq!(DType::promote(b, a).unwrap(), want, "{b:?} + {a:?}");
        }
    }

    #[test]
    fn weak_scalars_keep_array_dtype() {
        assert_eq!(
            DType::promote_weak(DType::Float32, ScalarKind::Float).unwrap(),
            DType::Float32
        );
        assert_eq!(
            DType::promote_weak(DType::Int32, ScalarKind::Int).unwrap(),
            DType::Int32
        );
        assert_eq!(
            DType::promote_weak(DType::Int32, ScalarKind::Float).unwrap(),
            DType::Float64
        );
    }

    #[test]
    fn ids_round_trip() {
        for d in DType::ALL {
            assert_eq!(DType::from_id(d.id()).unwrap(), d);
        }
        assert_eq!(
            DType::from_id("object").unwrap_err().code,
            ErrorCode::DTypeUnsupported
        );
    }

    #[test]
    fn uncomputable_dtypes_are_rejected() {
        let err = DType::promote(DType::Str(1), DType::Float64).unwrap_err();
        assert_eq!(err.code, ErrorCode::DTypeUnsupported);
    }
}

/// NumPy temporal metadata: a positive multiplier and a base unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TimeUnit {
    pub base: TimeBase,
    pub step: u32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TimeBase {
    Year,
    Month,
    Week,
    Day,
    Hour,
    Minute,
    Second,
    Milli,
    Micro,
    Nano,
    Pico,
    Femto,
    Atto,
}
pub fn unsupported() -> Error {
    Error::new(
        ErrorCode::DTypeUnsupported,
        "This dtype, unit conversion or operation is unsupported. Use an explicit supported dtype \
         conversion.",
    )
}
impl TimeUnit {
    pub const NANOSECOND: Self = Self {
        base: TimeBase::Nano,
        step: 1,
    };

    pub fn parse(s: &str) -> Result<Self> {
        let n = s.bytes().take_while(u8::is_ascii_digit).count();
        let step = if n == 0 {
            1
        } else {
            s[..n].parse().map_err(|_| unsupported())?
        };
        if step == 0 || step > i32::MAX as u32 {
            return Err(unsupported());
        }
        let base = match &s[n..] {
            "Y" => TimeBase::Year,
            "M" => TimeBase::Month,
            "W" => TimeBase::Week,
            "D" => TimeBase::Day,
            "h" => TimeBase::Hour,
            "m" => TimeBase::Minute,
            "s" => TimeBase::Second,
            "ms" => TimeBase::Milli,
            "us" => TimeBase::Micro,
            "ns" => TimeBase::Nano,
            "ps" => TimeBase::Pico,
            "fs" => TimeBase::Femto,
            "as" => TimeBase::Atto,
            _ => return Err(unsupported()),
        };
        Ok(Self { base, step })
    }

    pub fn id(self) -> String {
        let base = match self.base {
            TimeBase::Year => "Y",
            TimeBase::Month => "M",
            TimeBase::Week => "W",
            TimeBase::Day => "D",
            TimeBase::Hour => "h",
            TimeBase::Minute => "m",
            TimeBase::Second => "s",
            TimeBase::Milli => "ms",
            TimeBase::Micro => "us",
            TimeBase::Nano => "ns",
            TimeBase::Pico => "ps",
            TimeBase::Femto => "fs",
            TimeBase::Atto => "as",
        };
        if self.step == 1 {
            base.into()
        } else {
            format!("{}{base}", self.step)
        }
    }

    /// Attoseconds; calendar durations use the Gregorian 400-year average.
    pub fn scale(self) -> i128 {
        let seconds: i128 = match self.base {
            TimeBase::Year => 31556952,
            TimeBase::Month => 2629746,
            TimeBase::Week => 604800,
            TimeBase::Day => 86400,
            TimeBase::Hour => 3600,
            TimeBase::Minute => 60,
            TimeBase::Second => 1,
            _ => 0,
        };
        let scale = if seconds != 0 {
            seconds * 1_000_000_000_000_000_000
        } else {
            match self.base {
                TimeBase::Milli => 1_000_000_000_000_000,
                TimeBase::Micro => 1_000_000_000_000,
                TimeBase::Nano => 1_000_000_000,
                TimeBase::Pico => 1_000_000,
                TimeBase::Femto => 1000,
                _ => 1,
            }
        };
        scale * self.step as i128
    }

    pub fn calendar(self) -> bool {
        matches!(self.base, TimeBase::Year | TimeBase::Month)
    }

    pub fn common(self, other: Self) -> Result<Self> {
        if self.calendar() != other.calendar() {
            return Err(unsupported());
        }
        let base = self.base.max(other.base);
        let unit = Self { base, step: 1 };
        let (mut a, mut b) = (self.scale() / unit.scale(), other.scale() / unit.scale());
        while b != 0 {
            (a, b) = (b, a % b);
        }
        Ok(Self {
            base,
            step: u32::try_from(a).map_err(|_| unsupported())?,
        })
    }
}
