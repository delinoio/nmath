// SPDX-License-Identifier: Apache-2.0
//! Shared checked allocation and scalar transport for native numerical kernels.
pub(crate) use num_complex::Complex64 as C;

use crate::{
    buffer::{Buffer, Scalar},
    dtype::DType,
    error::{Error, ErrorCode, Result},
};

pub(crate) fn size(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d).ok_or_else(overflow))
}
pub(crate) fn bytes(n: usize, width: usize) -> Result<u64> {
    n.checked_mul(width)
        .and_then(|v| u64::try_from(v).ok())
        .ok_or_else(overflow)
}
pub(crate) fn overflow() -> Error {
    Error::new(
        ErrorCode::MemoryBudgetExceeded,
        "Numerical workspace size overflow.",
    )
}
pub(crate) fn read(b: &Buffer, i: usize) -> C {
    let [re, im] = b.get(i).as_complex();
    C::new(re, im)
}
pub(crate) fn write(b: &mut Buffer, i: usize, z: C) {
    match b {
        Buffer::F16(v) => v[i] = crate::buffer::Float16::from_f64(z.re),
        Buffer::F32(v) => v[i] = z.re as f32,
        Buffer::F64(v) => v[i] = z.re,
        Buffer::C64(v) => v[i] = [z.re as f32, z.im as f32],
        Buffer::C128(v) => v[i] = [z.re, z.im],
        _ => unreachable!("numerical output dtype was inferred"),
    }
}
pub(crate) fn zeros(dtype: DType, n: usize) -> Result<Buffer> {
    Buffer::filled(dtype, n, Scalar::Float(0.0))
}

/// Keep numerical admission independent of other computable dtype families.
pub(crate) fn numeric_dtype(dtype: DType) -> Result<()> {
    if matches!(
        dtype,
        DType::Bool
            | DType::Int8
            | DType::Int16
            | DType::Int32
            | DType::Int64
            | DType::UInt8
            | DType::UInt16
            | DType::UInt32
            | DType::UInt64
            | DType::Float16
            | DType::Float32
            | DType::Float64
            | DType::Complex64
            | DType::Complex128
    ) {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::DTypeUnsupported,
            "This numerical operation requires a numeric or boolean dtype.",
        ))
    }
}
