// SPDX-License-Identifier: Apache-2.0
//! One-axis CPU transforms. Radix-two butterflies and Bluestein convolution
//! use bounded, precharged workspace and cooperative cancellation.
use crate::{
    buffer::DenseArray,
    dtype::DType,
    error::{Error, ErrorCode, Result},
    memory::MemoryBudget,
    numerical::{C, bytes, overflow, read, size, write, zeros},
    sync::CancellationToken,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FftOp {
    Fft,
    Ifft,
    Rfft,
    Irfft,
}
impl FftOp {
    pub fn id(self) -> &'static str {
        match self {
            Self::Fft => "fft",
            Self::Ifft => "ifft",
            Self::Rfft => "rfft",
            Self::Irfft => "irfft",
        }
    }

    fn inverse(self) -> bool {
        matches!(self, Self::Ifft | Self::Irfft)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm {
    Backward,
    Forward,
    Ortho,
}
#[derive(Clone, Copy, Debug)]
pub struct FftOptions {
    pub op: FftOp,
    pub n: Option<usize>,
    pub axis: i64,
    pub norm: Norm,
}
fn invalid(message: &str) -> Error {
    Error::new(ErrorCode::ShapeMismatch, message)
}
impl FftOptions {
    pub fn infer(self, shape: &[usize], dtype: DType) -> Result<(Vec<usize>, DType, usize, usize)> {
        crate::numerical::numeric_dtype(dtype)?;
        if self.op == FftOp::Rfft && matches!(dtype, DType::Complex64 | DType::Complex128) {
            return Err(Error::new(
                ErrorCode::DTypeUnsupported,
                "rfft requires real input. Cast explicitly.",
            ));
        }
        let rank = shape.len() as i64;
        let axis = if self.axis < 0 {
            self.axis + rank
        } else {
            self.axis
        };
        if axis < 0 || axis >= rank {
            return Err(invalid("FFT axis is outside the input rank."));
        }
        let axis = axis as usize;
        let n = match self.n {
            Some(n) => n,
            None if self.op == FftOp::Irfft => shape[axis]
                .checked_sub(1)
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| invalid("irfft needs a positive transform length."))?,
            None => shape[axis],
        };
        if n == 0 {
            return Err(invalid("FFT transform length must be positive."));
        }
        let single = matches!(dtype, DType::Float16 | DType::Float32 | DType::Complex64);
        let dtype = match (self.op == FftOp::Irfft, single) {
            (true, true) if dtype == DType::Float16 => DType::Float16,
            (true, true) => DType::Float32,
            (true, false) => DType::Float64,
            (false, true) => DType::Complex64,
            (false, false) => DType::Complex128,
        };
        let mut output = shape.to_vec();
        output[axis] = if self.op == FftOp::Rfft { n / 2 + 1 } else { n };
        size(&output)?;
        Ok((output, dtype, axis, n))
    }
}
fn radix2(values: &mut [C], inverse: bool, token: &CancellationToken) -> Result<()> {
    let n = values.len();
    let mut j = 0;
    for i in 1..n {
        if i % 1024 == 0 {
            token.check()?;
        }
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            values.swap(i, j);
        }
    }
    let mut width = 2;
    while width <= n {
        let angle = if inverse {
            std::f64::consts::TAU
        } else {
            -std::f64::consts::TAU
        } / width as f64;
        for base in (0..n).step_by(width) {
            token.check()?;
            for k in 0..width / 2 {
                if k % 1024 == 0 {
                    token.check()?;
                }
                let phase = C::from_polar(1., angle * k as f64);
                let a = values[base + k];
                let b = values[base + k + width / 2] * phase;
                values[base + k] = a + b;
                values[base + k + width / 2] = a - b;
            }
        }
        if width == n {
            break;
        }
        width *= 2;
    }
    Ok(())
}
fn transform(
    values: &mut [C],
    scratch: &mut [C],
    inverse: bool,
    token: &CancellationToken,
) -> Result<()> {
    let n = values.len();
    if n.is_power_of_two() {
        return radix2(values, inverse, token);
    }
    let m = scratch.len() / 2;
    let (a, b) = scratch.split_at_mut(m);
    a.fill(C::new(0., 0.));
    b.fill(C::new(0., 0.));
    let sign = if inverse { 1. } else { -1. };
    // Reduce the integer square modulo 2n before conversion to avoid large
    // floating arguments. usize products use u128 to avoid index overflow.
    let chirp = |k: usize| {
        C::from_polar(
            1.,
            sign * std::f64::consts::PI * ((k as u128 * k as u128) % (2 * n as u128)) as f64
                / n as f64,
        )
    };
    for k in 0..n {
        if k % 1024 == 0 {
            token.check()?;
        }
        let phase = chirp(k);
        a[k] = values[k] * phase;
        b[k] = phase.conj();
        if k != 0 {
            b[m - k] = phase.conj();
        }
    }
    radix2(a, false, token)?;
    radix2(b, false, token)?;
    for k in 0..m {
        if k % 1024 == 0 {
            token.check()?;
        }
        a[k] *= b[k];
    }
    radix2(a, true, token)?;
    for k in 0..n {
        values[k] = a[k] * chirp(k) / m as f64;
    }
    Ok(())
}
pub fn execute(
    options: FftOptions,
    input: &DenseArray,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    token.check()?;
    let (shape, dtype, axis, n) = options.infer(&input.shape, input.buffer.dtype())?;
    let count = size(&shape)?;
    let charge = budget.reserve(bytes(count, dtype.item_size())?)?;
    let mut output = zeros(dtype, count)?;
    if count == 0 {
        return DenseArray::new(shape, output, charge);
    }
    let scratch_len = if n.is_power_of_two() {
        0
    } else {
        n.checked_mul(2)
            .and_then(|v| v.checked_sub(1))
            .and_then(usize::checked_next_power_of_two)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(overflow)?
    };
    let _workspace = budget.reserve(bytes(
        n.checked_add(scratch_len).ok_or_else(overflow)?,
        std::mem::size_of::<C>(),
    )?)?;
    let mut values = vec![C::new(0., 0.); n];
    let mut scratch = vec![C::new(0., 0.); scratch_len];
    let inner = size(&input.shape[axis + 1..])?;
    let outer = size(&input.shape[..axis])?;
    let input_n = input.shape[axis];
    let output_n = shape[axis];
    let scale = match options.norm {
        Norm::Ortho => 1. / (n as f64).sqrt(),
        Norm::Backward if options.op.inverse() => 1. / n as f64,
        Norm::Forward if !options.op.inverse() => 1. / n as f64,
        _ => 1.,
    };
    for batch in 0..outer {
        for lane in 0..inner {
            token.check()?;
            values.fill(C::new(0., 0.));
            let copy_n = if options.op == FftOp::Irfft {
                input_n.min(n / 2 + 1)
            } else {
                input_n.min(n)
            };
            for (k, v) in values.iter_mut().enumerate().take(copy_n) {
                if k % 1024 == 0 {
                    token.check()?;
                }
                *v = read(&input.buffer, (batch * input_n + k) * inner + lane);
            }
            if options.op == FftOp::Irfft {
                values[0].im = 0.;
                if n % 2 == 0 {
                    values[n / 2].im = 0.;
                }
                for k in 1..=(n - 1) / 2 {
                    values[n - k] = values[k].conj();
                }
            }
            transform(&mut values, &mut scratch, options.op.inverse(), token)?;
            for (k, v) in values.iter().enumerate().take(output_n) {
                if k % 1024 == 0 {
                    token.check()?;
                }
                write(
                    &mut output,
                    (batch * output_n + k) * inner + lane,
                    *v * scale,
                );
            }
        }
    }
    token.check()?;
    DenseArray::new(shape, output, charge)
}
