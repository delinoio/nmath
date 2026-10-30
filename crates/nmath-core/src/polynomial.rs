// SPDX-License-Identifier: Apache-2.0
//! Native Horner evaluation with explicit coefficient ordering and tensor
//! layout.
use crate::{
    buffer::DenseArray,
    dtype::{DType, DTypeKind},
    error::{Error, ErrorCode, Result},
    memory::MemoryBudget,
    shape::broadcast_shapes,
    sync::CancellationToken,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolynomialOrder {
    Ascending,
    Descending,
}
impl PolynomialOrder {
    pub fn id(self) -> &'static str {
        match self {
            Self::Ascending => "polynomial.polyval",
            Self::Descending => "polyval",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PolynomialOptions {
    pub order: PolynomialOrder,
    pub tensor: bool,
}
impl PolynomialOptions {
    pub fn infer(
        self,
        coefficients: &[usize],
        cd: DType,
        points: &[usize],
        xd: DType,
    ) -> Result<(Vec<usize>, DType)> {
        crate::numerical::numeric_dtype(cd)?;
        crate::numerical::numeric_dtype(xd)?;
        if coefficients.is_empty() && self.order == PolynomialOrder::Descending {
            return Err(Error::new(
                ErrorCode::ShapeMismatch,
                "Descending polynomial coefficients require an axis. Use a length-one array for a \
                 constant.",
            ));
        }
        if self.order == PolynomialOrder::Descending && self.tensor {
            return Err(Error::new(
                ErrorCode::ShapeMismatch,
                "Descending polynomial evaluation requires tensor=false.",
            ));
        }
        if coefficients.first() == Some(&0) {
            return if self.order == PolynomialOrder::Ascending {
                Err(Error::new(
                    ErrorCode::IndexOutOfBounds,
                    "Ascending polynomial coefficients must not be empty. Provide at least one \
                     coefficient.",
                ))
            } else {
                Ok((points.to_vec(), xd))
            };
        }
        let tail = coefficients.get(1..).unwrap_or(&[]);
        let shape = if self.tensor {
            [tail, points].concat()
        } else {
            broadcast_shapes(tail, points)?
        };
        let (cd, xd) = if self.order == PolynomialOrder::Ascending {
            (
                if matches!(cd.kind(), DTypeKind::Bool | DTypeKind::Int) {
                    DType::Float64
                } else {
                    cd
                },
                if xd == DType::Bool { DType::Int64 } else { xd },
            )
        } else {
            (cd, xd)
        };
        let dtype = DType::promote(cd, xd)?;
        crate::buffer::checked_byte_len(
            if shape.contains(&0) {
                0
            } else {
                crate::numerical::size(&shape)?
            },
            dtype,
        )?;
        Ok((shape, dtype))
    }
}

pub fn execute(
    options: PolynomialOptions,
    coefficients: &DenseArray,
    points: &DenseArray,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    crate::kernels::polynomial(options, coefficients, points, budget, token)
}

#[derive(Clone, Copy, Debug)]
pub enum DerivativeScale {
    Real(f64),
    Complex([f64; 2]),
}

#[derive(Clone, Copy, Debug)]
pub struct DerivativeOptions {
    pub order: PolynomialOrder,
    pub count: u64,
    pub axis: i64,
    pub scale: DerivativeScale,
}

impl DerivativeOptions {
    pub fn id(self) -> &'static str {
        match self.order {
            PolynomialOrder::Ascending => "polynomial.polyder",
            PolynomialOrder::Descending => "polyder",
        }
    }

    pub fn infer(self, shape: &[usize], dtype: DType) -> Result<(Vec<usize>, DType, usize)> {
        crate::numerical::numeric_dtype(dtype)?;
        if self.order == PolynomialOrder::Descending && (shape.len() != 1 || self.axis != 0) {
            return Err(Error::new(
                ErrorCode::ShapeMismatch,
                "Descending polynomial derivatives require one coefficient axis. Use ascending \
                 coefficients for multivariate polynomials.",
            ));
        }
        let mut shape = if shape.is_empty() {
            vec![1]
        } else {
            shape.to_vec()
        };
        let rank = shape.len() as i64;
        let axis = if self.axis < 0 {
            self.axis + rank
        } else {
            self.axis
        };
        if !(0..rank).contains(&axis) {
            return Err(Error::new(
                ErrorCode::IndexInvalid,
                "The polynomial derivative axis is out of range. Select an existing axis.",
            ));
        }
        let axis = axis as usize;
        let n = shape[axis];
        let dtype = match self.order {
            PolynomialOrder::Ascending => {
                if self.count > 0
                    && self.count < n as u64
                    && matches!(self.scale, DerivativeScale::Complex(_))
                    && dtype.kind() != DTypeKind::Complex
                {
                    return Err(Error::new(
                        ErrorCode::DTypeUnsupported,
                        "Complex derivative scaling requires complex coefficients. Cast them \
                         explicitly.",
                    ));
                }
                if matches!(dtype.kind(), DTypeKind::Bool | DTypeKind::Int) {
                    DType::Float64
                } else {
                    dtype
                }
            }
            PolynomialOrder::Descending if self.count > 0 => DType::promote(dtype, DType::Int64)?,
            PolynomialOrder::Descending => dtype,
        };
        shape[axis] = if self.count == 0 {
            n
        } else if self.count >= n as u64 {
            if self.order == PolynomialOrder::Ascending {
                n.min(1)
            } else {
                0
            }
        } else {
            n - self.count as usize
        };
        let size = if shape.contains(&0) {
            0
        } else {
            crate::numerical::size(&shape)?
        };
        crate::buffer::checked_byte_len(size, dtype)?;
        Ok((shape, dtype, axis))
    }
}

pub fn derivative(
    options: DerivativeOptions,
    coefficients: &DenseArray,
    budget: &MemoryBudget,
    token: &CancellationToken,
) -> Result<DenseArray> {
    crate::kernels::polynomial_derivative(options, coefficients, budget, token)
}
