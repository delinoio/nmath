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
