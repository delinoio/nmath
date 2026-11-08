// SPDX-License-Identifier: Apache-2.0
use nmath_core::{
    Buffer, CancellationToken, DenseArray, Device, ErrorCode,
    linalg::{self, LinalgOp},
    memory::{MemoryBudget, Reservation},
};

fn dense(shape: Vec<usize>, buffer: Buffer) -> DenseArray {
    DenseArray::new(shape, buffer, Reservation::none()).unwrap()
}

fn determinant(a: &DenseArray, budget: &MemoryBudget, token: &CancellationToken) -> DenseArray {
    linalg::execute(LinalgOp::Det, a, None, budget, token).unwrap()
}

#[test]
fn compensating_pivots_preserve_finite_determinants() {
    let budget = MemoryBudget::new(Device::Cpu, u64::MAX);
    let token = CancellationToken::new();
    for diagonal in [
        [1e100, 1e100, 1e100, 1e100, 1e-100, 1e-100, 1e-100, 1e-100],
        [1e-100, 1e-100, 1e-100, 1e-100, 1e100, 1e100, 1e100, 1e100],
    ] {
        let mut values = vec![0.0; 64];
        for (i, value) in diagonal.into_iter().enumerate() {
            values[i * 8 + i] = value;
        }
        let a = dense(vec![8, 8], Buffer::F64(values));
        let result = determinant(&a, &budget, &token);
        let Buffer::F64(values) = &result.buffer else {
            panic!("determinant must retain float64");
        };
        assert!(values[0].is_finite() && values[0] != 0.0);
        assert!((values[0] - 1.0).abs() < 1e-12, "{}", values[0]);
    }
    assert_eq!(budget.usage().used, 0);
}

#[test]
fn signs_phases_singular_and_empty_cases() {
    let budget = MemoryBudget::new(Device::Cpu, u64::MAX);
    let token = CancellationToken::new();
    let negative = dense(vec![2, 2], Buffer::F64(vec![-2.0, 0.0, 0.0, 3.0]));
    assert_eq!(
        determinant(&negative, &budget, &token).buffer,
        Buffer::F64(vec![-6.0])
    );
    let swapped = dense(vec![2, 2], Buffer::F64(vec![0.0, 1.0, 1.0, 0.0]));
    assert_eq!(
        determinant(&swapped, &budget, &token).buffer,
        Buffer::F64(vec![-1.0])
    );

    let complex = dense(
        vec![2, 2],
        Buffer::C128(vec![[1e100, 1e100], [0.0, 0.0], [0.0, 0.0], [1e-100, 0.0]]),
    );
    let result = determinant(&complex, &budget, &token);
    let Buffer::C128(values) = &result.buffer else {
        panic!("determinant must retain complex128");
    };
    assert!((values[0][0] - 1.0).abs() < 1e-12);
    assert!((values[0][1] - 1.0).abs() < 1e-12);
    drop(result);

    let singular = dense(vec![2, 2], Buffer::F64(vec![1.0, 2.0, 2.0, 4.0]));
    assert_eq!(
        determinant(&singular, &budget, &token).buffer,
        Buffer::F64(vec![0.0])
    );
    let empty = dense(vec![0, 0], Buffer::C64(vec![]));
    assert_eq!(
        determinant(&empty, &budget, &token).buffer,
        Buffer::C64(vec![[1.0, 0.0]])
    );
    assert_eq!(budget.usage().used, 0);
}

#[test]
fn determinant_retains_batch_dtype_cancellation_and_budget() {
    let budget = MemoryBudget::new(Device::Cpu, u64::MAX);
    let token = CancellationToken::new();
    let batch = dense(
        vec![2, 2, 2],
        Buffer::F32(vec![2.0, 0.0, 0.0, 3.0, 0.0, 1.0, 1.0, 0.0]),
    );
    let result = determinant(&batch, &budget, &token);
    assert_eq!(result.shape, vec![2]);
    assert_eq!(result.buffer, Buffer::F32(vec![6.0, -1.0]));
    drop(result);

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert_eq!(
        linalg::execute(LinalgOp::Det, &batch, None, &budget, &cancelled)
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
    let insufficient = MemoryBudget::new(Device::Cpu, 0);
    assert_eq!(
        linalg::execute(LinalgOp::Det, &batch, None, &insufficient, &token)
            .unwrap_err()
            .code,
        ErrorCode::MemoryBudgetExceeded
    );
    assert_eq!(insufficient.usage().used, 0);
    assert_eq!(budget.usage().used, 0);
}
