// SPDX-License-Identifier: Apache-2.0
use nmath_core::{
    Buffer, CancellationToken, DenseArray, Device, ErrorCode, Expr, NdArray, ShiftOp, kernels,
    memory::{MemoryBudget, Reservation},
};

fn dense(shape: &[usize], buffer: Buffer) -> DenseArray {
    DenseArray::new(shape.to_vec(), buffer, Reservation::none()).unwrap()
}

#[test]
fn shifts_roll_the_requested_axes_in_numpy_direction() {
    let token = CancellationToken::new();
    let budget = MemoryBudget::new(Device::Cpu, 1 << 20);
    let even = dense(&[4], Buffer::I32(vec![0, 1, 2, 3]));
    let forward = kernels::shift(ShiftOp::FftShift, &even, &[0], &budget, &token).unwrap();
    assert_eq!(forward.buffer, Buffer::I32(vec![2, 3, 0, 1]));
    let backward = kernels::shift(ShiftOp::IfftShift, &even, &[0], &budget, &token).unwrap();
    assert_eq!(backward.buffer, Buffer::I32(vec![2, 3, 0, 1]));
    drop(forward);
    drop(backward);

    let odd = dense(&[5], Buffer::I32(vec![0, 1, 2, 3, 4]));
    assert_eq!(
        kernels::shift(ShiftOp::FftShift, &odd, &[0], &budget, &token)
            .unwrap()
            .buffer,
        Buffer::I32(vec![3, 4, 0, 1, 2])
    );
    assert_eq!(
        kernels::shift(ShiftOp::IfftShift, &odd, &[0], &budget, &token)
            .unwrap()
            .buffer,
        Buffer::I32(vec![2, 3, 4, 0, 1])
    );

    let matrix = dense(&[2, 3], Buffer::I32(vec![0, 1, 2, 3, 4, 5]));
    assert_eq!(
        kernels::shift(ShiftOp::FftShift, &matrix, &[0, 1], &budget, &token)
            .unwrap()
            .buffer,
        Buffer::I32(vec![5, 3, 4, 2, 0, 1])
    );
    assert_eq!(
        kernels::shift(ShiftOp::FftShift, &matrix, &[0], &budget, &token)
            .unwrap()
            .buffer,
        Buffer::I32(vec![3, 4, 5, 0, 1, 2])
    );

    let floats = dense(&[3], Buffer::F64(vec![1.5, -0.0, 2.5]));
    let shifted = kernels::shift(ShiftOp::FftShift, &floats, &[0], &budget, &token).unwrap();
    assert_eq!(shifted.buffer, Buffer::F64(vec![2.5, 1.5, -0.0]));
    drop(shifted);

    let empty = dense(&[0, 3], Buffer::F32(vec![]));
    assert_eq!(
        kernels::shift(ShiftOp::IfftShift, &empty, &[0, 1], &budget, &token)
            .unwrap()
            .shape,
        vec![0, 3]
    );
    assert_eq!(budget.usage().used, 0);
}

#[test]
fn expression_validation_rejects_repeated_and_out_of_range_axes() {
    let matrix = dense(&[2, 3], Buffer::I32(vec![0, 1, 2, 3, 4, 5]));
    let input = Expr::input(&NdArray::from_dense(0, matrix));
    let all = Expr::shift(ShiftOp::FftShift, &input, None).unwrap();
    assert_eq!(all.shape(), &[2, 3]);
    assert_eq!(all.dtype(), nmath_core::DType::Int32);
    assert_eq!(
        Expr::shift(ShiftOp::FftShift, &input, Some(vec![0, 0]))
            .unwrap_err()
            .code,
        ErrorCode::IndexInvalid
    );
    assert_eq!(
        Expr::shift(ShiftOp::FftShift, &input, Some(vec![2]))
            .unwrap_err()
            .code,
        ErrorCode::IndexOutOfBounds
    );
    let negative = Expr::shift(ShiftOp::IfftShift, &input, Some(vec![-1])).unwrap();
    assert_eq!(negative.shape(), &[2, 3]);
    assert_eq!(
        nmath_core::gpu::validate(&all).unwrap_err().code,
        ErrorCode::GpuUnsupportedOperation
    );
}

#[test]
fn shift_budgets_and_cancellation() {
    let matrix = dense(&[2, 3], Buffer::I32(vec![0, 1, 2, 3, 4, 5]));
    let token = CancellationToken::new();
    for limit in [0, 8, 15] {
        let budget = MemoryBudget::new(Device::Cpu, limit);
        assert_eq!(
            kernels::shift(ShiftOp::FftShift, &matrix, &[0], &budget, &token)
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded
        );
        assert_eq!(budget.usage().used, 0);
    }
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let budget = MemoryBudget::new(Device::Cpu, 1 << 20);
    assert_eq!(
        kernels::shift(ShiftOp::FftShift, &matrix, &[0], &budget, &cancelled)
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
}
