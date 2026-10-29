// SPDX-License-Identifier: Apache-2.0
use nmath_core::{
    Buffer, CancellationToken, DenseArray, Device, Expr, NdArray, Scalar, Selector, Value,
    expr::JobEnv,
    layout::CopyPolicy,
    memory::{MemoryBudget, Reservation},
};
fn array() -> NdArray {
    NdArray::from_dense(
        1,
        DenseArray::new(
            vec![2, 3],
            Buffer::I32(vec![0, 1, 2, 3, 4, 5]),
            Reservation::none(),
        )
        .unwrap(),
    )
}
fn env(limit: u64) -> JobEnv {
    JobEnv::new(
        Device::Cpu,
        MemoryBudget::new(Device::Cpu, limit),
        CancellationToken::new(),
    )
}
fn result(env: &mut JobEnv, expr: &Expr) -> NdArray {
    match env.eval(expr).unwrap() {
        Value::View(a) => a,
        Value::Dense(a) => NdArray::from_dense(1, a),
    }
}
fn write(a: &NdArray, index: &[Selector], value: i32) {
    let token = CancellationToken::new();
    let turn = a.write_ticket().wait(&token).unwrap();
    let dense = DenseArray::new(vec![], Buffer::I32(vec![value]), Reservation::none()).unwrap();
    a.assign(&turn, index, &dense, &token).unwrap();
}
#[test]
fn transpose_and_reshape_views_share_writes_and_lifetime() {
    let a = array();
    let input = Expr::input(&a);
    let transposed = Expr::transpose(&input, None).unwrap();
    let mut job = env(4096);
    job.freeze(&transposed).unwrap();
    let view = result(&mut job, &transposed);
    assert_eq!(view.shape(), &[3, 2]);
    assert_eq!(view.to_buffer(), Buffer::I32(vec![0, 3, 1, 4, 2, 5]));
    assert!(view.shares_storage(&a));
    write(&view, &[Selector::Index(1), Selector::Index(0)], 9);
    assert_eq!(a.get_scalar(&[0, 1]).unwrap(), Scalar::Int(9));
    let flat = Expr::reshape(&input, &[-1], CopyPolicy::Never).unwrap();
    let flat = result(&mut job, &flat);
    assert!(flat.shares_storage(&view));
    write(&a, &[Selector::Index(1), Selector::Index(2)], 10);
    assert_eq!(flat.get_scalar(&[5]).unwrap(), Scalar::Int(10));
    drop(a);
    drop(input);
    drop(transposed);
    drop(job);
    assert_eq!(view.get_scalar(&[2, 1]).unwrap(), Scalar::Int(10));
}
#[test]
fn copy_required_transpose_reshape_uses_frozen_values() {
    let a = array();
    let transposed = Expr::transpose(&Expr::input(&a), None).unwrap();
    assert!(Expr::reshape(&transposed, &[6], CopyPolicy::Never).is_err());
    let flat = Expr::reshape(&transposed, &[6], CopyPolicy::IfNeeded).unwrap();
    let mut job = env(4096);
    job.freeze(&flat).unwrap();
    write(&a, &[Selector::Index(0), Selector::Index(0)], 99);
    let output = result(&mut job, &flat);
    assert!(!output.shares_storage(&a));
    assert_eq!(output.to_buffer(), Buffer::I32(vec![0, 3, 1, 4, 2, 5]));
    drop(output);
    drop(job);
}
#[test]
fn forced_copies_and_arithmetic_are_independent_snapshots() {
    let a = array();
    let input = Expr::input(&a);
    let copy = Expr::reshape(&input, &[3, 2], CopyPolicy::Always).unwrap();
    let transposed = Expr::transpose(&input, None).unwrap();
    let sum = Expr::binary(
        nmath_core::BinaryOp::Add,
        &transposed,
        &Expr::scalar(Scalar::Int(1)),
    )
    .unwrap();
    for expr in [copy, sum] {
        let mut job = env(4096);
        job.freeze(&expr).unwrap();
        let output = result(&mut job, &expr);
        assert!(!output.shares_storage(&a));
        assert_eq!(
            output.get_scalar(&[0, 0]).unwrap(),
            Scalar::Int(if expr.operation_id() == "add" { 1 } else { 0 })
        );
        drop(output);
        let budget = job.budget.clone();
        drop(job);
        assert_eq!(budget.usage().used, 0);
    }
}
#[test]
fn computed_transpose_keeps_layout_and_no_copy_validation() {
    let a = array();
    let computed = Expr::binary(
        nmath_core::BinaryOp::Add,
        &Expr::input(&a),
        &Expr::scalar(Scalar::Int(1)),
    )
    .unwrap();
    let transposed = Expr::transpose(&computed, None).unwrap();
    assert!(Expr::reshape(&transposed, &[6], CopyPolicy::Never).is_err());
    let mut job = env(4096);
    job.freeze(&transposed).unwrap();
    let output = result(&mut job, &transposed);
    assert!(!output.shares_storage(&a));
    assert_eq!(output.to_buffer(), Buffer::I32(vec![1, 4, 2, 5, 3, 6]));
    let flattened = Expr::reshape(&Expr::input(&output), &[6], CopyPolicy::Never);
    assert!(flattened.is_err());
}
#[test]
fn layout_jobs_release_budget_on_rejection_and_cancellation() {
    let a = array();
    let copy = Expr::reshape(&Expr::input(&a), &[6], CopyPolicy::Always).unwrap();
    let mut job = env(24);
    job.freeze(&copy).unwrap();
    assert_eq!(job.budget.usage().used, 24);
    assert_eq!(
        job.eval(&copy).unwrap_err().code,
        nmath_core::ErrorCode::MemoryBudgetExceeded
    );
    assert_eq!(job.budget.usage().used, 24);
    let budget = job.budget.clone();
    drop(job);
    assert_eq!(budget.usage().used, 0);
    let mut job = env(4096);
    job.freeze(&copy).unwrap();
    job.token.cancel();
    assert_eq!(
        job.eval(&copy).unwrap_err().code,
        nmath_core::ErrorCode::Cancelled
    );
    assert_eq!(
        copy.check_gpu_support().unwrap_err().code,
        nmath_core::ErrorCode::GpuUnsupportedOperation
    );
}

#[test]
fn stepped_and_reversed_views_reshape_without_copy() {
    let a = array();
    for step in [2, -1] {
        let flat = Expr::reshape(&Expr::input(&a), &[6], CopyPolicy::Never).unwrap();
        let sliced = Expr::index(
            &flat,
            vec![Selector::Slice {
                start: None,
                stop: None,
                step: Some(step),
            }],
        )
        .unwrap();
        let shaped = Expr::reshape(&sliced, &[1, -1], CopyPolicy::Never).unwrap();
        let mut job = env(4096);
        job.freeze(&shaped).unwrap();
        let output = result(&mut job, &shaped);
        assert!(output.shares_storage(&a));
        assert_eq!(
            output.to_buffer(),
            if step == 2 {
                Buffer::I32(vec![0, 2, 4])
            } else {
                Buffer::I32(vec![5, 4, 3, 2, 1, 0])
            }
        );
    }
}
#[test]
fn forced_c_order_copy_of_transpose_is_contiguous() {
    let a = array();
    let transposed = Expr::transpose(&Expr::input(&a), None).unwrap();
    let copy = Expr::reshape(&transposed, &[3, 2], CopyPolicy::Always).unwrap();
    let mut job = env(4096);
    job.freeze(&copy).unwrap();
    let output = result(&mut job, &copy);
    assert!(!output.shares_storage(&a));
    let flat = Expr::reshape(&Expr::input(&output), &[6], CopyPolicy::Never).unwrap();
    let mut next = env(4096);
    next.freeze(&flat).unwrap();
    assert!(result(&mut next, &flat).shares_storage(&output));
}
