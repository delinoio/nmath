// SPDX-License-Identifier: Apache-2.0
use std::sync::mpsc;

use nmath_core::{
    BinaryOp, Buffer, CancellationToken, Context, ContextOptions, Device, ErrorCode, Expr,
    JobOutput, Value,
};

#[test]
#[ignore = "requires actual supported native GPU hardware"]
fn actual_context_gpu_and_separate_budgets() {
    let ctx = Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 0,
        cpu_memory_budget: 1024,
        gpu_memory_budget: 1024,
    })
    .unwrap();
    let a = ctx.array(vec![2], Buffer::I32(vec![2, 3])).unwrap();
    let e = Expr::binary(BinaryOp::Mul, &Expr::input(&a), &Expr::input(&a)).unwrap();
    let (tx, rx) = mpsc::channel();
    ctx.compute(
        &e,
        Device::Gpu,
        CancellationToken::new(),
        Box::new(move |v| tx.send(v).unwrap()),
    )
    .unwrap();
    let result = rx.recv().unwrap().unwrap();
    let JobOutput::Value(Value::Dense(result)) = result else {
        panic!()
    };
    assert_eq!(result.buffer, Buffer::I32(vec![4, 9]));
    assert_eq!(ctx.metrics().gpu_memory.used, 0);
    assert!(ctx.metrics().gpu_memory.peak > 0);
    assert_eq!(ctx.metrics().cpu_memory.used, 16);
    drop(result);
    assert_eq!(ctx.metrics().cpu_memory.used, 8);
    let invalid = Expr::binary(BinaryOp::Div, &Expr::input(&a), &Expr::input(&a)).unwrap();
    let before = ctx.metrics();
    let err = ctx
        .compute(
            &invalid,
            Device::Gpu,
            CancellationToken::new(),
            Box::new(|_| panic!("not admitted")),
        )
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::GpuUnsupportedOperation);
    assert_eq!(
        ctx.metrics().gpu_memory.cumulative_reserved,
        before.gpu_memory.cumulative_reserved
    );
}

#[test]
#[ignore = "requires actual supported native GPU hardware"]
fn actual_context_float32_preserves_underflow_and_rejects_power() {
    let ctx = Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 0,
        cpu_memory_budget: 4096,
        gpu_memory_budget: 4096,
    })
    .unwrap();
    let a = ctx
        .array(
            vec![2],
            Buffer::F32(vec![f32::from_bits(1), f32::MIN_POSITIVE]),
        )
        .unwrap();
    let b = ctx.array(vec![2], Buffer::F32(vec![2.0, 0.5])).unwrap();
    let e = Expr::binary(BinaryOp::Mul, &Expr::input(&a), &Expr::input(&b)).unwrap();
    let (tx, rx) = mpsc::channel();
    ctx.compute(
        &e,
        Device::Gpu,
        CancellationToken::new(),
        Box::new(move |v| tx.send(v).unwrap()),
    )
    .unwrap();
    let JobOutput::Value(Value::Dense(result)) = rx.recv().unwrap().unwrap() else {
        panic!()
    };
    let Buffer::F32(words) = &result.buffer else {
        panic!()
    };
    assert_eq!(
        words.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [2, 0x00400000]
    );
    assert_eq!(ctx.metrics().gpu_memory.used, 0);
    assert_eq!(ctx.metrics().cpu_memory.used, 24);
    drop(result);
    assert_eq!(ctx.metrics().cpu_memory.used, 16);
    let power = Expr::binary(BinaryOp::Pow, &Expr::input(&a), &Expr::input(&b)).unwrap();
    let before = ctx.metrics();
    assert_eq!(
        ctx.compute(
            &power,
            Device::Gpu,
            CancellationToken::new(),
            Box::new(|_| panic!("not admitted"))
        )
        .unwrap_err()
        .code,
        ErrorCode::GpuUnsupportedOperation
    );
    assert_eq!(
        ctx.metrics().gpu_memory.cumulative_reserved,
        before.gpu_memory.cumulative_reserved
    );
}
