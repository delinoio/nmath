// SPDX-License-Identifier: Apache-2.0
use std::{sync::mpsc, time::Duration};

use super::*;
use crate::{
    Buffer, Context, ContextOptions, Device, ErrorCode, Expr, Value,
    context::{Done, JobOutput, OperationKind},
    dtype::DType,
    sync::CancellationToken,
};

fn context() -> Context {
    Context::new(ContextOptions {
        max_concurrent_jobs: 2,
        max_queue_length: 8,
        cpu_memory_budget: 1_000_000,
        gpu_memory_budget: 0,
    })
    .unwrap()
}
fn channel() -> (Done, mpsc::Receiver<crate::Result<JobOutput>>) {
    let (tx, rx) = mpsc::channel();
    (
        Box::new(move |result| {
            tx.send(result).unwrap();
        }),
        rx,
    )
}
fn wait(rx: mpsc::Receiver<crate::Result<JobOutput>>) -> crate::Result<JobOutput> {
    rx.recv_timeout(Duration::from_secs(10)).unwrap()
}
fn values(out: JobOutput) -> Buffer {
    match out {
        JobOutput::Value(Value::Dense(d)) => d.buffer,
        _ => panic!("expected dense result"),
    }
}

#[test]
fn fifo_wait_cancellation_and_independent_reexecution_preserve_cached_words() {
    let ctx = context();
    let generator = ctx.generator(42);
    let expression = Expr::random_operation(
        ctx.id(),
        &generator,
        vec![3],
        RandomOperation::Random(DType::Float32),
    )
    .unwrap();
    let before = generator.current();
    let held = generator.ticket().wait(&CancellationToken::new()).unwrap();
    let mut jobs = Vec::new();
    for index in 0..4 {
        let token = CancellationToken::new();
        let (done, rx) = channel();
        ctx.compute(&expression, Device::Cpu, token.clone(), done)
            .unwrap();
        jobs.push((index, token, rx));
    }
    // One running waiter and one queued request are abandoned.
    jobs[0].1.cancel();
    jobs[2].1.cancel();
    assert_eq!(generator.current(), before);
    drop(held);
    let mut oracle = before;
    for (index, _, rx) in jobs {
        let result = wait(rx);
        if index == 0 || index == 2 {
            assert_eq!(result.unwrap_err().code, ErrorCode::Cancelled);
        } else {
            assert_eq!(
                values(result.unwrap()),
                Buffer::F32((0..3).map(|_| oracle.next_f32()).collect())
            );
        }
    }
    assert_eq!(generator.current(), oracle);
    let (done, rx) = channel();
    ctx.compute(&expression, Device::Cpu, CancellationToken::new(), done)
        .unwrap();
    assert_eq!(
        values(wait(rx).unwrap()),
        Buffer::F32((0..3).map(|_| oracle.next_f32()).collect())
    );
    assert_eq!(generator.current(), oracle);
    ctx.close();
}

#[test]
fn late_cancellation_after_evaluation_discards_pending_state() {
    let ctx = context();
    let generator = ctx.generator(42);
    let expression = Expr::random_operation(
        ctx.id(),
        &generator,
        vec![3],
        RandomOperation::Random(DType::Float32),
    )
    .unwrap();
    let before = generator.current();
    let ticket = generator.ticket();
    let (done, rx) = channel();
    ctx.submit(
        OperationKind::Compute,
        "random",
        Device::Cpu,
        CancellationToken::new(),
        Box::new(move |env| {
            env.generator_turns.push(ticket.wait(&env.token)?);
            env.freeze(&expression)?;
            let value = env.eval(&expression)?;
            env.token.cancel();
            Ok(JobOutput::Value(value))
        }),
        done,
    )
    .unwrap();
    assert_eq!(wait(rx).unwrap_err().code, ErrorCode::Cancelled);
    assert_eq!(generator.current(), before);
    ctx.close();
}

#[test]
fn evaluation_error_discards_pending_generator_state() {
    let ctx = context();
    let generator = ctx.generator(42);
    let expression = Expr::random_operation(
        ctx.id(),
        &generator,
        vec![3],
        RandomOperation::Random(DType::Float32),
    )
    .unwrap();
    let before = generator.current();
    let ticket = generator.ticket();
    let (done, rx) = channel();
    ctx.submit(
        OperationKind::Compute,
        "random",
        Device::Cpu,
        CancellationToken::new(),
        Box::new(move |env| {
            env.generator_turns.push(ticket.wait(&env.token)?);
            env.freeze(&expression)?;
            env.eval(&expression)?;
            Err(crate::Error::new(
                ErrorCode::MemoryBudgetExceeded,
                "Test failure after random evaluation.",
            ))
        }),
        done,
    )
    .unwrap();
    assert_eq!(wait(rx).unwrap_err().code, ErrorCode::MemoryBudgetExceeded);
    assert_eq!(generator.current(), before);
    ctx.close();
}
