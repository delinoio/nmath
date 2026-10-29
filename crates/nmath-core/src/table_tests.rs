// SPDX-License-Identifier: Apache-2.0
use std::{sync::mpsc, time::Duration};

use super::*;
use crate::{ContextOptions, Selector};
fn context(budget: u64) -> Context {
    Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 1,
        cpu_memory_budget: budget,
        gpu_memory_budget: 0,
    })
    .unwrap()
}
fn source(a: &crate::NdArray) -> Plan {
    Plan::Source(vec![Column {
        values: Expr::input(a),
        valid: None,
    }])
}
fn run(c: &Context, p: Plan) -> Result<Table> {
    let (tx, rx) = mpsc::channel();
    compute(
        c,
        p,
        CancellationToken::new(),
        Box::new(move |r| tx.send(r).unwrap()),
    )?;
    rx.recv_timeout(Duration::from_secs(10)).unwrap()
}
fn blocker(c: &Context) -> mpsc::Sender<()> {
    let (start_tx, start_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    c.submit(
        OperationKind::Compute,
        "table-test-block",
        Device::Cpu,
        CancellationToken::new(),
        Box::new(move |_| {
            start_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(JobOutput::Unit)
        }),
        Box::new(|_| {}),
    )
    .unwrap();
    start_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    release_tx
}
fn write(a: &crate::NdArray, value: i64) {
    let c = context(1024);
    let d = c.array(vec![], Buffer::I64(vec![value])).unwrap();
    let budget = crate::memory::MemoryBudget::new(Device::Cpu, 1024);
    let dense = d.snapshot(&budget).unwrap();
    let token = CancellationToken::new();
    let turn = a.write_ticket().wait(&token).unwrap();
    a.assign(&turn, &[Selector::Index(0)], &dense, &token)
        .unwrap();
}
#[test]
fn queue_admission_snapshot_and_cancel() {
    let c = context(4096);
    let a = c.array(vec![1], Buffer::I64(vec![1])).unwrap();
    let release = blocker(&c);
    let (tx, rx) = mpsc::channel();
    compute(
        &c,
        source(&a),
        CancellationToken::new(),
        Box::new(move |r| tx.send(r).unwrap()),
    )
    .unwrap();
    assert_eq!(
        compute(&c, source(&a), CancellationToken::new(), Box::new(|_| {}))
            .unwrap_err()
            .code,
        ErrorCode::SchedQueueFull
    );
    write(&a, 7);
    release.send(()).unwrap();
    let out = rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
    assert_eq!(out.columns[0].values.buffer, Buffer::I64(vec![7]));
    drop(out);
    // close waits for the scheduler's post-completion cleanup before a new
    // test.
    c.close();
    let c = context(4096);
    let a = c.array(vec![1], Buffer::I64(vec![1])).unwrap();
    let release = blocker(&c);
    let token = CancellationToken::new();
    let (tx, rx) = mpsc::channel();
    compute(
        &c,
        source(&a),
        token.clone(),
        Box::new(move |r| tx.send(r).unwrap()),
    )
    .unwrap();
    token.cancel();
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(5))
            .unwrap()
            .err()
            .unwrap()
            .code,
        ErrorCode::Cancelled
    );
    release.send(()).unwrap();
    c.close();
}
#[test]
fn frozen_columns_use_one_environment_and_retain_values() {
    let c = context(4096);
    let a = c.array(vec![2], Buffer::I64(vec![1, 2])).unwrap();
    let b = Expr::binary(
        crate::BinaryOp::Add,
        &Expr::input(&a),
        &Expr::scalar(Scalar::Int(10)),
    )
    .unwrap();
    let p = Plan::Source(vec![
        Column {
            values: Expr::input(&a),
            valid: None,
        },
        Column {
            values: b,
            valid: None,
        },
    ]);
    let budget = crate::memory::MemoryBudget::new(Device::Cpu, 4096);
    let mut env = JobEnv::new(Device::Cpu, budget, CancellationToken::new());
    let mut inputs = Vec::new();
    p.sources(&mut inputs);
    {
        let _gate = SNAPSHOT_GATE.lock().unwrap();
        for input in &inputs {
            env.freeze(input).unwrap();
        }
    }
    write(&a, 99);
    let out = execute(&p, &mut env).unwrap();
    assert_eq!(out.columns[0].values.buffer, Buffer::I64(vec![1, 2]));
    assert_eq!(out.columns[1].values.buffer, Buffer::I64(vec![11, 12]));
}
#[test]
fn budgets_contexts_exact_wide_keys_and_empty() {
    let c = context(8);
    let a = c.array(vec![1], Buffer::I64(vec![1])).unwrap();
    assert_eq!(
        run(&c, source(&a)).err().unwrap().code,
        ErrorCode::MemoryBudgetExceeded
    );
    assert_eq!(c.metrics().cpu_memory.used, 8);
    c.close();
    let c = context(4096);
    let other = context(4096);
    let a = other.array(vec![1], Buffer::I64(vec![1])).unwrap();
    assert_eq!(
        run(&c, source(&a)).err().unwrap().code,
        ErrorCode::ContextMismatch
    );
    assert_eq!(c.metrics().total_submitted, 0);
    let a = c
        .array(vec![3], Buffer::U64(vec![u64::MAX, u64::MAX - 1, u64::MAX]))
        .unwrap();
    let out = run(
        &c,
        Plan::Group(
            Box::new(source(&a)),
            0,
            0,
            Aggregate::Count,
            false,
            Flavor::Pandas,
        ),
    )
    .unwrap();
    assert_eq!(
        out.columns[0].values.buffer,
        Buffer::U64(vec![u64::MAX, u64::MAX - 1])
    );
    assert_eq!(out.columns[1].values.buffer, Buffer::I64(vec![2, 1]));
    let a = c.array(vec![0], Buffer::I64(vec![])).unwrap();
    assert_eq!(
        run(&c, Plan::Sort(Box::new(source(&a)), 0, false, true))
            .unwrap()
            .rows,
        0
    );
    c.close();
    other.close();
}
#[test]
fn running_cancellation_releases_intermediates() {
    let c = context(2_000_000);
    let a = c
        .array(vec![20000], Buffer::I64((0..20000).rev().collect()))
        .unwrap();
    let baseline = c.metrics().cpu_memory.used;
    let (tx, rx) = mpsc::channel();
    let token = CancellationToken::new();
    compute(
        &c,
        Plan::Sort(Box::new(source(&a)), 0, false, true),
        token.clone(),
        Box::new(move |r| tx.send(r).unwrap()),
    )
    .unwrap();
    // Cancellation is safe both before the first comparison and inside sorting.
    token.cancel();
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(5))
            .unwrap()
            .err()
            .unwrap()
            .code,
        ErrorCode::Cancelled
    );
    c.close();
    assert_eq!(c.metrics().cpu_memory.used, baseline);
}
