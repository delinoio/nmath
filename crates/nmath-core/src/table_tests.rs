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
    let frozen = streaming::Frozen::new(&p, &mut env).unwrap();
    write(&a, 99);
    let out = execute(&p, &mut env, &frozen).unwrap();
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
fn pandas_group_excludes_nan_without_changing_polars_or_null_groups() {
    let c = context(8192);
    let keys = c
        .array(vec![4], Buffer::F64(vec![2.0, f64::NAN, 1.0, 3.0]))
        .unwrap();
    let values = c.array(vec![4], Buffer::I32(vec![4, 20, 3, 100])).unwrap();
    let mask = c
        .array(vec![4], Buffer::Bool(vec![true, true, true, false]))
        .unwrap();
    let source = || {
        Plan::Source(vec![
            Column {
                values: Expr::input(&keys),
                valid: Some(Expr::input(&mask)),
            },
            Column {
                values: Expr::input(&values),
                valid: None,
            },
        ])
    };
    let run_group = |drop_nulls, flavor| {
        run(
            &c,
            Plan::Group(Box::new(source()), 0, 1, Aggregate::Sum, drop_nulls, flavor),
        )
        .unwrap()
    };
    let pandas = run_group(true, Flavor::Pandas);
    assert_eq!(pandas.columns[0].values.buffer, Buffer::F64(vec![2.0, 1.0]));
    assert_eq!(pandas.columns[1].values.buffer, Buffer::I32(vec![4, 3]));
    let retained = run_group(false, Flavor::Pandas);
    assert_eq!(retained.rows, 4);
    assert_eq!(
        retained.columns[1].values.buffer,
        Buffer::I32(vec![4, 20, 3, 100])
    );
    let polars = run_group(true, Flavor::Polars);
    assert_eq!(polars.rows, 3);
    assert_eq!(polars.columns[1].values.buffer, Buffer::I32(vec![4, 20, 3]));
    c.close();
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

#[test]
fn complex_columns_reject_before_admission() {
    let c = context(4096);
    let a = c.array(vec![1], Buffer::C128(vec![[1.0, 2.0]])).unwrap();
    let error = run(&c, source(&a)).err().unwrap();
    assert_eq!(error.code, ErrorCode::DTypeUnsupported);
}

#[test]
fn batched_arithmetic_filter_projection_fits_without_full_intermediates() {
    const N: usize = 8193;
    // Inputs + atomic snapshots fit. A full input copy and arithmetic result
    // would exceed this budget even before materializing masks and selection.
    let c = context(250_000);
    let a = c
        .array(vec![N], Buffer::I64((0..N as i64).collect()))
        .unwrap();
    let mask = c
        .array(
            vec![N],
            Buffer::Bool((0..N).map(|i| i % 1024 == 0).collect()),
        )
        .unwrap();
    let valid = c
        .array(vec![N], Buffer::Bool((0..N).map(|i| i != 1024).collect()))
        .unwrap();
    let mut expr = Expr::input(&a);
    for _ in 0..12 {
        expr = Expr::binary(crate::BinaryOp::Add, &expr, &Expr::scalar(Scalar::Int(1))).unwrap();
    }
    let p = Plan::Source(vec![
        Column {
            values: expr,
            valid: Some(Expr::input(&valid)),
        },
        Column {
            values: Expr::input(&mask),
            valid: Some(Expr::input(&valid)),
        },
    ]);
    // The projected predicate refers to a different position than at source.
    let p = Plan::Select(Box::new(p), vec![1, 0, 0]);
    let p = Plan::Filter(Box::new(p), 0);
    let p = Plan::Select(Box::new(p), vec![2, 1]);
    let baseline = c.metrics().cpu_memory.used;
    let out = run(&c, p).unwrap();
    let expected = Buffer::I64(
        (0..N)
            .filter(|i| i % 1024 == 0 && *i != 1024)
            .map(|i| i as i64 + 12)
            .collect(),
    );
    assert_eq!(out.rows, 8);
    assert_eq!(out.columns[0].values.buffer, expected);
    assert_eq!(out.columns[1].values.buffer, expected);
    assert_eq!(out.columns[0].valid.buffer, Buffer::Bool(vec![true; 8]));
    assert!(c.metrics().cpu_memory.peak <= 250_000);
    drop(out);
    c.close();
    assert_eq!(c.metrics().cpu_memory.used, baseline);
}

#[test]
fn batch_snapshot_includes_masks_and_tail_and_repeated_execution() {
    let c = context(200_000);
    let n = 2051;
    let a = c.array(vec![n], Buffer::I64(vec![3; n])).unwrap();
    let m = c.array(vec![n], Buffer::Bool(vec![true; n])).unwrap();
    let p = Plan::Source(vec![Column {
        values: Expr::input(&a),
        valid: Some(Expr::input(&m)),
    }]);
    let budget = crate::memory::MemoryBudget::new(Device::Cpu, 200_000);
    let mut env = JobEnv::new(Device::Cpu, budget, CancellationToken::new());
    let frozen = streaming::Frozen::new(&p, &mut env).unwrap();
    write(&a, 99);
    write(&m, 0);
    let out = execute(&p, &mut env, &frozen).unwrap();
    assert_eq!(out.columns[0].values.buffer, Buffer::I64(vec![3; n]));
    assert_eq!(out.columns[0].valid.buffer, Buffer::Bool(vec![true; n]));
    let out = run(&c, p).unwrap();
    assert_eq!(out.columns[0].values.buffer.get(0), Scalar::Int(99));
    assert!(!out.columns[0].valid(0));
    assert_eq!(out.columns[0].values.buffer.get(n - 1), Scalar::Int(3));
}

#[test]
fn batched_rejection_and_cancellation_release_all_charges() {
    let c = context(100_000);
    let a = c.array(vec![4096], Buffer::I64(vec![1; 4096])).unwrap();
    let baseline = c.metrics().cpu_memory.used;
    // Snapshot and its offsets fit, but the completed result plus a batch do
    // not.
    let error = run(&c, source(&a)).err().unwrap();
    assert_eq!(error.code, ErrorCode::MemoryBudgetExceeded);
    c.close();
    assert_eq!(c.metrics().cpu_memory.used, baseline);

    let budget = crate::memory::MemoryBudget::new(Device::Cpu, 1_000_000);
    let token = CancellationToken::new();
    let mut env = JobEnv::new(Device::Cpu, budget.clone(), token.clone());
    let p = source(&a);
    let frozen = streaming::Frozen::new(&p, &mut env).unwrap();
    token.cancel();
    assert_eq!(
        execute(&p, &mut env, &frozen).err().unwrap().code,
        ErrorCode::Cancelled
    );
    drop(frozen);
    drop(env);
    assert_eq!(budget.usage().used, 0);
}

#[test]
fn hash_join_preserves_duplicate_nan_zero_and_null_order() {
    let c = context(100_000);
    let keys = c
        .array(vec![5], Buffer::F64(vec![f64::NAN, -0.0, 0.0, 5.0, 8.0]))
        .unwrap();
    let mask = c
        .array(vec![5], Buffer::Bool(vec![true, true, true, false, true]))
        .unwrap();
    let ids = c.array(vec![5], Buffer::I64(vec![0, 1, 2, 3, 4])).unwrap();
    let p = Plan::Source(vec![
        Column {
            values: Expr::input(&keys),
            valid: Some(Expr::input(&mask)),
        },
        Column {
            values: Expr::input(&ids),
            valid: None,
        },
    ]);
    for nulls in [true, false] {
        let out = run(
            &c,
            Plan::Join(Box::new(p.clone()), Box::new(p.clone()), 0, 0, true, nulls),
        )
        .unwrap();
        assert_eq!(out.rows, 7);
        assert_eq!(
            out.columns[1].values.buffer,
            Buffer::I64(vec![0, 1, 1, 2, 2, 3, 4])
        );
        assert_eq!(
            out.columns[3].values.buffer,
            Buffer::I64(vec![0, 1, 2, 1, 2, if nulls { 3 } else { 0 }, 4])
        );
        assert_eq!(out.columns[3].valid(5), nulls);
    }
    let out = run(
        &c,
        Plan::Group(Box::new(p), 0, 1, Aggregate::Sum, false, Flavor::Polars),
    )
    .unwrap();
    assert_eq!(out.columns[1].values.buffer, Buffer::I64(vec![0, 3, 3, 4]));
    assert!(!out.columns[0].valid(2));
}

#[test]
fn hash_groups_scale_with_distinct_keys_and_keep_first_seen_order() {
    let c = context(4_000_000);
    let n = 20_000;
    let a = c
        .array(vec![n], Buffer::I64((0..n as i64).rev().collect()))
        .unwrap();
    let out = run(
        &c,
        Plan::Group(
            Box::new(source(&a)),
            0,
            0,
            Aggregate::Count,
            false,
            Flavor::Polars,
        ),
    )
    .unwrap();
    assert_eq!(out.rows, n);
    assert_eq!(
        out.columns[0].values.buffer,
        Buffer::I64((0..n as i64).rev().collect())
    );
    assert_eq!(out.columns[1].values.buffer, Buffer::U32(vec![1; n]));
}

#[test]
fn batch_kernels_match_dense_broadcast_cast_and_strided_snapshots() {
    let c = context(1_000_000);
    let n = 2051;
    let a = c
        .array(vec![n], Buffer::I32((0..n as i32).collect()))
        .unwrap();
    let a = a
        .view(&[Selector::Slice {
            start: None,
            stop: None,
            step: Some(-1),
        }])
        .unwrap();
    let one = c.array(vec![1], Buffer::I32(vec![7])).unwrap();
    let sum = Expr::binary(crate::BinaryOp::Add, &Expr::input(&a), &Expr::input(&one)).unwrap();
    let cast = Expr::cast(&sum, DType::Float32).unwrap();
    let e = Expr::unary(crate::UnaryOp::Neg, &cast).unwrap();
    let p = Plan::Source(vec![Column {
        values: e.clone(),
        valid: None,
    }]);
    let budget = crate::memory::MemoryBudget::new(Device::Cpu, 1_000_000);
    let mut env = JobEnv::new(Device::Cpu, budget, CancellationToken::new());
    env.freeze(&e).unwrap();
    let reference = env.eval_dense(&e).unwrap();
    let out = run(&c, p).unwrap();
    assert_eq!(out.columns[0].values.buffer, reference.buffer);
}

#[test]
fn full_projection_result_fits_with_batches_but_dense_expression_does_not() {
    let c = context(250_000);
    let n = 8193;
    let a = c.array(vec![n], Buffer::I64(vec![1; n])).unwrap();
    let e = Expr::binary(
        crate::BinaryOp::Add,
        &Expr::input(&a),
        &Expr::scalar(Scalar::Int(10)),
    )
    .unwrap();
    let e = Expr::binary(crate::BinaryOp::Mul, &e, &Expr::scalar(Scalar::Int(2))).unwrap();
    let budget = crate::memory::MemoryBudget::new(Device::Cpu, 250_000);
    let _inputs = budget.reserve((n * 8) as u64).unwrap();
    let mut env = JobEnv::new(Device::Cpu, budget, CancellationToken::new());
    env.freeze(&e).unwrap();
    assert_eq!(
        env.eval_dense(&e).unwrap_err().code,
        ErrorCode::MemoryBudgetExceeded
    );
    let out = run(
        &c,
        Plan::Source(vec![Column {
            values: e,
            valid: None,
        }]),
    )
    .unwrap();
    assert_eq!(out.columns[0].values.buffer, Buffer::I64(vec![22; n]));
}

#[test]
fn mixed_materialization_boundaries_share_the_atomic_snapshot() {
    let c = context(100_000);
    let a = c.array(vec![3], Buffer::I64(vec![1, 2, 3])).unwrap();
    let indexed = Expr::index(
        &Expr::input(&a),
        vec![Selector::Slice {
            start: None,
            stop: None,
            step: Some(-1),
        }],
    )
    .unwrap();
    let p = Plan::Join(
        Box::new(source(&a)),
        Box::new(Plan::Source(vec![Column {
            values: indexed,
            valid: None,
        }])),
        0,
        0,
        false,
        false,
    );
    let budget = crate::memory::MemoryBudget::new(Device::Cpu, 100_000);
    let mut env = JobEnv::new(Device::Cpu, budget, CancellationToken::new());
    let frozen = streaming::Frozen::new(&p, &mut env).unwrap();
    write(&a, 99);
    let out = execute(&p, &mut env, &frozen).unwrap();
    assert_eq!(out.columns[0].values.buffer, Buffer::I64(vec![1, 2, 3]));
    assert_eq!(out.columns[1].values.buffer, Buffer::I64(vec![1, 2, 3]));
}

#[test]
fn empty_pipeline_preserves_scalar_expression_errors() {
    let c = context(10_000);
    let a = c.array(vec![0], Buffer::I64(vec![])).unwrap();
    let scalar = Expr::binary(
        crate::BinaryOp::Pow,
        &Expr::scalar(Scalar::Int(2)),
        &Expr::scalar(Scalar::Int(-1)),
    )
    .unwrap();
    let e = Expr::binary(crate::BinaryOp::Add, &Expr::input(&a), &scalar).unwrap();
    let budget = crate::memory::MemoryBudget::new(Device::Cpu, 10_000);
    let mut env = JobEnv::new(Device::Cpu, budget, CancellationToken::new());
    env.freeze(&e).unwrap();
    let expected = env.eval_dense(&e).unwrap_err().code;
    let actual = run(
        &c,
        Plan::Source(vec![Column {
            values: e,
            valid: None,
        }]),
    )
    .err()
    .unwrap()
    .code;
    assert_eq!(actual, expected);
}

#[test]
fn hash_index_budget_rejection_is_atomic() {
    let c = context(10_000);
    let a = c.array(vec![100], Buffer::I64((0..100).collect())).unwrap();
    let budget = crate::memory::MemoryBudget::new(Device::Cpu, 900);
    let env = JobEnv::new(Device::Cpu, budget.clone(), CancellationToken::new());
    let column = MaterializedColumn {
        values: a.snapshot(&budget).unwrap(),
        valid: all_valid(&env, 100).unwrap(),
    };
    assert_eq!(
        KeyIndex::new(&column, true, false, &env)
            .err()
            .unwrap()
            .code,
        ErrorCode::MemoryBudgetExceeded
    );
    assert_eq!(budget.usage().used, 900);
    drop(column);
    assert_eq!(budget.usage().used, 0);
}

fn pivot_column(env: &Context, values: Buffer, valid: Option<Vec<bool>>) -> Column {
    let values = c_array(env, values);
    let valid = valid.map(|mask| c_array(env, Buffer::Bool(mask)));
    Column {
        values: Expr::input(&values),
        valid: valid.as_ref().map(Expr::input),
    }
}

fn c_array(env: &Context, values: Buffer) -> crate::NdArray {
    let len = values.len();
    env.array(vec![len], values).unwrap()
}

fn pivot_plan(
    env: &Context,
    row: Buffer,
    column: Buffer,
    value: Buffer,
    mode: PivotMode,
    flavor: Flavor,
) -> Plan {
    Plan::Pivot(
        Box::new(Plan::Source(vec![
            pivot_column(env, row, None),
            pivot_column(env, column, None),
            pivot_column(env, value, None),
        ])),
        0,
        1,
        2,
        mode,
        flavor,
    )
}

#[test]
fn pivot_unique_and_aggregates() {
    let ctx = context(1 << 20);
    let unique = pivot_plan(
        &ctx,
        Buffer::I64(vec![1, 1, 2]),
        Buffer::I64(vec![10, 20, 10]),
        Buffer::I64(vec![5, 6, 7]),
        PivotMode::Unique,
        Flavor::Pandas,
    );
    let table = run(&ctx, unique).unwrap();
    assert_eq!(table.rows, 2);
    assert_eq!(
        table.pivot_columns.as_deref(),
        Some(&["10".to_string(), "20".to_string()][..])
    );
    assert_eq!(table.columns[0].values.buffer, Buffer::I64(vec![1, 2]));
    assert_eq!(table.columns[1].values.buffer, Buffer::I64(vec![5, 7]));
    assert_eq!(
        table.columns[1].valid.buffer,
        Buffer::Bool(vec![true, true])
    );
    assert_eq!(table.columns[2].values.buffer, Buffer::I64(vec![6, 0]));
    assert_eq!(
        table.columns[2].valid.buffer,
        Buffer::Bool(vec![true, false])
    );

    let duplicate = pivot_plan(
        &ctx,
        Buffer::I64(vec![1, 1]),
        Buffer::I64(vec![10, 10]),
        Buffer::I64(vec![1, 2]),
        PivotMode::Unique,
        Flavor::Pandas,
    );
    assert_eq!(
        run(&ctx, duplicate).err().unwrap().code,
        ErrorCode::IndexInvalid
    );

    let summed = pivot_plan(
        &ctx,
        Buffer::I64(vec![1, 1, 2, 2]),
        Buffer::I64(vec![10, 10, 10, 20]),
        Buffer::I64(vec![1, 2, 3, 4]),
        PivotMode::Sum,
        Flavor::Pandas,
    );
    let table = run(&ctx, summed).unwrap();
    assert_eq!(table.columns[1].values.buffer, Buffer::I64(vec![3, 3]));
    assert_eq!(table.columns[2].values.buffer, Buffer::I64(vec![0, 4]));
    assert_eq!(
        table.columns[2].valid.buffer,
        Buffer::Bool(vec![false, true])
    );

    let mean = pivot_plan(
        &ctx,
        Buffer::I64(vec![1, 1]),
        Buffer::I64(vec![10, 10]),
        Buffer::I64(vec![2, 4]),
        PivotMode::Mean,
        Flavor::Polars,
    );
    let table = run(&ctx, mean).unwrap();
    assert_eq!(table.columns[1].values.buffer, Buffer::F64(vec![3.0]));

    let count = pivot_plan(
        &ctx,
        Buffer::I64(vec![1, 1]),
        Buffer::I64(vec![10, 10]),
        Buffer::I64(vec![2, 4]),
        PivotMode::Count,
        Flavor::Polars,
    );
    let table = run(&ctx, count).unwrap();
    assert_eq!(table.columns[1].values.buffer, Buffer::U32(vec![2]));
}

#[test]
fn pivot_keys_nulls_and_validation() {
    let ctx = context(1 << 20);
    let source = Plan::Source(vec![
        pivot_column(
            &ctx,
            Buffer::I64(vec![1, 2, 3]),
            Some(vec![true, false, true]),
        ),
        pivot_column(&ctx, Buffer::I64(vec![10, 10, 20]), None),
        pivot_column(&ctx, Buffer::I64(vec![1, 2, 3]), None),
    ]);
    let plan = Plan::Pivot(Box::new(source), 0, 1, 2, PivotMode::Sum, Flavor::Polars);
    let table = run(&ctx, plan).unwrap();
    assert_eq!(table.rows, 2);
    assert_eq!(table.columns[0].values.buffer, Buffer::I64(vec![1, 3]));

    let bad_value = Plan::Pivot(
        Box::new(Plan::Source(vec![
            pivot_column(&ctx, Buffer::I64(vec![1]), None),
            pivot_column(&ctx, Buffer::I64(vec![10]), None),
            pivot_column(&ctx, Buffer::C64(vec![[1.0, 0.0]]), None),
        ])),
        0,
        1,
        2,
        PivotMode::Sum,
        Flavor::Pandas,
    );
    assert_eq!(
        run(&ctx, bad_value).err().unwrap().code,
        ErrorCode::DTypeUnsupported
    );

    let empty = pivot_plan(
        &ctx,
        Buffer::I64(vec![]),
        Buffer::I64(vec![]),
        Buffer::I64(vec![]),
        PivotMode::Unique,
        Flavor::Polars,
    );
    let table = run(&ctx, empty).unwrap();
    assert_eq!(table.rows, 0);
    assert_eq!(table.columns.len(), 1);
    assert_eq!(table.pivot_columns.as_deref(), Some(&[][..]));
}
