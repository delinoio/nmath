use std::{sync::mpsc, time::Duration};

use nmath_core::{
    BinaryOp, Buffer, CancellationToken, Context, ContextOptions, DType, Device, ErrorCode,
    ExecutionStatus, Expr, NdArray, Result, Scalar, Selector, Value, context::JobOutput,
    io::WriteMode, layout::CopyPolicy,
};

fn ctx(jobs: usize, queue: usize) -> Context {
    Context::new(ContextOptions {
        max_concurrent_jobs: jobs,
        max_queue_length: queue,
        cpu_memory_budget: 1 << 30,
        gpu_memory_budget: 0,
    })
    .unwrap()
}

type Outcome = mpsc::Receiver<Result<JobOutput>>;

fn channel() -> (Box<dyn FnOnce(Result<JobOutput>) + Send>, Outcome) {
    let (tx, rx) = mpsc::channel();
    (Box::new(move |r| tx.send(r).unwrap()), rx)
}

fn dense_values(out: JobOutput) -> Buffer {
    match out {
        JobOutput::Value(Value::Dense(d)) => d.buffer,
        JobOutput::Value(Value::View(v)) | JobOutput::Array(v) => v.to_buffer(),
        JobOutput::Unit => panic!("expected a value"),
    }
}

fn wait(rx: &Outcome) -> Result<JobOutput> {
    rx.recv_timeout(Duration::from_secs(10))
        .expect("job did not finish")
}

fn compute_view(context: &Context, expr: &Expr) -> NdArray {
    let (done, rx) = channel();
    context
        .compute(expr, Device::Cpu, CancellationToken::new(), done)
        .unwrap();
    let JobOutput::Value(Value::View(view)) = wait(&rx).unwrap() else {
        panic!("expected a layout view")
    };
    view
}

/// A job that occupies a slot until the test drops `_turn`: an `assign` on a
/// private storage whose write line the test holds.
struct Blocker {
    _turn: nmath_core::sync::FifoGuard,
    rx: Outcome,
}

fn occupy(context: &Context) -> Blocker {
    let target = context.array(vec![1], Buffer::F64(vec![0.0])).unwrap();
    let turn = target
        .write_ticket()
        .wait(&CancellationToken::new())
        .unwrap();
    let (done, rx) = channel();
    context
        .assign(
            &target,
            vec![],
            &Expr::scalar(Scalar::Float(1.0)),
            CancellationToken::new(),
            done,
        )
        .unwrap();
    Blocker { _turn: turn, rx }
}

#[test]
fn rejects_zero_concurrency() {
    let err = Context::new(ContextOptions {
        max_concurrent_jobs: 0,
        max_queue_length: 0,
        cpu_memory_budget: 0,
        gpu_memory_budget: 0,
    })
    .err()
    .unwrap();
    assert_eq!(err.code, ErrorCode::ContextInvalidOptions);
}

#[test]
fn computes_expressions_off_thread() {
    let c = ctx(2, 2);
    let a = c.array(vec![3], Buffer::F64(vec![1.0, 2.0, 3.0])).unwrap();
    let e = Expr::binary(
        BinaryOp::Mul,
        &Expr::input(&a),
        &Expr::scalar(Scalar::Float(2.0)),
    )
    .unwrap();
    let (done, rx) = channel();
    c.compute(&e, Device::Cpu, CancellationToken::new(), done)
        .unwrap();
    assert_eq!(
        dense_values(wait(&rx).unwrap()),
        Buffer::F64(vec![2.0, 4.0, 6.0])
    );
}

#[test]
fn queue_is_fifo_and_bounded() {
    let c = ctx(1, 2);
    let blocker = occupy(&c);
    let a = c.array(vec![1], Buffer::I64(vec![1])).unwrap();
    let mut receivers = Vec::new();
    for _ in 0..2 {
        let (done, rx) = channel();
        c.compute(
            &Expr::input(&a),
            Device::Cpu,
            CancellationToken::new(),
            done,
        )
        .unwrap();
        receivers.push(rx);
    }
    let (done, _rx) = channel();
    let err = c
        .compute(
            &Expr::input(&a),
            Device::Cpu,
            CancellationToken::new(),
            done,
        )
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::SchedQueueFull);
    assert_eq!(c.metrics().total_rejected, 1);
    drop(blocker._turn);
    wait(&blocker.rx).unwrap();
    for rx in &receivers {
        wait(rx).unwrap();
    }
    let ids: Vec<u64> = c
        .metrics()
        .executions
        .iter()
        .filter(|r| r.kind == nmath_core::OperationKind::Compute)
        .map(|r| r.id)
        .collect();
    assert_eq!(ids.len(), 2);
    assert!(ids[0] < ids[1], "queued jobs finish in FIFO order");
}

#[test]
fn queued_jobs_see_changes_until_they_start() {
    let c = ctx(1, 4);
    let blocker = occupy(&c);
    let a = c.array(vec![2], Buffer::F64(vec![1.0, 1.0])).unwrap();
    let (done, rx) = channel();
    c.compute(
        &Expr::input(&a),
        Device::Cpu,
        CancellationToken::new(),
        done,
    )
    .unwrap();
    // Write directly while the compute job is queued behind the blocker.
    let token = CancellationToken::new();
    let turn = a.write_ticket().wait(&token).unwrap();
    let five = nmath_core::DenseArray::new(
        vec![],
        Buffer::F64(vec![5.0]),
        nmath_core::memory::Reservation::none(),
    )
    .unwrap();
    a.assign(&turn, &[Selector::Index(0)], &five, &token)
        .unwrap();
    drop(turn);
    drop(blocker._turn);
    wait(&blocker.rx).unwrap();
    assert_eq!(
        dense_values(wait(&rx).unwrap()),
        Buffer::F64(vec![5.0, 1.0])
    );
}

#[test]
fn cancelling_a_queued_job_never_runs_it() {
    let c = ctx(1, 4);
    let blocker = occupy(&c);
    let a = c.array(vec![1], Buffer::F64(vec![1.0])).unwrap();
    let token = CancellationToken::new();
    let (done, rx) = channel();
    c.compute(&Expr::input(&a), Device::Cpu, token.clone(), done)
        .unwrap();
    token.cancel();
    assert_eq!(wait(&rx).unwrap_err().code, ErrorCode::Cancelled);
    drop(blocker._turn);
    wait(&blocker.rx).unwrap();
    let m = c.metrics();
    assert_eq!(m.total_cancelled, 1);
    assert!(
        m.executions
            .iter()
            .any(|r| r.status == ExecutionStatus::Cancelled)
    );
}

#[test]
fn already_cancelled_signal_rejects_before_admission() {
    let c = ctx(1, 1);
    let a = c.array(vec![1], Buffer::F64(vec![1.0])).unwrap();
    let token = CancellationToken::new();
    token.cancel();
    let (done, _rx) = channel();
    assert_eq!(
        c.compute(&Expr::input(&a), Device::Cpu, token, done)
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
}

#[test]
fn close_cancels_queued_and_running_work() {
    let c = ctx(1, 4);
    let a = c.array(vec![1], Buffer::F64(vec![1.0])).unwrap();
    let held = a.write_ticket().wait(&CancellationToken::new()).unwrap();
    let (rdone, running) = channel();
    c.assign(
        &a,
        vec![],
        &Expr::scalar(Scalar::Float(2.0)),
        CancellationToken::new(),
        rdone,
    )
    .unwrap();
    let (qdone, queued) = channel();
    c.compute(
        &Expr::input(&a),
        Device::Cpu,
        CancellationToken::new(),
        qdone,
    )
    .unwrap();
    c.close();
    drop(held);
    assert_eq!(wait(&queued).unwrap_err().code, ErrorCode::Cancelled);
    assert_eq!(wait(&running).unwrap_err().code, ErrorCode::Cancelled);
    assert_eq!(
        a.to_buffer(),
        Buffer::F64(vec![1.0]),
        "cancelled write leaves target unchanged"
    );
    let (done, _rx) = channel();
    assert_eq!(
        c.compute(
            &Expr::input(&a),
            Device::Cpu,
            CancellationToken::new(),
            done
        )
        .unwrap_err()
        .code,
        ErrorCode::ContextClosed
    );
    assert_eq!(
        c.array(vec![0], Buffer::F64(vec![])).unwrap_err().code,
        ErrorCode::ContextClosed
    );
}

#[test]
fn memory_budget_rejects_and_releases() {
    let c = Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 1,
        cpu_memory_budget: 64,
        gpu_memory_budget: 0,
    })
    .unwrap();
    let a = c.array(vec![4], Buffer::F64(vec![1.0; 4])).unwrap();
    assert_eq!(
        c.array(vec![8], Buffer::F64(vec![0.0; 8]))
            .unwrap_err()
            .code,
        ErrorCode::MemoryBudgetExceeded
    );
    // Snapshot (32) + copy (32) + result (32) exceeds the remaining 32 bytes.
    let e = Expr::binary(BinaryOp::Add, &Expr::input(&a), &Expr::input(&a)).unwrap();
    let (done, rx) = channel();
    c.compute(&e, Device::Cpu, CancellationToken::new(), done)
        .unwrap();
    assert_eq!(wait(&rx).unwrap_err().code, ErrorCode::MemoryBudgetExceeded);
    assert_eq!(
        c.metrics().cpu_memory.used,
        32,
        "only the input stays charged"
    );
}

#[test]
fn execution_snapshots_charge_only_payload_and_copy_strided_values() {
    let tiny = Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 1,
        cpu_memory_budget: 3,
        gpu_memory_budget: 0,
    })
    .unwrap();
    let input = tiny.array(vec![1], Buffer::U8(vec![7])).unwrap();
    let (done, rx) = channel();
    tiny.compute(
        &Expr::input(&input),
        Device::Cpu,
        CancellationToken::new(),
        done,
    )
    .unwrap();
    assert_eq!(dense_values(wait(&rx).unwrap()), Buffer::U8(vec![7]));
    assert_eq!(tiny.metrics().cpu_memory.peak, 3);
    assert_eq!(tiny.metrics().cpu_memory.used, 1);

    let source = ctx(1, 1);
    let input = source
        .array(vec![6], Buffer::U8(vec![0, 1, 2, 3, 4, 5]))
        .unwrap();
    let view = input
        .view(&[Selector::Slice {
            start: None,
            stop: None,
            step: Some(-2),
        }])
        .unwrap();
    let target = Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 1,
        cpu_memory_budget: 2,
        gpu_memory_budget: 0,
    })
    .unwrap();
    let (done, rx) = channel();
    target
        .copy_from(&view, CancellationToken::new(), done)
        .unwrap();
    assert_eq!(wait(&rx).unwrap_err().code, ErrorCode::MemoryBudgetExceeded);
    assert_eq!(target.metrics().cpu_memory.used, 0);
    assert_eq!(target.metrics().cpu_memory.rejected, 1);

    let target = Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 1,
        cpu_memory_budget: 3,
        gpu_memory_budget: 0,
    })
    .unwrap();
    let (done, rx) = channel();
    target
        .copy_from(&view, CancellationToken::new(), done)
        .unwrap();
    let JobOutput::Array(copy) = wait(&rx).unwrap() else {
        panic!("copy did not return an array")
    };
    assert_eq!(copy.to_buffer(), Buffer::U8(vec![5, 3, 1]));
    assert_eq!(target.metrics().cpu_memory.used, 3);
    assert_eq!(target.metrics().cpu_memory.peak, 3);
    drop(copy);
    assert_eq!(target.metrics().cpu_memory.used, 0);
}

#[test]
fn cross_context_values_need_explicit_copy() {
    let a_ctx = ctx(1, 1);
    let b_ctx = ctx(1, 1);
    let a = a_ctx.array(vec![2], Buffer::I32(vec![1, 2])).unwrap();
    let (done, _rx) = channel();
    assert_eq!(
        b_ctx
            .compute(
                &Expr::input(&a),
                Device::Cpu,
                CancellationToken::new(),
                done
            )
            .unwrap_err()
            .code,
        ErrorCode::ContextMismatch
    );
    let (done, rx) = channel();
    b_ctx.copy_from(&a, CancellationToken::new(), done).unwrap();
    let JobOutput::Array(copy) = wait(&rx).unwrap() else {
        panic!()
    };
    assert_eq!(copy.context_id(), b_ctx.id());
    assert!(!copy.shares_storage(&a));
    assert_eq!(b_ctx.metrics().cpu_memory.used, 8);
}

#[test]
fn scalar_layout_results_retain_the_executing_context() {
    let context = ctx(1, 2);
    let other = ctx(1, 2);
    for policy in [CopyPolicy::IfNeeded, CopyPolicy::Always, CopyPolicy::Never] {
        let expr = Expr::reshape(&Expr::scalar(Scalar::Int(1)), &[1], policy).unwrap();
        let view = compute_view(&context, &expr);
        assert_eq!(view.context_id(), context.id());
        assert_eq!(view.shape(), &[1]);
        assert_eq!(view.dtype(), DType::Int64);
        assert_eq!(view.to_buffer(), Buffer::I64(vec![1]));

        let (done, rx) = channel();
        context
            .compute(
                &Expr::input(&view),
                Device::Cpu,
                CancellationToken::new(),
                done,
            )
            .unwrap();
        assert_eq!(dense_values(wait(&rx).unwrap()), Buffer::I64(vec![1]));

        let target = context.array(vec![1], Buffer::I64(vec![0])).unwrap();
        let (done, rx) = channel();
        context
            .assign(
                &target,
                vec![],
                &Expr::input(&view),
                CancellationToken::new(),
                done,
            )
            .unwrap();
        wait(&rx).unwrap();
        assert_eq!(target.to_buffer(), Buffer::I64(vec![1]));

        let (done, rx) = channel();
        context
            .assign(
                &view,
                vec![],
                &Expr::scalar(Scalar::Int(4)),
                CancellationToken::new(),
                done,
            )
            .unwrap();
        wait(&rx).unwrap();
        assert_eq!(view.to_buffer(), Buffer::I64(vec![4]));

        let (done, _rx) = channel();
        assert_eq!(
            other
                .compute(
                    &Expr::input(&view),
                    Device::Cpu,
                    CancellationToken::new(),
                    done,
                )
                .unwrap_err()
                .code,
            ErrorCode::ContextMismatch
        );
        let (done, _rx) = channel();
        assert_eq!(
            other
                .assign(
                    &view,
                    vec![],
                    &Expr::scalar(Scalar::Int(8)),
                    CancellationToken::new(),
                    done,
                )
                .unwrap_err()
                .code,
            ErrorCode::ContextMismatch
        );
        let foreign_target = other.array(vec![1], Buffer::I64(vec![0])).unwrap();
        let (done, _rx) = channel();
        assert_eq!(
            other
                .assign(
                    &foreign_target,
                    vec![],
                    &Expr::input(&view),
                    CancellationToken::new(),
                    done,
                )
                .unwrap_err()
                .code,
            ErrorCode::ContextMismatch
        );
        assert_eq!(view.to_buffer(), Buffer::I64(vec![4]));
        assert_eq!(foreign_target.to_buffer(), Buffer::I64(vec![0]));
    }

    let transposed = compute_view(
        &context,
        &Expr::transpose(&Expr::scalar(Scalar::Int(2)), None).unwrap(),
    );
    assert_eq!(transposed.context_id(), context.id());
    assert!(transposed.shape().is_empty());
    assert_eq!(transposed.dtype(), DType::Int64);
    assert_eq!(transposed.to_buffer(), Buffer::I64(vec![2]));

    let sum = Expr::binary(
        BinaryOp::Add,
        &Expr::scalar(Scalar::Int(2)),
        &Expr::scalar(Scalar::Int(3)),
    )
    .unwrap();
    let arithmetic_layout = Expr::reshape(&sum, &[1], CopyPolicy::IfNeeded).unwrap();
    let arithmetic = compute_view(&context, &arithmetic_layout);
    assert_eq!(arithmetic.context_id(), context.id());
    assert_eq!(arithmetic.shape(), &[1]);
    assert_eq!(arithmetic.dtype(), DType::Int64);
    assert_eq!(arithmetic.to_buffer(), Buffer::I64(vec![5]));
    let (done, rx) = channel();
    context
        .compute(
            &Expr::input(&arithmetic),
            Device::Cpu,
            CancellationToken::new(),
            done,
        )
        .unwrap();
    assert_eq!(dense_values(wait(&rx).unwrap()), Buffer::I64(vec![5]));

    drop(transposed);
    drop(arithmetic);
    assert_eq!(context.metrics().cpu_memory.used, 0);
    assert_eq!(other.metrics().cpu_memory.used, 0);
}

#[test]
fn random_state_commits_only_on_success() {
    let c = ctx(1, 4);
    let g = c.generator(42);
    let before = g.current();
    let e = Expr::random(c.id(), &g, vec![3]);
    let (done, rx) = channel();
    c.compute(&e, Device::Cpu, CancellationToken::new(), done)
        .unwrap();
    // NumPy 2.5.4: default_rng(42).random(3)
    assert_eq!(
        dense_values(wait(&rx).unwrap()),
        Buffer::F64(vec![
            0.7739560485559633,
            0.4388784397520523,
            0.8585979199113825
        ])
    );
    let after_first = g.current();
    assert_ne!(before, after_first);
    // Re-executing the same expression continues from the committed state.
    let (done, rx) = channel();
    c.compute(&e, Device::Cpu, CancellationToken::new(), done)
        .unwrap();
    let second = dense_values(wait(&rx).unwrap());
    assert_ne!(
        second,
        Buffer::F64(vec![
            0.7739560485559633,
            0.4388784397520523,
            0.8585979199113825
        ])
    );
    // A failing job leaves the state unchanged.
    let tiny = Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 0,
        cpu_memory_budget: 8,
        gpu_memory_budget: 0,
    })
    .unwrap();
    let committed = g.current();
    let (done, rx) = channel();
    tiny.compute(
        &Expr::random(tiny.id(), &g, vec![3]),
        Device::Cpu,
        CancellationToken::new(),
        done,
    )
    .unwrap();
    assert!(wait(&rx).is_err());
    assert_eq!(g.current(), committed);
}

#[test]
fn gpu_requests_fail_without_fallback() {
    let c = ctx(1, 1);
    let f64a = c.array(vec![1], Buffer::F64(vec![1.0])).unwrap();
    let (done, _rx) = channel();
    let err = c
        .compute(
            &Expr::input(&f64a),
            Device::Gpu,
            CancellationToken::new(),
            done,
        )
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::GpuUnsupportedOperation);
    let f32a = c.array(vec![1], Buffer::F32(vec![1.0])).unwrap();
    let (done, rx) = channel();
    c.compute(
        &Expr::input(&f32a),
        Device::Gpu,
        CancellationToken::new(),
        done,
    )
    .unwrap();
    // This context has no device budget. A host without a usable adapter
    // reports discovery failure; otherwise allocation must fail without
    // fallback.
    assert!(matches!(
        wait(&rx).unwrap_err().code,
        ErrorCode::GpuUnavailable | ErrorCode::MemoryBudgetExceeded
    ));
}

#[test]
fn basic_index_results_are_shared_views() {
    let c = ctx(1, 1);
    let a = c
        .array(vec![2, 2], Buffer::F64(vec![1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let e = Expr::index(&Expr::input(&a), vec![Selector::Index(1)]).unwrap();
    let (done, rx) = channel();
    c.compute(&e, Device::Cpu, CancellationToken::new(), done)
        .unwrap();
    let JobOutput::Value(Value::View(row)) = wait(&rx).unwrap() else {
        panic!("expected a view")
    };
    assert!(row.shares_storage(&a));
    let (done, rx) = channel();
    c.assign(
        &a,
        vec![Selector::Index(1), Selector::Index(1)],
        &Expr::scalar(Scalar::Float(9.0)),
        CancellationToken::new(),
        done,
    )
    .unwrap();
    wait(&rx).unwrap();
    assert_eq!(row.to_buffer(), Buffer::F64(vec![3.0, 9.0]));
}

#[test]
fn writes_serialize_in_request_order() {
    let c = ctx(4, 4);
    let a = c.array(vec![1], Buffer::I64(vec![0])).unwrap();
    let held = a.write_ticket().wait(&CancellationToken::new()).unwrap();
    let mut receivers = Vec::new();
    for v in 1..=3 {
        let (done, rx) = channel();
        c.assign(
            &a,
            vec![],
            &Expr::scalar(Scalar::Int(v)),
            CancellationToken::new(),
            done,
        )
        .unwrap();
        receivers.push(rx);
    }
    drop(held);
    for rx in &receivers {
        wait(rx).unwrap();
    }
    assert_eq!(a.to_buffer(), Buffer::I64(vec![3]));
}

#[test]
fn npy_jobs_round_trip_atomically() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.npy");
    let c = ctx(1, 1);
    let a = c
        .array(vec![2, 2], Buffer::F32(vec![1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let (done, rx) = channel();
    c.save_npy(
        &a,
        path.clone(),
        WriteMode::CreateNew,
        CancellationToken::new(),
        done,
    )
    .unwrap();
    wait(&rx).unwrap();
    let (done, rx) = channel();
    c.save_npy(
        &a,
        path.clone(),
        WriteMode::CreateNew,
        CancellationToken::new(),
        done,
    )
    .unwrap();
    assert_eq!(wait(&rx).unwrap_err().code, ErrorCode::IoExists);
    let (done, rx) = channel();
    c.load_npy(path, CancellationToken::new(), done).unwrap();
    let JobOutput::Array(loaded) = wait(&rx).unwrap() else {
        panic!()
    };
    assert_eq!(loaded.shape(), &[2, 2]);
    assert_eq!(loaded.to_buffer(), Buffer::F32(vec![1.0, 2.0, 3.0, 4.0]));
}

#[test]
fn empty_storage_rejects_unrepresentable_strides_without_losing_the_context() {
    let c = Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 0,
        cpu_memory_budget: 0,
        gpu_memory_budget: 0,
    })
    .unwrap();
    let wide = u32::MAX as usize;
    for shape in [
        vec![0, wide, wide],
        vec![wide, 0, wide],
        vec![wide, wide, 0],
    ] {
        assert_eq!(
            c.array(shape, Buffer::F64(vec![])).unwrap_err().code,
            ErrorCode::ShapeMismatch
        );
        assert_eq!(c.metrics().cpu_memory.used, 0);
        assert_eq!(c.metrics().total_submitted, 0);
    }
    let empty = c.array(vec![0, 3, 4], Buffer::F64(vec![])).unwrap();
    assert_eq!(empty.shape(), &[0, 3, 4]);
    assert_eq!(empty.to_buffer(), Buffer::F64(vec![]));
    assert_eq!(c.metrics().cpu_memory.used, 0);
    c.close();

    let c = ctx(1, 0);
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let ordinary = c.array(vec![2, 3], Buffer::F64(values.clone())).unwrap();
    assert_eq!(ordinary.shape(), &[2, 3]);
    assert_eq!(ordinary.to_buffer(), Buffer::F64(values));
    assert_eq!(ordinary.dtype(), nmath_core::DType::Float64);
    drop(ordinary);
    assert_eq!(c.metrics().cpu_memory.used, 0);
    c.close();
}

#[test]
fn generated_and_loaded_empty_storage_rejects_unrepresentable_strides() {
    let c = ctx(1, 0);
    let wide = u32::MAX as usize;
    let generator = c.generator(42);
    let original = generator.current();
    let expr = Expr::random(c.id(), &generator, vec![0, wide, wide]);
    let (done, rx) = channel();
    c.compute(&expr, Device::Cpu, CancellationToken::new(), done)
        .unwrap();
    assert_eq!(wait(&rx).unwrap_err().code, ErrorCode::ShapeMismatch);
    assert_eq!(generator.current(), original);
    assert_eq!(c.metrics().cpu_memory.used, 0);

    let dir = tempfile::tempdir().unwrap();
    for (index, shape) in [vec![0, wide, wide], vec![wide, wide, 0]]
        .into_iter()
        .enumerate()
    {
        let path = dir.path().join(format!("empty-{index}.npy"));
        std::fs::write(
            &path,
            nmath_core::io::encode_npy(&shape, &Buffer::F64(vec![])).unwrap(),
        )
        .unwrap();
        let (done, rx) = channel();
        c.load_npy(path, CancellationToken::new(), done).unwrap();
        assert_eq!(wait(&rx).unwrap_err().code, ErrorCode::ShapeMismatch);
        assert_eq!(c.metrics().cpu_memory.used, 0);
    }
    let valid = c.array(vec![0, 3, 4], Buffer::F64(vec![])).unwrap();
    assert_eq!(valid.shape(), &[0, 3, 4]);
    c.close();
}

#[test]
fn close_does_not_deadlock_on_jobs_started_from_the_queue() {
    let c = ctx(1, 4);
    let blocker = occupy(&c);
    // Queued behind the blocker, so its token keeps the queued-cancel callback
    // after it starts.
    let a = c.array(vec![1], Buffer::F64(vec![1.0])).unwrap();
    let held = a.write_ticket().wait(&CancellationToken::new()).unwrap();
    let (done, queued) = channel();
    c.assign(
        &a,
        vec![],
        &Expr::scalar(Scalar::Float(2.0)),
        CancellationToken::new(),
        done,
    )
    .unwrap();
    drop(blocker._turn);
    wait(&blocker.rx).unwrap();
    // Wait until the queued assignment is running and blocked on `held`.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !c
        .metrics()
        .executions
        .iter()
        .any(|r| r.kind == nmath_core::OperationKind::Write && r.status == ExecutionStatus::Running)
    {
        assert!(std::time::Instant::now() < deadline, "job did not start");
        std::thread::sleep(Duration::from_millis(5));
    }
    let (closed_tx, closed_rx) = mpsc::channel();
    let closer = c.clone();
    std::thread::spawn(move || {
        closer.close();
        closed_tx.send(()).unwrap();
    });
    closed_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("close() deadlocked");
    drop(held);
    assert_eq!(wait(&queued).unwrap_err().code, ErrorCode::Cancelled);
}

#[test]
fn assigning_a_view_valued_expression_copies_its_values() {
    let c = ctx(1, 1);
    let source = c
        .array(vec![2, 2], Buffer::F64(vec![1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let target = c.array(vec![2], Buffer::F64(vec![0.0, 0.0])).unwrap();
    let value = Expr::index(&Expr::input(&source), vec![Selector::Index(1)]).unwrap();
    let (done, rx) = channel();
    c.assign(&target, vec![], &value, CancellationToken::new(), done)
        .unwrap();
    wait(&rx).unwrap();
    assert_eq!(target.to_buffer(), Buffer::F64(vec![3.0, 4.0]));
    assert!(!target.shares_storage(&source));
}

#[test]
fn assignment_charges_conversion_and_selection_scratch() {
    let make_context = |budget| {
        Context::new(ContextOptions {
            max_concurrent_jobs: 1,
            max_queue_length: 0,
            cpu_memory_budget: budget,
            gpu_memory_budget: 0,
        })
        .unwrap()
    };
    let write = |context: &Context, target: &nmath_core::NdArray| {
        let (done, rx) = channel();
        context
            .assign(
                target,
                vec![],
                &Expr::scalar(Scalar::Float(1.0)),
                CancellationToken::new(),
                done,
            )
            .unwrap();
        wait(&rx)
    };

    let limited = make_context(40);
    let target = limited.array(vec![4], Buffer::F64(vec![0.0; 4])).unwrap();
    assert_eq!(
        write(&limited, &target).unwrap_err().code,
        ErrorCode::MemoryBudgetExceeded
    );
    assert_eq!(target.to_buffer(), Buffer::F64(vec![0.0; 4]));
    assert_eq!(limited.metrics().cpu_memory.used, 32);
    assert_eq!(limited.metrics().cpu_memory.peak, 40);

    let ample = make_context(256);
    let target = ample.array(vec![4], Buffer::F64(vec![0.0; 4])).unwrap();
    write(&ample, &target).unwrap();
    assert_eq!(target.to_buffer(), Buffer::F64(vec![1.0; 4]));
    assert_eq!(ample.metrics().cpu_memory.used, 32);
    assert_eq!(ample.metrics().cpu_memory.peak, 152);
    assert_eq!(ample.metrics().cpu_memory.cumulative_reserved, 200);
}

#[test]
fn numeric_assignment_rejects_before_cast_and_preserves_values() {
    let make_context = |budget| {
        Context::new(ContextOptions {
            max_concurrent_jobs: 1,
            max_queue_length: 0,
            cpu_memory_budget: budget,
            gpu_memory_budget: 0,
        })
        .unwrap()
    };
    let write = |context: &Context| {
        let target = context.array(vec![4], Buffer::I8(vec![0; 4])).unwrap();
        let source = context
            .array(vec![4], Buffer::F64(vec![1.9, -2.2, 3.0, 4.8]))
            .unwrap();
        let (done, rx) = channel();
        context
            .assign(
                &target,
                vec![],
                &Expr::input(&source),
                CancellationToken::new(),
                done,
            )
            .unwrap();
        (target, source, wait(&rx))
    };

    let limited = make_context(100);
    let (target, _source, outcome) = write(&limited);
    assert_eq!(outcome.unwrap_err().code, ErrorCode::MemoryBudgetExceeded);
    assert_eq!(target.to_buffer(), Buffer::I8(vec![0; 4]));
    assert_eq!(limited.metrics().cpu_memory.used, 36);
    assert_eq!(limited.metrics().cpu_memory.peak, 100);

    let ample = make_context(256);
    let (target, _source, outcome) = write(&ample);
    outcome.unwrap();
    assert_eq!(target.to_buffer(), Buffer::I8(vec![1, -2, 3, 4]));
    assert_eq!(ample.metrics().cpu_memory.used, 36);
    assert!(ample.metrics().cpu_memory.peak > 100);
}

#[test]
fn special_assignment_keeps_a_single_conversion_charge() {
    let context = Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 0,
        cpu_memory_budget: 100,
        gpu_memory_budget: 0,
    })
    .unwrap();
    let target = context
        .array(
            vec![2],
            Buffer::Unicode {
                width: 2,
                values: vec![0; 4],
            },
        )
        .unwrap();
    let (done, rx) = channel();
    context
        .assign(
            &target,
            vec![],
            &Expr::scalar(Scalar::Text("a".into())),
            CancellationToken::new(),
            done,
        )
        .unwrap();
    wait(&rx).unwrap();
    assert_eq!(
        target.to_buffer(),
        Buffer::Unicode {
            width: 2,
            values: vec![97, 0, 97, 0]
        }
    );
    assert_eq!(context.metrics().cpu_memory.used, 16);
    assert_eq!(context.metrics().cpu_memory.peak, 100);

    let dtype = nmath_core::DType::Datetime64(nmath_core::dtype::TimeUnit::NANOSECOND);
    let context = Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 0,
        cpu_memory_budget: 104,
        gpu_memory_budget: 0,
    })
    .unwrap();
    let target = context
        .array(
            vec![2],
            Buffer::Temporal {
                dtype,
                ticks: vec![0; 2],
            },
        )
        .unwrap();
    let (done, rx) = channel();
    context
        .assign(
            &target,
            vec![],
            &Expr::scalar(Scalar::Temporal(7, dtype)),
            CancellationToken::new(),
            done,
        )
        .unwrap();
    wait(&rx).unwrap();
    assert_eq!(
        target.to_buffer(),
        Buffer::Temporal {
            dtype,
            ticks: vec![7; 2]
        }
    );
    assert_eq!(context.metrics().cpu_memory.used, 16);
    assert_eq!(context.metrics().cpu_memory.peak, 104);
}

#[test]
fn assignment_index_scratch_failure_preserves_later_writes_and_order() {
    let context = Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 0,
        cpu_memory_budget: 256,
        gpu_memory_budget: 0,
    })
    .unwrap();
    let target = context.array(vec![5], Buffer::F64(vec![0.0; 5])).unwrap();
    let scalar = Expr::scalar(Scalar::Float(3.0));
    let (done, rx) = channel();
    context
        .assign(
            &target,
            vec![Selector::Indices(vec![0; 16])],
            &scalar,
            CancellationToken::new(),
            done,
        )
        .unwrap();
    assert_eq!(wait(&rx).unwrap_err().code, ErrorCode::MemoryBudgetExceeded);
    assert_eq!(target.to_buffer(), Buffer::F64(vec![0.0; 5]));
    assert_eq!(context.metrics().cpu_memory.used, 40);

    let (done, rx) = channel();
    context
        .assign(
            &target,
            vec![Selector::Slice {
                start: None,
                stop: None,
                step: Some(-2),
            }],
            &scalar,
            CancellationToken::new(),
            done,
        )
        .unwrap();
    wait(&rx).unwrap();
    assert_eq!(
        target.to_buffer(),
        Buffer::F64(vec![3.0, 0.0, 3.0, 0.0, 3.0])
    );

    let values = context.array(vec![2], Buffer::F64(vec![7.0, 8.0])).unwrap();
    let (done, rx) = channel();
    context
        .assign(
            &target,
            vec![Selector::Indices(vec![3, 1])],
            &Expr::input(&values),
            CancellationToken::new(),
            done,
        )
        .unwrap();
    wait(&rx).unwrap();
    assert_eq!(
        target.to_buffer(),
        Buffer::F64(vec![3.0, 8.0, 3.0, 7.0, 3.0])
    );
}

#[test]
fn assignment_memory_rejection_preserves_generator_state() {
    let context = Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 0,
        cpu_memory_budget: 64,
        gpu_memory_budget: 0,
    })
    .unwrap();
    let target = context.array(vec![4], Buffer::F64(vec![0.0; 4])).unwrap();
    let generator = context.generator(42);
    let before = generator.current();
    let (done, rx) = channel();
    context
        .assign(
            &target,
            vec![],
            &Expr::random(context.id(), &generator, vec![4]),
            CancellationToken::new(),
            done,
        )
        .unwrap();
    assert_eq!(wait(&rx).unwrap_err().code, ErrorCode::MemoryBudgetExceeded);
    assert_eq!(generator.current(), before);
    assert_eq!(target.to_buffer(), Buffer::F64(vec![0.0; 4]));
    assert_eq!(context.metrics().cpu_memory.used, 32);
}

#[test]
fn random_assignments_commit_generator_state() {
    let c = ctx(1, 4);
    let g = c.generator(42);
    let before = g.current();
    let target = c.array(vec![3], Buffer::F64(vec![0.0; 3])).unwrap();
    let value = Expr::random(c.id(), &g, vec![3]);
    let (done, rx) = channel();
    c.assign(&target, vec![], &value, CancellationToken::new(), done)
        .unwrap();
    wait(&rx).unwrap();
    // NumPy 2.5.4: default_rng(42).random(3)
    let first = Buffer::F64(vec![
        0.7739560485559633,
        0.4388784397520523,
        0.8585979199113825,
    ]);
    assert_eq!(target.to_buffer(), first);
    assert_ne!(g.current(), before);
    let (done, rx) = channel();
    c.assign(&target, vec![], &value, CancellationToken::new(), done)
        .unwrap();
    wait(&rx).unwrap();
    assert_ne!(
        target.to_buffer(),
        first,
        "the next assignment continues the stream"
    );
    // A failing assignment leaves the generator state unchanged.
    let committed = g.current();
    let (done, rx) = channel();
    c.assign(
        &target,
        vec![Selector::Index(0)],
        &value,
        CancellationToken::new(),
        done,
    )
    .unwrap();
    assert_eq!(wait(&rx).unwrap_err().code, ErrorCode::ShapeBroadcast);
    assert_eq!(g.current(), committed);
}

#[test]
fn load_npy_reserves_encoded_and_decoded_buffers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.npy");
    let bytes = nmath_core::io::encode_npy(&[4], &Buffer::F64(vec![1.0; 4])).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    let load = |budget: u64| {
        let c = Context::new(ContextOptions {
            max_concurrent_jobs: 1,
            max_queue_length: 0,
            cpu_memory_budget: budget,
            gpu_memory_budget: 0,
        })
        .unwrap();
        let (done, rx) = channel();
        c.load_npy(path.clone(), CancellationToken::new(), done)
            .unwrap();
        (c, wait(&rx))
    };
    // A zero budget rejects before the file is read.
    assert_eq!(load(0).1.unwrap_err().code, ErrorCode::MemoryBudgetExceeded);
    // The file alone fits, but not together with the 32-byte decoded buffer.
    assert_eq!(
        load(bytes.len() as u64).1.unwrap_err().code,
        ErrorCode::MemoryBudgetExceeded
    );
    let (c, loaded) = load(bytes.len() as u64 + 32);
    let JobOutput::Array(loaded) = loaded.unwrap() else {
        panic!()
    };
    assert_eq!(loaded.to_buffer(), Buffer::F64(vec![1.0; 4]));
    assert_eq!(
        c.metrics().cpu_memory.used,
        32,
        "only the array stays charged"
    );
}

#[test]
fn load_npy_fits_exact_encoded_and_decoded_budget() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("one.npy");
    let bytes = nmath_core::io::encode_npy(&[1], &Buffer::F64(vec![1.0])).unwrap();
    assert_eq!(bytes.len(), 136);
    std::fs::write(&path, bytes).unwrap();

    for (limit, succeeds) in [(143, false), (144, true)] {
        let context = Context::new(ContextOptions {
            max_concurrent_jobs: 1,
            max_queue_length: 0,
            cpu_memory_budget: limit,
            gpu_memory_budget: 0,
        })
        .unwrap();
        let (done, rx) = channel();
        context
            .load_npy(path.clone(), CancellationToken::new(), done)
            .unwrap();
        if succeeds {
            let JobOutput::Array(array) = wait(&rx).unwrap() else {
                panic!("expected an array")
            };
            assert_eq!(array.shape(), &[1]);
            assert_eq!(array.to_buffer(), Buffer::F64(vec![1.0]));
            assert_eq!(context.metrics().cpu_memory.used, 8);
            assert_eq!(context.metrics().cpu_memory.peak, 144);
            drop(array);
        } else {
            assert_eq!(wait(&rx).unwrap_err().code, ErrorCode::MemoryBudgetExceeded);
            assert_eq!(context.metrics().cpu_memory.peak, 136);
        }
        assert_eq!(context.metrics().cpu_memory.used, 0);
    }
}

#[test]
fn scheduled_load_npy_accepts_alternate_header_quotes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("quoted.npy");
    let header = b"{\"descr\" : \"<f8\", \"fortran_order\" : False, \"shape\" : (1,), }";
    let mut padded = header.to_vec();
    padded.extend(vec![b' '; (64 - (10 + padded.len() + 1) % 64) % 64]);
    padded.push(b'\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(padded.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&padded);
    bytes.extend_from_slice(&1.25f64.to_le_bytes());
    std::fs::write(&path, bytes).unwrap();

    let c = ctx(1, 0);
    let (done, rx) = channel();
    c.load_npy(path, CancellationToken::new(), done).unwrap();
    let JobOutput::Array(loaded) = wait(&rx).unwrap() else {
        panic!("expected an array");
    };
    assert_eq!(loaded.shape(), &[1]);
    assert_eq!(loaded.to_buffer(), Buffer::F64(vec![1.25]));
}
