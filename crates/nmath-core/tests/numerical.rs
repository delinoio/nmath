// SPDX-License-Identifier: Apache-2.0
use nmath_core::{
    Buffer, CancellationToken, Context, ContextOptions, DType, DenseArray, Device, ErrorCode, Expr,
    Scalar, Selector, Value,
    expr::JobEnv,
    fft::{self, FftOp, FftOptions, Norm},
    linalg::{self, LinalgOp},
    memory::{MemoryBudget, Reservation},
};
use serde_json::Value as Json;
fn budget() -> MemoryBudget {
    MemoryBudget::new(Device::Cpu, u64::MAX)
}
fn dense(shape: Vec<usize>, buffer: Buffer) -> DenseArray {
    DenseArray::new(shape, buffer, Reservation::none()).unwrap()
}
fn unpack(j: &Json) -> DenseArray {
    let shape = serde_json::from_value(j["shape"].clone()).unwrap();
    let dtype = DType::from_id(j["dtype"].as_str().unwrap()).unwrap();
    let values = j["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|z| [z[0].as_f64().unwrap(), z[1].as_f64().unwrap()])
        .collect();
    dense(shape, Buffer::C128(values).cast(dtype).unwrap())
}
fn options(op: &str, n: Option<usize>, axis: i64, norm: &str) -> FftOptions {
    FftOptions {
        op: match op {
            "fft" => FftOp::Fft,
            "ifft" => FftOp::Ifft,
            "rfft" => FftOp::Rfft,
            "irfft" => FftOp::Irfft,
            _ => panic!(),
        },
        n,
        axis,
        norm: match norm {
            "backward" => Norm::Backward,
            "forward" => Norm::Forward,
            "ortho" => Norm::Ortho,
            _ => panic!(),
        },
    }
}
fn close(a: &Buffer, b: &Buffer, atol: f64) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        let x = a.get(i).as_complex();
        let y = b.get(i).as_complex();
        for j in 0..2 {
            assert!(
                (x[j] - y[j]).abs() < atol,
                "element {i}/{j}: {} != {} (bound {atol})",
                x[j],
                y[j]
            );
        }
    }
}
#[test]
fn fixed_numpy_oracle_criteria() {
    let fixtures = [
        include_str!("fixtures/numerical/linalg.json"),
        include_str!("fixtures/numerical/fft_identity.json"),
        include_str!("fixtures/numerical/fft_adaptation.json"),
    ];
    let fixtures: Vec<Json> = fixtures
        .iter()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    let budget = budget();
    let token = CancellationToken::new();
    for case in fixtures.iter().flat_map(|f| f["cases"].as_array().unwrap()) {
        let a = unpack(&case["a"]);
        let expected = unpack(&case["expected"]);
        let op = case["op"].as_str().unwrap();
        if case["family"] == "linalg" {
            let op = match op {
                "solve" => LinalgOp::Solve,
                "inv" => LinalgOp::Inv,
                _ => LinalgOp::Det,
            };
            let b = (!case["b"].is_null()).then(|| unpack(&case["b"]));
            let out = linalg::execute(op, &a, b.as_ref(), &budget, &token).unwrap();
            assert_eq!(
                (out.dtype(), &out.shape),
                (expected.dtype(), &expected.shape)
            );
            // NumPy assert_almost_equal uses 1.5 * 10**(-decimal). Use the
            // stricter 10**(-decimal), per component, for these exact source
            // inputs.
            let tolerance = 10f64.powi(-(case["decimal"].as_i64().unwrap() as i32));
            if op == LinalgOp::Det {
                close(&out.buffer, &unpack(&case["reference"]).buffer, tolerance);
            } else {
                let product =
                    nmath_core::kernels::matmul(&a, &out, out.dtype(), &budget, &token).unwrap();
                let target = if let Some(b) = b {
                    b
                } else {
                    dense(vec![2, 2], Buffer::F64(vec![1., 0., 0., 1.]))
                };
                close(&product.buffer, &target.buffer, tolerance);
            }
        } else {
            let n = case["n"].as_u64().unwrap() as usize;
            let opts = options(
                op,
                Some(n),
                case["axis"].as_i64().unwrap_or(-1),
                case["norm"].as_str().unwrap_or("backward"),
            );
            let out = fft::execute(opts, &a, &budget, &token).unwrap();
            assert_eq!(
                (out.dtype(), &out.shape),
                (expected.dtype(), &expected.shape),
                "{case}"
            );
            if case["family"] == "fft_identity" {
                let inverse = options(
                    if op == "rfft" { "irfft" } else { "ifft" },
                    Some(n),
                    -1,
                    "backward",
                );
                let roundtrip = fft::execute(inverse, &out, &budget, &token).unwrap();
                let mut target = Buffer::filled(a.dtype(), n, Scalar::Float(0.)).unwrap();
                for i in 0..n.min(a.size()) {
                    target.set_from(i, &a.buffer, i);
                }
                for i in 0..n {
                    let x = roundtrip.buffer.get(i).as_complex();
                    let y = target.get(i).as_complex();
                    assert!(
                        (x[0] - y[0]).hypot(x[1] - y[1]) <= case["atol"].as_f64().unwrap(),
                        "FFT complex absolute error exceeds the source bound: {case}"
                    );
                }
            }
        }
    }
    assert_eq!(budget.usage().used, 0);
}
#[test]
fn square_batches_empty_and_singular() {
    let budget = budget();
    let t = CancellationToken::new();
    let a = dense(
        vec![2, 1, 2, 2],
        Buffer::F64(vec![2., 0., 0., 4., 1., 0., 0., 2.]),
    );
    let b = dense(vec![3, 2, 1], Buffer::F64(vec![2., 4., 4., 8., 6., 12.]));
    let out = linalg::execute(LinalgOp::Solve, &a, Some(&b), &budget, &t).unwrap();
    assert_eq!(out.shape, vec![2, 3, 2, 1]);
    assert_eq!(
        out.buffer,
        Buffer::F64(vec![1., 1., 2., 2., 3., 3., 2., 2., 4., 4., 6., 6.])
    );
    let empty = dense(vec![2, 0, 0], Buffer::C64(vec![]));
    assert_eq!(
        linalg::execute(LinalgOp::Det, &empty, None, &budget, &t)
            .unwrap()
            .buffer,
        Buffer::C64(vec![[1., 0.]; 2])
    );
    assert_eq!(
        linalg::execute(LinalgOp::Inv, &empty, None, &budget, &t)
            .unwrap()
            .shape,
        vec![2, 0, 0]
    );
    let singular = dense(vec![2, 2], Buffer::F64(vec![1., 2., 2., 4.]));
    assert_eq!(
        linalg::execute(LinalgOp::Inv, &singular, None, &budget, &t)
            .unwrap_err()
            .code,
        ErrorCode::LinalgSingular
    );
    assert_eq!(
        linalg::execute(LinalgOp::Det, &singular, None, &budget, &t)
            .unwrap()
            .buffer,
        Buffer::F64(vec![0.])
    );
    assert!(linalg::infer(LinalgOp::Inv, &[2, 3], DType::Float64, None).is_err());
    assert!(linalg::infer(LinalgOp::Inv, &[2, 2], DType::Float16, None).is_err());
    assert!(
        linalg::infer(
            LinalgOp::Solve,
            &[2, 2],
            DType::Float64,
            Some((&[3, 2], DType::Float64))
        )
        .is_err()
    );
}
#[test]
fn fft_conjugation_normalization_and_empty_axes() {
    let b = budget();
    let t = CancellationToken::new();
    let z = dense(
        vec![3],
        Buffer::C128(vec![[1., 100.], [2., 3.], [4., 200.]]),
    );
    let out = fft::execute(options("irfft", None, -1, "backward"), &z, &b, &t).unwrap();
    assert_eq!(out.buffer, Buffer::F64(vec![2.25, -2.25, 0.25, 0.75]));
    let one = dense(vec![1], Buffer::F64(vec![8.]));
    for norm in ["backward", "forward", "ortho"] {
        assert_eq!(
            fft::execute(options("fft", None, -1, norm), &one, &b, &t)
                .unwrap()
                .buffer,
            Buffer::C128(vec![[8., 0.]])
        );
    }
    let empty = dense(vec![0, 3], Buffer::F64(vec![]));
    assert_eq!(
        fft::execute(options("fft", None, -1, "backward"), &empty, &b, &t)
            .unwrap()
            .shape,
        vec![0, 3]
    );
    assert!(fft::execute(options("fft", None, 0, "backward"), &empty, &b, &t).is_err());
    assert!(fft::execute(options("rfft", None, -1, "backward"), &z, &b, &t).is_err());
}
#[test]
fn workspace_failure_and_cancellation_release_charges() {
    let a = dense(vec![2, 2], Buffer::F64(vec![1., 0., 0., 1.]));
    for limit in [0, 31, 32, 159] {
        let b = MemoryBudget::new(Device::Cpu, limit);
        assert_eq!(
            linalg::execute(LinalgOp::Inv, &a, None, &b, &CancellationToken::new())
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded
        );
        assert_eq!(b.usage().used, 0);
    }
    let a = dense(vec![3], Buffer::F64(vec![1., 2., 3.]));
    let b = MemoryBudget::new(Device::Cpu, 48);
    assert_eq!(
        fft::execute(
            options("fft", None, -1, "backward"),
            &a,
            &b,
            &CancellationToken::new()
        )
        .unwrap_err()
        .code,
        ErrorCode::MemoryBudgetExceeded
    );
    assert_eq!(b.usage().used, 0);
    let t = CancellationToken::new();
    t.cancel();
    assert_eq!(
        fft::execute(options("fft", None, -1, "backward"), &a, &budget(), &t)
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
}
#[test]
fn lazy_graph_freezes_inputs_and_rejects_gpu_and_context_mix() {
    let c = Context::new(ContextOptions {
        max_concurrent_jobs: 1,
        max_queue_length: 2,
        cpu_memory_budget: 1 << 20,
        gpu_memory_budget: 0,
    })
    .unwrap();
    let a = c
        .array(vec![2, 2], Buffer::F64(vec![2., 0., 0., 4.]))
        .unwrap();
    let e = Expr::linalg(LinalgOp::Inv, &Expr::input(&a), None).unwrap();
    let mut env = JobEnv::new(Device::Cpu, budget(), CancellationToken::new());
    env.freeze(&e).unwrap();
    let t = CancellationToken::new();
    let turn = a.write_ticket().wait(&t).unwrap();
    a.assign(
        &turn,
        &[Selector::Index(0), Selector::Index(0)],
        &dense(vec![], Buffer::F64(vec![8.])),
        &t,
    )
    .unwrap();
    drop(turn);
    let Value::Dense(out) = env.eval(&e).unwrap() else {
        panic!()
    };
    assert_eq!(out.buffer, Buffer::F64(vec![0.5, 0., 0., 0.25]));
    assert_eq!(
        e.check_gpu_support().unwrap_err().code,
        ErrorCode::GpuUnsupportedOperation
    );
    let foreign =
        nmath_core::NdArray::from_dense(c.id() + 1, dense(vec![2], Buffer::F64(vec![1., 1.])));
    assert_eq!(
        Expr::linalg(
            LinalgOp::Solve,
            &Expr::input(&a),
            Some(&Expr::input(&foreign))
        )
        .unwrap_err()
        .code,
        ErrorCode::ContextMismatch
    );
    c.close();
}

#[test]
fn queued_numerical_jobs_snapshot_at_start_and_cancel_without_execution() {
    use std::{sync::mpsc, time::Duration};
    for is_fft in [false, true] {
        let c = Context::new(ContextOptions {
            max_concurrent_jobs: 1,
            max_queue_length: 2,
            cpu_memory_budget: 1 << 20,
            gpu_memory_budget: 0,
        })
        .unwrap();
        let blocker = c.array(vec![1], Buffer::F64(vec![0.])).unwrap();
        let hold = blocker
            .write_ticket()
            .wait(&CancellationToken::new())
            .unwrap();
        let (done, blocked) = mpsc::channel();
        c.assign(
            &blocker,
            vec![],
            &Expr::scalar(Scalar::Float(1.)),
            CancellationToken::new(),
            Box::new(move |r| done.send(r).unwrap()),
        )
        .unwrap();
        let a = c.array(vec![1, 1], Buffer::F64(vec![2.])).unwrap();
        let expression = if is_fft {
            Expr::fft(options("fft", None, -1, "backward"), &Expr::input(&a)).unwrap()
        } else {
            Expr::linalg(LinalgOp::Inv, &Expr::input(&a), None).unwrap()
        };
        let (done, result) = mpsc::channel();
        c.compute(
            &expression,
            Device::Cpu,
            CancellationToken::new(),
            Box::new(move |r| done.send(r).unwrap()),
        )
        .unwrap();
        let cancel = CancellationToken::new();
        let (done, cancelled) = mpsc::channel();
        c.compute(
            &expression,
            Device::Cpu,
            cancel.clone(),
            Box::new(move |r| done.send(r).unwrap()),
        )
        .unwrap();
        let (done, _) = mpsc::channel();
        assert_eq!(
            c.compute(
                &expression,
                Device::Cpu,
                CancellationToken::new(),
                Box::new(move |r| {
                    let _ = done.send(r);
                })
            )
            .unwrap_err()
            .code,
            ErrorCode::SchedQueueFull
        );
        cancel.cancel();
        assert_eq!(
            cancelled
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
        let t = CancellationToken::new();
        let turn = a.write_ticket().wait(&t).unwrap();
        a.assign(&turn, &[], &dense(vec![], Buffer::F64(vec![4.])), &t)
            .unwrap();
        drop(turn);
        drop(hold);
        blocked
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap();
        let nmath_core::JobOutput::Value(Value::Dense(out)) = result
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(
            out.buffer.get(0).as_complex(),
            if is_fft { [4., 0.] } else { [0.25, 0.] }
        );
        c.close();
    }
}

#[test]
fn cancellation_during_fft_releases_workspace() {
    use std::{
        thread,
        time::{Duration, Instant},
    };
    let b = budget();
    let t = CancellationToken::new();
    let worker_budget = b.clone();
    let worker_token = t.clone();
    let input = dense(vec![1 << 20], Buffer::F64(vec![1.; 1 << 20]));
    let worker = thread::spawn(move || {
        fft::execute(
            options("fft", None, -1, "backward"),
            &input,
            &worker_budget,
            &worker_token,
        )
    });
    let start = Instant::now();
    while b.usage().used == 0 {
        assert!(start.elapsed() < Duration::from_secs(10));
        thread::yield_now();
    }
    t.cancel();
    assert_eq!(
        worker.join().unwrap().unwrap_err().code,
        ErrorCode::Cancelled
    );
    assert_eq!(b.usage().used, 0);
}
