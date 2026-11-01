// SPDX-License-Identifier: Apache-2.0
//! NumPy test_linalg invariants for factorizations.
use nmath_core::{
    Buffer, CancellationToken, DType, DenseArray, Device, ErrorCode, Expr, NdArray,
    linalg::{self, LinalgOp},
    memory::{MemoryBudget, Reservation},
};
use serde_json::Value as Json;

fn packed(j: &Json) -> DenseArray {
    let text = j["hex"].as_str().unwrap();
    let bytes: Vec<u8> = (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect();
    macro_rules! real {
        ($variant:ident, $t:ty, $n:expr) => {
            Buffer::$variant(
                bytes
                    .chunks_exact($n)
                    .map(|c| <$t>::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            )
        };
    }
    macro_rules! complex {
        ($variant:ident, $t:ty, $n:expr) => {
            Buffer::$variant(
                bytes
                    .chunks_exact(2 * $n)
                    .map(|c| {
                        [
                            <$t>::from_le_bytes(c[..$n].try_into().unwrap()),
                            <$t>::from_le_bytes(c[$n..].try_into().unwrap()),
                        ]
                    })
                    .collect(),
            )
        };
    }
    let buffer = match j["dtype"].as_str().unwrap() {
        "bool" => Buffer::Bool(bytes.iter().map(|&v| v != 0).collect()),
        "int8" => real!(I8, i8, 1),
        "uint8" => real!(U8, u8, 1),
        "int16" => real!(I16, i16, 2),
        "uint16" => real!(U16, u16, 2),
        "int32" => real!(I32, i32, 4),
        "uint32" => real!(U32, u32, 4),
        "int64" => real!(I64, i64, 8),
        "uint64" => real!(U64, u64, 8),
        "float16" => Buffer::F16(
            bytes
                .chunks_exact(2)
                .map(|c| {
                    nmath_core::buffer::Float16::from_bits(u16::from_le_bytes(
                        c.try_into().unwrap(),
                    ))
                })
                .collect(),
        ),
        "float32" => real!(F32, f32, 4),
        "float64" => real!(F64, f64, 8),
        "complex64" => complex!(C64, f32, 4),
        "complex128" => complex!(C128, f64, 8),
        _ => panic!("unknown fixture dtype"),
    };
    let shape = serde_json::from_value(j["shape"].clone()).unwrap();
    DenseArray::new(shape, buffer, Reservation::none()).unwrap()
}

type Z = [f64; 2];

fn mul(a: Z, b: Z) -> Z {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}
fn conj(a: Z) -> Z {
    [a[0], -a[1]]
}
fn abs(a: Z) -> f64 {
    a[0].hypot(a[1])
}

/// Row-major matrices of one batch: `(batches, rows, cols, values)`.
fn matrices(a: &DenseArray) -> (usize, usize, usize, Vec<Z>) {
    let r = a.shape.len();
    let (rows, cols) = (a.shape[r - 2], a.shape[r - 1]);
    let values: Vec<Z> = (0..a.size())
        .map(|i| a.buffer.get(i).as_complex())
        .collect();
    let batches = if rows * cols == 0 {
        0
    } else {
        values.len() / (rows * cols)
    };
    (batches, rows, cols, values)
}

fn matmul(a: &[Z], b: &[Z], m: usize, k: usize, n: usize) -> Vec<Z> {
    let mut out = vec![[0.0, 0.0]; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = [0.0, 0.0];
            for t in 0..k {
                let p = mul(a[i * k + t], b[t * n + j]);
                s = [s[0] + p[0], s[1] + p[1]];
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn hermitian_t(a: &[Z], m: usize, n: usize) -> Vec<Z> {
    (0..n * m).map(|i| conj(a[(i % m) * n + i / m])).collect()
}

fn single(d: DType) -> bool {
    matches!(d, DType::Float32 | DType::Complex64)
}
/// `numpy.testing.assert_almost_equal` with test_linalg's decimal choice.
fn almost(actual: &[Z], desired: &[Z], d: DType, label: &str) {
    let tol = 1.5 * 10f64.powi(-(if single(d) { 6 } else { 12 }));
    assert_eq!(actual.len(), desired.len(), "{label}");
    for (i, (a, b)) in actual.iter().zip(desired).enumerate() {
        let diff = abs([a[0] - b[0], a[1] - b[1]]);
        let both_inf = a[0].is_infinite() && a[0] == b[0];
        assert!(
            both_inf || diff < tol,
            "{label} element {i}: {a:?} vs {b:?}"
        );
    }
}
/// `assert_allclose` with test_linalg's `get_rtol` and atol 0.
fn close(actual: &[Z], desired: &[Z], d: DType, label: &str) {
    let rtol = if single(d) { 1e-5 } else { 1e-11 };
    for (i, (a, b)) in actual.iter().zip(desired).enumerate() {
        let diff = abs([a[0] - b[0], a[1] - b[1]]);
        assert!(
            diff <= rtol * abs(*b),
            "{label} element {i}: {a:?} vs {b:?}"
        );
    }
}
fn identity(n: usize) -> Vec<Z> {
    (0..n * n)
        .map(|i| {
            if i / n == i % n {
                [1.0, 0.0]
            } else {
                [0.0, 0.0]
            }
        })
        .collect()
}

fn run(op: LinalgOp, a: &DenseArray) -> DenseArray {
    let budget = MemoryBudget::new(Device::Cpu, 1 << 24);
    let expr = Expr::linalg(
        op,
        &Expr::input(&NdArray::from_dense(1, a.clone_dense())),
        None,
    )
    .unwrap();
    let out = linalg::execute(op, a, None, &budget, &CancellationToken::new()).unwrap();
    assert_eq!(out.shape, expr.shape(), "{op:?} shape");
    assert_eq!(out.dtype(), expr.dtype(), "{op:?} dtype");
    out
}

trait CloneDense {
    fn clone_dense(&self) -> DenseArray;
}
impl CloneDense for DenseArray {
    fn clone_dense(&self) -> DenseArray {
        DenseArray::new(self.shape.clone(), self.buffer.clone(), Reservation::none()).unwrap()
    }
}

fn shape(case: &Json, key: &str) -> Vec<usize> {
    serde_json::from_value(case["shapes"][key].clone()).unwrap()
}

fn values(a: &DenseArray) -> Vec<Z> {
    (0..a.size())
        .map(|i| a.buffer.get(i).as_complex())
        .collect()
}

#[test]
fn fixed_numpy_factorization_invariants() {
    let fixture: Json =
        serde_json::from_str(include_str!("fixtures/decompositions/oracle.json")).unwrap();
    assert_eq!(fixture["numpy"], "2.5.4");
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let a = packed(&case["input"]);
        let (batches, m, n, input) = matrices(&a);
        let label = format!("case {i}: {} {:?} {:?}", case["kind"], a.dtype(), a.shape);
        match case["kind"].as_str().unwrap() {
            "general" => {
                let k = m.min(n);
                let s = run(LinalgOp::SvdS, &a);
                let expected = packed(&case["svdvals"]);
                assert_eq!(s.dtype(), expected.dtype(), "{label}");
                let d = s.dtype();
                almost(
                    &values(&s),
                    &values(&expected),
                    d,
                    &format!("{label} svdvals"),
                );
                for full in [false, true] {
                    let u = run(LinalgOp::SvdU { full }, &a);
                    let vh = run(LinalgOp::SvdVh { full }, &a);
                    let key = if full { "" } else { "Reduced" };
                    assert_eq!(u.shape, shape(case, &format!("svdU{key}")), "{label}");
                    assert_eq!(vh.shape, shape(case, &format!("svdVh{key}")), "{label}");
                    let (uc, vr) = (u.shape[u.shape.len() - 1], vh.shape[vh.shape.len() - 2]);
                    let (u, vh, s) = (values(&u), values(&vh), values(&s));
                    for b in 0..batches {
                        let ub = &u[b * m * uc..(b + 1) * m * uc];
                        let vb = &vh[b * vr * n..(b + 1) * vr * n];
                        let sb = &s[b * k..(b + 1) * k];
                        // a == (u[:, :k] * s) @ vh[:k]
                        let us: Vec<Z> = (0..m * k)
                            .map(|x| mul(ub[(x / k) * uc + x % k], sb[x % k]))
                            .collect();
                        let rebuilt = matmul(&us, &vb[..k * n], m, k, n);
                        close(
                            &rebuilt,
                            &input[b * m * n..(b + 1) * m * n],
                            a_dtype(&a),
                            &format!("{label} svd full={full}"),
                        );
                        almost(
                            &matmul(&hermitian_t(ub, m, uc), ub, uc, m, uc),
                            &identity(uc),
                            a_dtype(&a),
                            &format!("{label} U^H U"),
                        );
                        almost(
                            &matmul(vb, &hermitian_t(vb, vr, n), vr, n, vr),
                            &identity(vr),
                            a_dtype(&a),
                            &format!("{label} Vh Vh^H"),
                        );
                    }
                }
                for complete in [false, true] {
                    let q = run(LinalgOp::QrQ { complete }, &a);
                    let r = run(LinalgOp::QrR { complete }, &a);
                    let key = if complete { "Complete" } else { "" };
                    assert_eq!(q.shape, shape(case, &format!("qrQ{key}")), "{label}");
                    assert_eq!(r.shape, shape(case, &format!("qrR{key}")), "{label}");
                    let kk = q.shape[q.shape.len() - 1];
                    let (q, r) = (values(&q), values(&r));
                    for b in 0..batches {
                        let qb = &q[b * m * kk..(b + 1) * m * kk];
                        let rb = &r[b * kk * n..(b + 1) * kk * n];
                        almost(
                            &matmul(qb, rb, m, kk, n),
                            &input[b * m * n..(b + 1) * m * n],
                            a_dtype(&a),
                            &format!("{label} QR"),
                        );
                        almost(
                            &matmul(&hermitian_t(qb, m, kk), qb, kk, m, kk),
                            &identity(kk),
                            a_dtype(&a),
                            &format!("{label} Q^H Q"),
                        );
                        for row in 0..kk {
                            for col in 0..row.min(n) {
                                assert_eq!(rb[row * n + col], [0.0, 0.0], "{label} R triangular");
                            }
                        }
                    }
                }
                if case.get("slogdetSign").is_some() {
                    let sign = run(LinalgOp::SlogdetSign, &a);
                    let logabs = run(LinalgOp::SlogdetLogabs, &a);
                    let (es, el) = (packed(&case["slogdetSign"]), packed(&case["slogdetLogabs"]));
                    assert_eq!(sign.dtype(), es.dtype(), "{label}");
                    assert_eq!(logabs.dtype(), el.dtype(), "{label}");
                    almost(
                        &values(&sign),
                        &values(&es),
                        sign.dtype(),
                        &format!("{label} sign"),
                    );
                    almost(
                        &values(&logabs),
                        &values(&el),
                        logabs.dtype(),
                        &format!("{label} logabs"),
                    );
                }
            }
            "hermitian" => {
                for (upper, key) in [(false, "eigvalsh"), (true, "eigvalshUpper")] {
                    let w = run(LinalgOp::EighValues { upper }, &a);
                    let expected = packed(&case[key]);
                    assert_eq!(w.dtype(), expected.dtype(), "{label}");
                    almost(
                        &values(&w),
                        &values(&expected),
                        w.dtype(),
                        &format!("{label} {key}"),
                    );
                    let v = values(&run(LinalgOp::EighVectors { upper }, &a));
                    let w = values(&w);
                    for b in 0..batches {
                        let vb = &v[b * n * n..(b + 1) * n * n];
                        let av = matmul(&input[b * n * n..(b + 1) * n * n], vb, n, n, n);
                        let wv: Vec<Z> = (0..n * n).map(|x| mul(w[b * n + x % n], vb[x])).collect();
                        close(
                            &av,
                            &wv,
                            a_dtype(&a),
                            &format!("{label} A v = w v upper={upper}"),
                        );
                    }
                }
            }
            _ => {
                for (upper, key) in [(false, "cholesky"), (true, "choleskyUpper")] {
                    let l = run(LinalgOp::Cholesky { upper }, &a);
                    let expected = packed(&case[key]);
                    assert_eq!(l.dtype(), expected.dtype(), "{label}");
                    almost(
                        &values(&l),
                        &values(&expected),
                        l.dtype(),
                        &format!("{label} {key}"),
                    );
                    let lv = values(&l);
                    let eps = if single(a_dtype(&a)) {
                        f32::EPSILON as f64
                    } else {
                        f64::EPSILON
                    };
                    let atol = 500.0 * n as f64 * eps;
                    for b in 0..batches {
                        let lb = &lv[b * n * n..(b + 1) * n * n];
                        let rebuilt = if upper {
                            matmul(&hermitian_t(lb, n, n), lb, n, n, n)
                        } else {
                            matmul(lb, &hermitian_t(lb, n, n), n, n, n)
                        };
                        for (x, y) in rebuilt.iter().zip(&input[b * n * n..(b + 1) * n * n]) {
                            assert!(abs([x[0] - y[0], x[1] - y[1]]) <= atol, "{label} L L^H");
                        }
                    }
                }
            }
        }
    }
}

fn a_dtype(a: &DenseArray) -> DType {
    match a.dtype() {
        DType::Float32 | DType::Complex64 => a.dtype(),
        DType::Complex128 => DType::Complex128,
        _ => DType::Float64,
    }
}

#[test]
fn factorization_errors_and_empty_inputs() {
    let dense = |shape: Vec<usize>, buffer: Buffer| {
        DenseArray::new(shape, buffer, Reservation::none()).unwrap()
    };
    let budget = MemoryBudget::new(Device::Cpu, 1 << 20);
    let token = CancellationToken::new();
    let indefinite = dense(vec![2, 2], Buffer::F64(vec![1.0, 2.0, 2.0, 1.0]));
    assert_eq!(
        linalg::execute(
            LinalgOp::Cholesky { upper: false },
            &indefinite,
            None,
            &budget,
            &token
        )
        .unwrap_err()
        .code,
        ErrorCode::LinalgNotPositiveDefinite
    );
    let nan = dense(vec![2, 2], Buffer::F64(vec![f64::NAN, 0.0, 0.0, 1.0]));
    assert_eq!(
        linalg::execute(LinalgOp::SvdS, &nan, None, &budget, &token)
            .unwrap_err()
            .code,
        ErrorCode::LinalgNoConvergence
    );
    let wide = Expr::input(&NdArray::from_dense(
        1,
        dense(vec![2, 3], Buffer::F64(vec![0.0; 6])),
    ));
    assert_eq!(
        Expr::linalg(LinalgOp::EighValues { upper: false }, &wide, None)
            .unwrap_err()
            .code,
        ErrorCode::ShapeMismatch
    );
    let half = Expr::input(&NdArray::from_dense(
        1,
        dense(
            vec![1, 1],
            Buffer::F16(vec![nmath_core::buffer::Float16::ONE]),
        ),
    ));
    assert_eq!(
        Expr::linalg(LinalgOp::SvdS, &half, None).unwrap_err().code,
        ErrorCode::DTypeUnsupported
    );
    let empty = dense(vec![0, 3], Buffer::F64(vec![]));
    let s = linalg::execute(LinalgOp::SvdS, &empty, None, &budget, &token).unwrap();
    assert_eq!(s.shape, vec![0]);
    let u = linalg::execute(
        LinalgOp::SvdVh { full: true },
        &empty,
        None,
        &budget,
        &token,
    )
    .unwrap();
    assert_eq!(u.shape, vec![3, 3]);
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert_eq!(
        linalg::execute(
            LinalgOp::QrR { complete: false },
            &indefinite,
            None,
            &budget,
            &cancelled
        )
        .unwrap_err()
        .code,
        ErrorCode::Cancelled
    );
    let none = MemoryBudget::new(Device::Cpu, 0);
    assert_eq!(
        linalg::execute(LinalgOp::SvdS, &indefinite, None, &none, &token)
            .unwrap_err()
            .code,
        ErrorCode::MemoryBudgetExceeded
    );
    assert_eq!(none.usage().used, 0);
    let gpu = Expr::linalg(
        LinalgOp::SvdS,
        &Expr::input(&NdArray::from_dense(1, indefinite.clone_dense())),
        None,
    )
    .unwrap();
    assert_eq!(
        nmath_core::gpu::validate(&gpu).unwrap_err().code,
        ErrorCode::GpuUnsupportedOperation
    );
}
