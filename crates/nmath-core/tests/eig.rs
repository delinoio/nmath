// SPDX-License-Identifier: Apache-2.0
//! NumPy test_linalg invariants for general eigendecomposition.
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

fn single(d: DType) -> bool {
    matches!(d, DType::Float32 | DType::Complex64)
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

fn values(a: &DenseArray) -> Vec<Z> {
    (0..a.size())
        .map(|i| a.buffer.get(i).as_complex())
        .collect()
}

fn run(op: LinalgOp, a: &DenseArray) -> DenseArray {
    let budget = MemoryBudget::new(Device::Cpu, 1 << 24);
    let copy = DenseArray::new(a.shape.clone(), a.buffer.clone(), Reservation::none()).unwrap();
    let expr = Expr::linalg(op, &Expr::input(&NdArray::from_dense(1, copy)), None).unwrap();
    let out = linalg::execute(op, a, None, &budget, &CancellationToken::new()).unwrap();
    assert_eq!(out.shape, expr.shape());
    assert_eq!(out.dtype(), expr.dtype());
    out
}

#[test]
fn fixed_numpy_eig_invariants() {
    let fixture: Json = serde_json::from_str(include_str!("fixtures/eig/oracle.json")).unwrap();
    assert_eq!(fixture["numpy"], "2.5.4");
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let a = packed(&case["input"]);
        let expected = packed(&case["eigenvalues"]);
        let label = format!("case {i}: {:?} {:?}", a.dtype(), a.shape);
        let (batches, _, n, input) = matrices(&a);
        let w = run(LinalgOp::EigValues, &a);
        let v = run(LinalgOp::EigVectors, &a);
        assert_eq!(w.dtype(), expected.dtype(), "{label}");
        let d = w.dtype();
        let tol = 1.5 * 10f64.powi(-(if single(d) { 6 } else { 12 }));
        let (w, v, want) = (values(&w), values(&v), values(&expected));
        for b in 0..batches {
            // Multiset match: each NumPy eigenvalue pairs with the nearest
            // unused one.
            let mut used = vec![false; n];
            for e in &want[b * n..(b + 1) * n] {
                let (j, dist) = (0..n)
                    .filter(|&j| !used[j])
                    .map(|j| {
                        let g = w[b * n + j];
                        (j, abs([g[0] - e[0], g[1] - e[1]]))
                    })
                    .min_by(|x, y| x.1.total_cmp(&y.1))
                    .unwrap();
                assert!(
                    dist < tol,
                    "{label} eigenvalue {e:?}: nearest differs by {dist}"
                );
                used[j] = true;
            }
            let vb = &v[b * n * n..(b + 1) * n * n];
            let av = matmul(&input[b * n * n..(b + 1) * n * n], vb, n, n, n);
            let wv: Vec<Z> = (0..n * n).map(|x| mul(w[b * n + x % n], vb[x])).collect();
            close(&av, &wv, d, &format!("{label} A v = v w"));
        }
    }
}

#[test]
fn eig_errors_and_admission() {
    let dense = |shape: Vec<usize>, buffer: Buffer| {
        DenseArray::new(shape, buffer, Reservation::none()).unwrap()
    };
    let budget = MemoryBudget::new(Device::Cpu, 1 << 20);
    let nan = dense(vec![2, 2], Buffer::F64(vec![f64::NAN, 0.0, 0.0, 1.0]));
    assert_eq!(
        linalg::execute(
            LinalgOp::EigValues,
            &nan,
            None,
            &budget,
            &CancellationToken::new()
        )
        .unwrap_err()
        .code,
        ErrorCode::LinalgNoConvergence
    );
    let wide = Expr::input(&NdArray::from_dense(
        1,
        dense(vec![2, 3], Buffer::F64(vec![0.0; 6])),
    ));
    assert_eq!(
        Expr::linalg(LinalgOp::EigValues, &wide, None)
            .unwrap_err()
            .code,
        ErrorCode::ShapeMismatch
    );
    let real = Expr::input(&NdArray::from_dense(
        1,
        dense(vec![2, 2], Buffer::F32(vec![1.0, 0.0, 0.0, 2.0])),
    ));
    let w = Expr::linalg(LinalgOp::EigValues, &real, None).unwrap();
    assert_eq!(w.dtype(), DType::Complex64);
    assert_eq!(
        nmath_core::gpu::validate(&w).unwrap_err().code,
        ErrorCode::GpuUnsupportedOperation
    );
    let empty = dense(vec![0, 0], Buffer::F64(vec![]));
    assert_eq!(
        linalg::execute(
            LinalgOp::EigValues,
            &empty,
            None,
            &budget,
            &CancellationToken::new()
        )
        .unwrap()
        .shape,
        vec![0]
    );
}
