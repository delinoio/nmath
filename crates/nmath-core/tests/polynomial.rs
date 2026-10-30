// SPDX-License-Identifier: Apache-2.0
use nmath_core::{
    Buffer, CancellationToken, DType, DenseArray, Device, ErrorCode, Expr, NdArray, Scalar,
    Selector,
    expr::JobEnv,
    memory::{MemoryBudget, Reservation},
    polynomial::{self, PolynomialOptions, PolynomialOrder},
};
use serde_json::Value as Json;

fn options(ascending: bool, tensor: bool) -> PolynomialOptions {
    PolynomialOptions {
        order: if ascending {
            PolynomialOrder::Ascending
        } else {
            PolynomialOrder::Descending
        },
        tensor,
    }
}
fn dense(shape: &[usize], buffer: Buffer) -> DenseArray {
    DenseArray::new(shape.to_vec(), buffer, Reservation::none()).unwrap()
}
fn unpack(j: &Json) -> DenseArray {
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
    dense(
        &serde_json::from_value::<Vec<usize>>(j["shape"].clone()).unwrap(),
        buffer,
    )
}
fn equal(actual: &Buffer, expected: &Buffer, label: &str) {
    assert_eq!(actual.dtype(), expected.dtype(), "{label}");
    assert_eq!(actual.len(), expected.len(), "{label}");
    fn float(a: f64, b: f64, label: &str) {
        if b.is_nan() {
            assert!(a.is_nan(), "{label}: {a} != NaN");
        } else {
            assert_eq!(a.to_bits(), b.to_bits(), "{label}: {a} != {b}");
        }
    }
    for i in 0..actual.len() {
        match (actual.get(i), expected.get(i)) {
            (Scalar::Float(a), Scalar::Float(b)) => float(a, b, label),
            (Scalar::Complex(a), Scalar::Complex(b)) => {
                float(a[0], b[0], label);
                float(a[1], b[1], label);
            }
            (a, b) => assert_eq!(a, b, "{label}"),
        }
    }
}
#[test]
fn fixed_numpy_polynomial_oracle() {
    let fixture: Json =
        serde_json::from_str(include_str!("fixtures/polynomial/oracle.json")).unwrap();
    assert_eq!(fixture["numpy"], "2.5.4");
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let c = unpack(&case["c"]);
        let x = unpack(&case["x"]);
        let expected = unpack(&case["expected"]);
        let budget = MemoryBudget::new(Device::Cpu, 1 << 20);
        let out = polynomial::execute(
            options(
                case["ascending"].as_bool().unwrap(),
                case["tensor"].as_bool().unwrap(),
            ),
            &c,
            &x,
            &budget,
            &CancellationToken::new(),
        )
        .unwrap();
        let label = format!(
            "case {i}: c={:?}, x={:?}, ascending={}",
            c.dtype(),
            x.dtype(),
            case["ascending"]
        );
        assert_eq!(out.shape, expected.shape, "{label}");
        equal(&out.buffer, &expected.buffer, &label);
        assert!(budget.usage().used >= out.buffer.byte_len());
        drop(out);
        assert_eq!(budget.usage().used, 0);
    }
}
#[test]
fn validation_budget_cancellation_and_gpu_admission() {
    let opts = options(true, false);
    assert_eq!(
        opts.infer(&[0], DType::Float64, &[2], DType::Float64)
            .unwrap_err()
            .code,
        ErrorCode::IndexOutOfBounds
    );
    assert_eq!(
        opts.infer(&[3, 2], DType::Float64, &[3], DType::Float64)
            .unwrap_err()
            .code,
        ErrorCode::ShapeBroadcast
    );
    assert_eq!(
        opts.infer(&[3], DType::Str(2), &[3], DType::Float64)
            .unwrap_err()
            .code,
        ErrorCode::DTypeUnsupported
    );
    assert!(
        options(false, false)
            .infer(&[], DType::Float64, &[2], DType::Float64)
            .is_err()
    );
    let c = dense(&[3], Buffer::I64(vec![1, 2, 3]));
    let x = dense(&[2], Buffer::F64(vec![1., 2.]));
    for limit in [0, 16, 64, 95] {
        let budget = MemoryBudget::new(Device::Cpu, limit);
        assert_eq!(
            polynomial::execute(opts, &c, &x, &budget, &CancellationToken::new())
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded
        );
        assert_eq!(budget.usage().used, 0);
    }
    let token = CancellationToken::new();
    token.cancel();
    assert_eq!(
        polynomial::execute(opts, &c, &x, &MemoryBudget::new(Device::Cpu, 4096), &token)
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
    let a = NdArray::from_dense(1, c);
    let b = NdArray::from_dense(2, x);
    assert_eq!(
        Expr::polynomial(opts, &Expr::input(&a), &Expr::input(&b))
            .unwrap_err()
            .code,
        ErrorCode::ContextMismatch
    );
    let expression =
        Expr::polynomial(options(false, false), &Expr::input(&a), &Expr::input(&a)).unwrap();
    assert_eq!(
        nmath_core::gpu::validate(&expression).unwrap_err().code,
        ErrorCode::GpuUnsupportedOperation
    );
}
#[test]
fn lazy_polynomial_uses_execution_snapshots_for_both_operands() {
    let c = NdArray::from_dense(1, dense(&[2], Buffer::I32(vec![1, 2])));
    let x = NdArray::from_dense(1, dense(&[2], Buffer::I32(vec![3, 4])));
    let expr = Expr::polynomial(options(false, false), &Expr::input(&c), &Expr::input(&x)).unwrap();
    let write = |a: &NdArray, value| {
        let token = CancellationToken::new();
        let guard = a.write_ticket().wait(&token).unwrap();
        a.assign(
            &guard,
            &[Selector::Index(0)],
            &dense(&[], Buffer::I32(vec![value])),
            &token,
        )
        .unwrap();
    };
    let mut job = JobEnv::new(
        Device::Cpu,
        MemoryBudget::new(Device::Cpu, 4096),
        CancellationToken::new(),
    );
    job.freeze(&expr).unwrap();
    write(&c, 10);
    write(&x, 20);
    let out = match job.eval(&expr).unwrap() {
        nmath_core::Value::Dense(v) => v,
        _ => panic!("polynomial must produce independent storage"),
    };
    assert_eq!(out.buffer, Buffer::I32(vec![5, 6]));
    let mut next = JobEnv::new(
        Device::Cpu,
        MemoryBudget::new(Device::Cpu, 4096),
        CancellationToken::new(),
    );
    next.freeze(&expr).unwrap();
    let out = match next.eval(&expr).unwrap() {
        nmath_core::Value::Dense(v) => v,
        _ => panic!(),
    };
    assert_eq!(out.buffer, Buffer::I32(vec![202, 42]));
}
