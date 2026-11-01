// SPDX-License-Identifier: Apache-2.0
use nmath_core::{
    Buffer, CancellationToken, DenseArray, Device, ErrorCode, Expr, NdArray, Scalar, ScanOp,
    kernels::{self, validate_scan},
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
            (Scalar::Float(a), Scalar::Float(b)) => float(a, b, &format!("{label} at {i}")),
            (Scalar::Complex(a), Scalar::Complex(b)) => {
                float(a[0], b[0], &format!("{label} at {i} real"));
                float(a[1], b[1], &format!("{label} at {i} imag"));
            }
            (a, b) => assert_eq!(a, b, "{label} at {i}"),
        }
    }
}

fn op(id: &str) -> ScanOp {
    match id {
        "cumsum" => ScanOp::CumSum,
        "cumprod" => ScanOp::CumProd,
        _ => panic!("unknown fixture operation"),
    }
}

#[test]
fn fixed_numpy_scan_oracle() {
    let fixture: Json = serde_json::from_str(include_str!("fixtures/scans/oracle.json")).unwrap();
    assert_eq!(fixture["numpy"], "2.5.4");
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let input = packed(&case["input"]);
        let expected = packed(&case["expected"]);
        let axis = case["axis"].as_i64();
        let budget = MemoryBudget::new(Device::Cpu, 1 << 20);
        let label = format!(
            "case {i}: {} {:?} dtype={:?} axis={:?}",
            case["op"],
            input.shape,
            input.dtype(),
            case["axis"]
        );
        let out = kernels::scan(
            op(case["op"].as_str().unwrap()),
            &input,
            axis,
            &budget,
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(out.shape, expected.shape, "{label}");
        equal(&out.buffer, &expected.buffer, &label);
        drop(out);
        assert_eq!(budget.usage().used, 0, "{label}");
    }
}

#[test]
fn expression_inference_validation_and_gpu_rejection() {
    let dense = DenseArray::new(
        vec![2, 3],
        Buffer::I16(vec![1, 2, 3, 4, 5, 6]),
        Reservation::none(),
    )
    .unwrap();
    let input = Expr::input(&NdArray::from_dense(0, dense));

    let flat = Expr::scan(ScanOp::CumSum, &input, None).unwrap();
    assert_eq!(flat.dtype(), nmath_core::DType::Int64);
    assert_eq!(flat.shape(), &[6]);
    let along = Expr::scan(ScanOp::CumProd, &input, Some(-1)).unwrap();
    assert_eq!(along.dtype(), nmath_core::DType::Int64);
    assert_eq!(along.shape(), &[2, 3]);
    assert_eq!(
        Expr::scan(ScanOp::CumSum, &input, Some(2))
            .unwrap_err()
            .code,
        ErrorCode::IndexOutOfBounds
    );

    for dtype in ["str[4]", "datetime64[D]", "timedelta64[s]"] {
        assert_eq!(
            validate_scan(nmath_core::DType::from_id(dtype).unwrap())
                .unwrap_err()
                .code,
            ErrorCode::DTypeUnsupported
        );
    }
    assert_eq!(
        nmath_core::gpu::validate(&flat).unwrap_err().code,
        ErrorCode::GpuUnsupportedOperation
    );
}

#[test]
fn empty_budgets_and_cancellation() {
    let token = CancellationToken::new();
    let budget = MemoryBudget::new(Device::Cpu, 1 << 20);
    let empty = DenseArray::new(vec![0], Buffer::F64(vec![]), Reservation::none()).unwrap();
    let flat = kernels::scan(ScanOp::CumSum, &empty, None, &budget, &token).unwrap();
    assert_eq!(flat.shape, [0]);

    let values = DenseArray::new(
        vec![2, 3],
        Buffer::I32(vec![1, 2, 3, 4, 5, 6]),
        Reservation::none(),
    )
    .unwrap();
    for limit in [0, 8, 15] {
        let budget = MemoryBudget::new(Device::Cpu, limit);
        assert_eq!(
            kernels::scan(ScanOp::CumProd, &values, Some(1), &budget, &token)
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded
        );
        assert_eq!(budget.usage().used, 0);
    }

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let budget = MemoryBudget::new(Device::Cpu, 1 << 20);
    assert_eq!(
        kernels::scan(ScanOp::CumSum, &values, None, &budget, &cancelled)
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
}
