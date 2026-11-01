// SPDX-License-Identifier: Apache-2.0
use nmath_core::{
    Buffer, CancellationToken, DenseArray, Device, ErrorCode, Expr, NdArray, ReduceOp, Scalar,
    kernels::{self, validate_special_reduce},
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

fn op(id: &str) -> ReduceOp {
    match id {
        "prod" => ReduceOp::Prod,
        "argmin" => ReduceOp::ArgMin,
        "argmax" => ReduceOp::ArgMax,
        _ => panic!("unknown fixture operation"),
    }
}

#[test]
fn fixed_numpy_reduction_oracle() {
    let fixture: Json =
        serde_json::from_str(include_str!("fixtures/reductions/oracle.json")).unwrap();
    assert_eq!(fixture["numpy"], "2.5.4");
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let input = packed(&case["input"]);
        let expected = packed(&case["expected"]);
        let axis = case["axis"].as_i64().map(|axis| vec![axis]);
        let budget = MemoryBudget::new(Device::Cpu, 1 << 20);
        let label = format!(
            "case {i}: {} {:?} dtype={:?} axis={:?}",
            case["op"],
            input.shape,
            input.dtype(),
            case["axis"]
        );
        let out = kernels::reduce(
            op(case["op"].as_str().unwrap()),
            &input,
            axis.as_deref(),
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
fn expression_inference_and_validation() {
    let dense = DenseArray::new(
        vec![2, 3],
        Buffer::I16(vec![1, 2, 3, 4, 5, 6]),
        Reservation::none(),
    )
    .unwrap();
    let input = Expr::input(&NdArray::from_dense(0, dense));

    let product = Expr::reduce(ReduceOp::Prod, &input, Some(vec![0])).unwrap();
    assert_eq!(product.dtype(), nmath_core::DType::Int64);
    assert_eq!(product.shape(), &[3]);

    let index = Expr::reduce(ReduceOp::ArgMin, &input, None).unwrap();
    assert_eq!(index.dtype(), nmath_core::DType::Int64);
    assert!(index.shape().is_empty());

    let str_dtype = nmath_core::DType::from_id("str[4]").unwrap();
    let datetime = nmath_core::DType::from_id("datetime64[D]").unwrap();
    for (operation, dtype) in [
        (ReduceOp::ArgMin, str_dtype),
        (ReduceOp::ArgMax, str_dtype),
        (ReduceOp::Prod, str_dtype),
        (ReduceOp::Prod, datetime),
        (ReduceOp::ArgMin, datetime),
    ] {
        assert_eq!(
            validate_special_reduce(operation, dtype, None)
                .unwrap_err()
                .code,
            ErrorCode::DTypeUnsupported
        );
    }
    assert_eq!(
        validate_special_reduce(ReduceOp::ArgMax, nmath_core::DType::Float64, Some(&[0, 1]))
            .unwrap_err()
            .code,
        ErrorCode::IndexInvalid
    );
    assert!(
        validate_special_reduce(
            ReduceOp::Prod,
            nmath_core::DType::from_id("timedelta64[s]").unwrap(),
            None
        )
        .is_err()
    );

    // The GPU subset admits only sum, mean, min and max before admission.
    for operation in [ReduceOp::Prod, ReduceOp::ArgMin, ReduceOp::ArgMax] {
        let expr = Expr::reduce(operation, &input, None).unwrap();
        assert_eq!(
            nmath_core::gpu::validate(&expr).unwrap_err().code,
            ErrorCode::GpuUnsupportedOperation
        );
    }
}

#[test]
fn empty_axes_budgets_and_cancellation() {
    let empty = DenseArray::new(vec![0], Buffer::F64(vec![]), Reservation::none()).unwrap();
    let token = CancellationToken::new();
    let budget = MemoryBudget::new(Device::Cpu, 1 << 20);
    let product = kernels::reduce(ReduceOp::Prod, &empty, None, &budget, &token).unwrap();
    assert!(product.shape.is_empty());
    assert_eq!(product.buffer.get(0), Scalar::Float(1.0));
    assert_eq!(
        kernels::reduce(ReduceOp::ArgMin, &empty, None, &budget, &token)
            .unwrap_err()
            .code,
        ErrorCode::ShapeMismatch
    );

    let values = DenseArray::new(
        vec![2, 3],
        Buffer::I32(vec![1, 2, 3, 4, 5, 6]),
        Reservation::none(),
    )
    .unwrap();
    for limit in [0, 8, 15] {
        let budget = MemoryBudget::new(Device::Cpu, limit);
        for operation in [ReduceOp::Prod, ReduceOp::ArgMax] {
            assert_eq!(
                kernels::reduce(operation, &values, None, &budget, &token)
                    .unwrap_err()
                    .code,
                ErrorCode::MemoryBudgetExceeded
            );
            assert_eq!(budget.usage().used, 0);
        }
    }

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let budget = MemoryBudget::new(Device::Cpu, 1 << 20);
    assert_eq!(
        kernels::reduce(ReduceOp::Prod, &values, None, &budget, &cancelled)
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
}
