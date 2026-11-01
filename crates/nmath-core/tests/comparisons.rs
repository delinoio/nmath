// SPDX-License-Identifier: Apache-2.0
use nmath_core::{
    BinaryOp, Buffer, CancellationToken, DType, DenseArray, Device, ErrorCode, Expr, NdArray,
    kernels,
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

fn op(id: &str) -> BinaryOp {
    match id {
        "eq" => BinaryOp::Eq,
        "ne" => BinaryOp::Ne,
        "lt" => BinaryOp::Lt,
        "le" => BinaryOp::Le,
        "gt" => BinaryOp::Gt,
        "ge" => BinaryOp::Ge,
        _ => panic!("unknown fixture operation"),
    }
}

#[test]
fn fixed_numpy_comparison_oracle() {
    let fixture: Json =
        serde_json::from_str(include_str!("fixtures/comparisons/oracle.json")).unwrap();
    assert_eq!(fixture["numpy"], "2.5.4");
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let a = packed(&case["a"]);
        let b = packed(&case["b"]);
        let expected = packed(&case["expected"]);
        let compute = DType::promote(a.dtype(), b.dtype()).unwrap();
        let budget = MemoryBudget::new(Device::Cpu, 1 << 20);
        let label = format!(
            "case {i}: {} a={:?} b={:?}",
            case["op"],
            a.dtype(),
            b.dtype()
        );
        let out = kernels::binary(
            op(case["op"].as_str().unwrap()),
            &a,
            &b,
            compute,
            &budget,
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(out.shape, expected.shape, "{label}");
        assert_eq!(out.buffer, expected.buffer, "{label}");
        drop(out);
        assert_eq!(budget.usage().used, 0, "{label}");
    }
}

#[test]
fn expression_dtype_errors_and_gpu_admission() {
    let ints = DenseArray::new(vec![3], Buffer::I32(vec![1, 2, 3]), Reservation::none()).unwrap();
    let floats = DenseArray::new(
        vec![3],
        Buffer::F64(vec![1.0, 2.0, 3.0]),
        Reservation::none(),
    )
    .unwrap();
    let a = Expr::input(&NdArray::from_dense(0, ints));
    let b = Expr::input(&NdArray::from_dense(0, floats));
    let compared = Expr::binary(BinaryOp::Lt, &a, &b).unwrap();
    assert_eq!(compared.dtype(), DType::Bool);
    assert_eq!(compared.shape(), &[3]);
    assert_eq!(
        nmath_core::gpu::validate(&compared).unwrap_err().code,
        ErrorCode::GpuUnsupportedOperation
    );

    let complex = Expr::input(&NdArray::from_dense(
        0,
        DenseArray::new(
            vec![2],
            Buffer::C64(vec![[1.0, 0.0], [0.0, 1.0]]),
            Reservation::none(),
        )
        .unwrap(),
    ));
    assert_eq!(
        Expr::binary(BinaryOp::Lt, &complex, &complex)
            .unwrap_err()
            .code,
        ErrorCode::DTypeUnsupported
    );
    assert!(Expr::binary(BinaryOp::Eq, &complex, &complex).is_ok());

    let strings = Expr::input(&NdArray::from_dense(
        0,
        DenseArray::new(
            vec![1],
            Buffer::Unicode {
                width: 1,
                values: vec![b'a' as u32],
            },
            Reservation::none(),
        )
        .unwrap(),
    ));
    assert_eq!(
        Expr::binary(BinaryOp::Eq, &strings, &strings)
            .unwrap_err()
            .code,
        ErrorCode::DTypeUnsupported
    );
}

#[test]
fn comparison_budgets_and_cancellation() {
    let a = DenseArray::new(
        vec![2, 3],
        Buffer::I32(vec![1, 2, 3, 4, 5, 6]),
        Reservation::none(),
    )
    .unwrap();
    let b = DenseArray::new(vec![3], Buffer::I32(vec![2, 2, 2]), Reservation::none()).unwrap();
    let token = CancellationToken::new();
    for limit in [0, 1, 3] {
        let budget = MemoryBudget::new(Device::Cpu, limit);
        assert_eq!(
            kernels::binary(BinaryOp::Eq, &a, &b, DType::Int32, &budget, &token)
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
        kernels::binary(BinaryOp::Ne, &a, &b, DType::Int32, &budget, &cancelled)
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
}
