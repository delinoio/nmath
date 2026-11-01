// SPDX-License-Identifier: Apache-2.0
use nmath_core::{
    BinaryOp, Buffer, CancellationToken, DType, DenseArray, Device, ErrorCode, Expr, NdArray,
    Scalar, kernels,
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

fn input(a: DenseArray) -> Expr {
    Expr::input(&NdArray::from_dense(1, a))
}

#[test]
fn fixed_numpy_bitwise_oracle() {
    let fixture: Json = serde_json::from_str(include_str!("fixtures/bitwise/oracle.json")).unwrap();
    assert_eq!(fixture["numpy"], "2.5.4");
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let op = BinaryOp::from_id(case["op"].as_str().unwrap()).unwrap();
        let a = packed(&case["a"]);
        let b = packed(&case["b"]);
        let expected = packed(&case["expected"]);
        let label = format!("case {i}: {:?} a={:?} b={:?}", op, a.dtype(), b.dtype());
        let compute = DType::promote(a.dtype(), b.dtype()).unwrap();
        let expr =
            Expr::binary(op, &input(packed(&case["a"])), &input(packed(&case["b"]))).unwrap();
        assert_eq!(expr.dtype(), expected.dtype(), "{label}");
        let budget = MemoryBudget::new(Device::Cpu, 1 << 20);
        let out = kernels::binary(op, &a, &b, compute, &budget, &CancellationToken::new()).unwrap();
        assert_eq!(out.shape, expected.shape, "{label}");
        assert_eq!(out.buffer, expected.buffer, "{label}");
        drop(out);
        assert_eq!(budget.usage().used, 0, "{label}");
    }
}

#[test]
fn bitwise_rejections_and_gpu_admission() {
    let dense =
        |buffer: Buffer| DenseArray::new(vec![buffer.len()], buffer, Reservation::none()).unwrap();
    let i64s = input(dense(Buffer::I64(vec![1, 2])));
    let u64s = input(dense(Buffer::U64(vec![1, 2])));
    let floats = input(dense(Buffer::F64(vec![1.0, 2.0])));
    let i32s = input(dense(Buffer::I32(vec![1, 2])));
    for op in [BinaryOp::BitAnd, BinaryOp::LeftShift] {
        // int64 with uint64 promotes to float64, which has no bitwise loop.
        assert_eq!(
            Expr::binary(op, &i64s, &u64s).unwrap_err().code,
            ErrorCode::DTypeUnsupported
        );
        assert_eq!(
            Expr::binary(op, &floats, &floats).unwrap_err().code,
            ErrorCode::DTypeUnsupported
        );
        // A JS number is a weak float scalar.
        assert_eq!(
            Expr::binary(op, &i32s, &Expr::scalar(Scalar::Float(1.0)))
                .unwrap_err()
                .code,
            ErrorCode::DTypeUnsupported
        );
        let ok = Expr::binary(op, &i32s, &Expr::scalar(Scalar::Int(1))).unwrap();
        assert_eq!(ok.dtype(), DType::Int32);
        assert_eq!(
            nmath_core::gpu::validate(&ok).unwrap_err().code,
            ErrorCode::GpuUnsupportedOperation
        );
    }
    let logical = Expr::binary(BinaryOp::LogicalAnd, &floats, &i32s).unwrap();
    assert_eq!(logical.dtype(), DType::Bool);
    let strings = input(dense(Buffer::Unicode {
        width: 1,
        values: vec![97, 98],
    }));
    assert_eq!(
        Expr::binary(BinaryOp::BitOr, &strings, &strings)
            .unwrap_err()
            .code,
        ErrorCode::DTypeUnsupported
    );
    assert_eq!(
        Expr::binary(BinaryOp::LogicalOr, &strings, &strings)
            .unwrap_err()
            .code,
        ErrorCode::DTypeUnsupported
    );
}

#[test]
fn bitwise_budgets_and_cancellation() {
    let a = DenseArray::new(vec![4], Buffer::I8(vec![1, 2, 3, 4]), Reservation::none()).unwrap();
    let b = DenseArray::new(vec![1], Buffer::I16(vec![1]), Reservation::none()).unwrap();
    let token = CancellationToken::new();
    // int16 result (8 bytes) plus the widened int8 operand (8 bytes).
    for limit in [0, 8, 15] {
        let budget = MemoryBudget::new(Device::Cpu, limit);
        assert_eq!(
            kernels::binary(BinaryOp::BitXor, &a, &b, DType::Int16, &budget, &token)
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded
        );
        assert_eq!(budget.usage().used, 0);
    }
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let budget = MemoryBudget::new(Device::Cpu, 1 << 10);
    assert_eq!(
        kernels::binary(
            BinaryOp::LogicalAnd,
            &a,
            &b,
            DType::Int16,
            &budget,
            &cancelled
        )
        .unwrap_err()
        .code,
        ErrorCode::Cancelled
    );
}
