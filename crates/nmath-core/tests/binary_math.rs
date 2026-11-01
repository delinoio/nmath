// SPDX-License-Identifier: Apache-2.0
use nmath_core::{
    BinaryOp, Buffer, CancellationToken, DType, DenseArray, Device, ErrorCode, Expr, NdArray,
    Scalar,
    buffer::Float16,
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

/// Maps a float's bits onto a line where adjacent floats differ by one.
fn ordered(bits: u64, sign: u64) -> i128 {
    if bits & sign != 0 {
        -((bits & !sign) as i128)
    } else {
        bits as i128
    }
}

/// ULP distance, or `None` when exactly one side is NaN. NaN payloads are
/// ignored.
fn ulps(actual: &Buffer, expected: &Buffer) -> Vec<Option<u128>> {
    fn pairs<T: Copy>(
        a: &[T],
        b: &[T],
        nan: impl Fn(T) -> bool,
        bits: impl Fn(T) -> u64,
        sign: u64,
    ) -> Vec<Option<u128>> {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| match (nan(x), nan(y)) {
                (true, true) => Some(0),
                (false, false) => Some(ordered(bits(x), sign).abs_diff(ordered(bits(y), sign))),
                _ => None,
            })
            .collect()
    }
    match (actual, expected) {
        (Buffer::F16(a), Buffer::F16(b)) => pairs(
            a,
            b,
            |x: Float16| x.is_nan(),
            |x| x.to_bits() as u64,
            1 << 15,
        ),
        (Buffer::F32(a), Buffer::F32(b)) => {
            pairs(a, b, |x: f32| x.is_nan(), |x| x.to_bits() as u64, 1 << 31)
        }
        (Buffer::F64(a), Buffer::F64(b)) => pairs(a, b, |x: f64| x.is_nan(), f64::to_bits, 1 << 63),
        (Buffer::C64(a), Buffer::C64(b)) => {
            let flat = |v: &[[f32; 2]]| v.iter().flatten().copied().collect::<Vec<_>>();
            pairs(
                &flat(a),
                &flat(b),
                |x: f32| x.is_nan(),
                |x| x.to_bits() as u64,
                1 << 31,
            )
        }
        (Buffer::C128(a), Buffer::C128(b)) => {
            let flat = |v: &[[f64; 2]]| v.iter().flatten().copied().collect::<Vec<_>>();
            pairs(
                &flat(a),
                &flat(b),
                |x: f64| x.is_nan(),
                f64::to_bits,
                1 << 63,
            )
        }
        _ => panic!("ULP comparison requires matching float dtypes"),
    }
}

fn input(a: DenseArray) -> Expr {
    Expr::input(&NdArray::from_dense(1, a))
}

#[test]
fn fixed_numpy_binary_math_oracle() {
    let fixture: Json =
        serde_json::from_str(include_str!("fixtures/binary_math/oracle.json")).unwrap();
    assert_eq!(fixture["numpy"], "2.5.4");
    let mut ops = std::collections::HashSet::new();
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let op = BinaryOp::from_id(case["op"].as_str().unwrap()).unwrap();
        ops.insert(op);
        let a = packed(&case["a"]);
        let b = packed(&case["b"]);
        let expected = packed(&case["expected"]);
        let label = format!("case {i}: {:?} a={:?} b={:?}", op, a.dtype(), b.dtype());
        let expr =
            Expr::binary(op, &input(packed(&case["a"])), &input(packed(&case["b"]))).unwrap();
        assert_eq!(expr.dtype(), expected.dtype(), "{label}");
        let compute = nmath_core::ufunc::binary_loop_dtype(op, a.dtype(), b.dtype()).unwrap();
        let budget = MemoryBudget::new(Device::Cpu, 1 << 20);
        let out = kernels::binary(op, &a, &b, compute, &budget, &CancellationToken::new()).unwrap();
        assert_eq!(out.shape, expected.shape, "{label}");
        if matches!(
            expected.dtype(),
            DType::Float16 | DType::Float32 | DType::Float64 | DType::Complex64 | DType::Complex128
        ) {
            for (j, d) in ulps(&out.buffer, &expected.buffer).into_iter().enumerate() {
                assert_eq!(
                    d,
                    Some(0),
                    "{label} element {j}: got {:?}, expected {:?}",
                    out.buffer.get(j),
                    expected.buffer.get(j)
                );
            }
        } else {
            assert_eq!(out.buffer, expected.buffer, "{label}");
        }
        drop(out);
        assert_eq!(budget.usage().used, 0, "{label}");
    }
    assert_eq!(ops.len(), 13, "every binary math function has oracle cases");
}

#[test]
fn binary_math_rejections_and_gpu_admission() {
    let dense =
        |buffer: Buffer| DenseArray::new(vec![buffer.len()], buffer, Reservation::none()).unwrap();
    let complex = input(dense(Buffer::C64(vec![[1.0, 2.0]])));
    let floats = input(dense(Buffer::F32(vec![1.0])));
    for op in [
        BinaryOp::Fmod,
        BinaryOp::FloorDivide,
        BinaryOp::Arctan2,
        BinaryOp::Copysign,
    ] {
        assert_eq!(
            Expr::binary(op, &complex, &complex).unwrap_err().code,
            ErrorCode::DTypeUnsupported
        );
    }
    assert_eq!(
        Expr::binary(BinaryOp::Maximum, &complex, &complex)
            .unwrap()
            .dtype(),
        DType::Complex64
    );
    let strings = input(dense(Buffer::Unicode {
        width: 1,
        values: vec![97],
    }));
    assert_eq!(
        Expr::binary(BinaryOp::Maximum, &strings, &strings)
            .unwrap_err()
            .code,
        ErrorCode::DTypeUnsupported
    );
    // NEP 50: a weak float scalar keeps float32.
    let weak = Expr::binary(BinaryOp::Hypot, &floats, &Expr::scalar(Scalar::Float(2.0))).unwrap();
    assert_eq!(weak.dtype(), DType::Float32);
    assert_eq!(
        nmath_core::gpu::validate(&Expr::binary(BinaryOp::Maximum, &floats, &floats).unwrap())
            .unwrap_err()
            .code,
        ErrorCode::GpuUnsupportedOperation
    );
    let _ = Float16::ZERO;
}

#[test]
fn binary_math_budgets_and_cancellation() {
    let a = DenseArray::new(vec![4], Buffer::I16(vec![1, 2, 3, 4]), Reservation::none()).unwrap();
    let b = DenseArray::new(vec![1], Buffer::I16(vec![3]), Reservation::none()).unwrap();
    // float32 result (16 bytes) plus casts of both operands (16 + 4 bytes).
    for limit in [0, 16, 35] {
        let budget = MemoryBudget::new(Device::Cpu, limit);
        assert_eq!(
            kernels::binary(
                BinaryOp::Hypot,
                &a,
                &b,
                DType::Int16,
                &budget,
                &CancellationToken::new()
            )
            .unwrap_err()
            .code,
            ErrorCode::MemoryBudgetExceeded
        );
        assert_eq!(budget.usage().used, 0);
    }
    let budget = MemoryBudget::new(Device::Cpu, 36);
    let out = kernels::binary(
        BinaryOp::FloorDivide,
        &a,
        &b,
        DType::Int16,
        &budget,
        &CancellationToken::new(),
    )
    .unwrap();
    assert_eq!(out.buffer, Buffer::I16(vec![0, 0, 1, 1]));
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert_eq!(
        kernels::binary(BinaryOp::Maximum, &a, &b, DType::Int16, &budget, &cancelled)
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
}
