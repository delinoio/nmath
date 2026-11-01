// SPDX-License-Identifier: Apache-2.0
use nmath_core::{
    Buffer, CancellationToken, DType, DenseArray, Device, ErrorCode, Expr, NdArray,
    buffer::Float16,
    memory::{MemoryBudget, Reservation},
    ufunc::{self, MathOp},
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

fn run(op: MathOp, input: &DenseArray) -> DenseArray {
    let budget = MemoryBudget::new(Device::Cpu, 1 << 24);
    let out = ufunc::execute(op, input, &budget, &CancellationToken::new()).unwrap();
    assert!(budget.usage().used > 0);
    out
}

#[test]
fn fixed_numpy_math_oracle() {
    let fixture: Json = serde_json::from_str(include_str!("fixtures/math/oracle.json")).unwrap();
    assert_eq!(fixture["numpy"], "2.5.4");
    let mut ops = std::collections::HashSet::new();
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let op = MathOp::from_id(case["op"].as_str().unwrap()).unwrap();
        ops.insert(op);
        let input = packed(&case["input"]);
        let numpy = packed(&case["numpy"]);
        let label = format!("case {i}: {} {:?}", case["op"], input.dtype());
        assert_eq!(
            op.result_dtype(input.dtype()).unwrap(),
            numpy.dtype(),
            "{label}"
        );
        let out = run(op, &input);
        assert_eq!(out.shape, numpy.shape, "{label}");
        match case.get("reference") {
            Some(reference) => {
                let reference = packed(reference);
                let max = case["maxUlp"].as_u64().unwrap() as u128;
                for (j, d) in ulps(&out.buffer, &reference.buffer).into_iter().enumerate() {
                    assert!(
                        d.is_some_and(|d| d <= max),
                        "{label} element {j}: {d:?} ULP > {max}, got {:?}, reference {:?}",
                        out.buffer.get(j),
                        reference.buffer.get(j)
                    );
                }
            }
            None if matches!(
                numpy.dtype(),
                DType::Float16
                    | DType::Float32
                    | DType::Float64
                    | DType::Complex64
                    | DType::Complex128
            ) =>
            {
                for (j, d) in ulps(&out.buffer, &numpy.buffer).into_iter().enumerate() {
                    assert_eq!(d, Some(0), "{label} element {j}: {:?}", out.buffer.get(j));
                }
            }
            None => assert_eq!(out.buffer, numpy.buffer, "{label}"),
        }
    }
    assert_eq!(
        ops.len(),
        MathOp::ALL.len(),
        "every function has oracle cases"
    );
}

#[test]
fn rejected_dtypes_and_gpu_admission() {
    let array = |buffer: Buffer| {
        Expr::input(&NdArray::from_dense(
            1,
            DenseArray::new(vec![buffer.len()], buffer, Reservation::none()).unwrap(),
        ))
    };
    let complex = array(Buffer::C128(vec![[1.0, 2.0]]));
    let ints = array(Buffer::I32(vec![1, 0]));
    let flags = array(Buffer::Bool(vec![true]));
    let strings = array(Buffer::Unicode {
        width: 1,
        values: vec![b'a' as u32],
    });
    for (op, input) in [
        (MathOp::Sqrt, &complex),
        (MathOp::Absolute, &complex),
        (MathOp::Floor, &complex),
        (MathOp::Signbit, &complex),
        (MathOp::Reciprocal, &ints),
        (MathOp::Sign, &flags),
        (MathOp::Exp, &strings),
        (MathOp::Isnan, &strings),
    ] {
        assert_eq!(
            Expr::math(op, input).unwrap_err().code,
            ErrorCode::DTypeUnsupported,
            "{op:?}"
        );
    }
    let roots = Expr::math(MathOp::Sqrt, &ints).unwrap();
    assert_eq!(roots.dtype(), DType::Float64);
    assert_eq!(roots.operation_id(), "sqrt");
    assert_eq!(
        Expr::math(MathOp::Conjugate, &complex).unwrap().dtype(),
        DType::Complex128
    );
    assert_eq!(
        nmath_core::gpu::validate(
            &Expr::math(MathOp::Sqrt, &array(Buffer::F32(vec![4.0]))).unwrap()
        )
        .unwrap_err()
        .code,
        ErrorCode::GpuUnsupportedOperation
    );
    assert_eq!(
        MathOp::from_id("nope").unwrap_err().code,
        ErrorCode::IndexInvalid
    );
}

#[test]
fn math_budgets_and_cancellation() {
    let input =
        DenseArray::new(vec![4], Buffer::I16(vec![1, 4, 9, 16]), Reservation::none()).unwrap();
    // float32 result (16 bytes) plus one float64 scratch copy (32 bytes).
    for limit in [0, 16, 47] {
        let budget = MemoryBudget::new(Device::Cpu, limit);
        assert_eq!(
            ufunc::execute(MathOp::Sqrt, &input, &budget, &CancellationToken::new())
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded
        );
        assert_eq!(budget.usage().used, 0);
    }
    let budget = MemoryBudget::new(Device::Cpu, 48);
    let out = ufunc::execute(MathOp::Sqrt, &input, &budget, &CancellationToken::new()).unwrap();
    assert_eq!(out.buffer, Buffer::F32(vec![1.0, 2.0, 3.0, 4.0]));
    assert_eq!(budget.usage().used, 16);
    drop(out);
    assert_eq!(budget.usage().used, 0);
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert_eq!(
        ufunc::execute(MathOp::Exp, &input, &budget, &cancelled)
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
}
