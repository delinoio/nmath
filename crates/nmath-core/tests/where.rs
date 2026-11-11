// SPDX-License-Identifier: Apache-2.0
use nmath_core::{
    Buffer, CancellationToken, DType, DenseArray, Device, ErrorCode, Expr, NdArray, Scalar,
    Selector, Value,
    expr::JobEnv,
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

fn operand(j: &Json) -> Expr {
    if let Some(array) = j.get("array") {
        return Expr::input(&NdArray::from_dense(1, packed(array)).unwrap());
    }
    let value = &j["value"];
    Expr::scalar(match j["scalar"].as_str().unwrap() {
        "bool" => Scalar::Bool(value.as_bool().unwrap()),
        "int" => Scalar::Int(value.as_i64().unwrap()),
        "float" => Scalar::Float(value.as_f64().unwrap()),
        "complex" => Scalar::Complex([value[0].as_f64().unwrap(), value[1].as_f64().unwrap()]),
        _ => panic!("unknown fixture scalar"),
    })
}

fn eval(
    expr: &Expr,
    budget: &MemoryBudget,
    token: CancellationToken,
) -> nmath_core::Result<DenseArray> {
    let mut job = JobEnv::new(Device::Cpu, budget.clone(), token);
    job.freeze(expr)?;
    match job.eval(expr)? {
        Value::Dense(v) => Ok(v),
        Value::View(_) => panic!("where must produce independent storage"),
    }
}

fn dense(shape: Vec<usize>, buffer: Buffer) -> DenseArray {
    DenseArray::new(shape, buffer, Reservation::none()).unwrap()
}

#[test]
fn fixed_numpy_where_oracle() {
    let fixture: Json = serde_json::from_str(include_str!("fixtures/where/oracle.json")).unwrap();
    assert_eq!(fixture["numpy"], "2.5.4");
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let condition = Expr::input(&NdArray::from_dense(1, packed(&case["condition"])).unwrap());
        let x = operand(&case["x"]);
        let y = operand(&case["y"]);
        let expected = packed(&case["expected"]);
        let label = format!("case {i}: x={} y={}", case["x"], case["y"]);
        let expr = Expr::select(&condition, &x, &y).unwrap();
        assert_eq!(expr.dtype(), expected.dtype(), "{label}");
        assert_eq!(expr.shape(), expected.shape.as_slice(), "{label}");
        let budget = MemoryBudget::new(Device::Cpu, 1 << 20);
        let out = eval(&expr, &budget, CancellationToken::new()).unwrap();
        assert_eq!(out.shape, expected.shape, "{label}");
        assert_eq!(out.buffer, expected.buffer, "{label}");
    }
}

#[test]
fn where_validation_and_gpu_admission() {
    let condition = Expr::input(
        &NdArray::from_dense(1, dense(vec![2], Buffer::Bool(vec![true, false]))).unwrap(),
    );
    let ints =
        Expr::input(&NdArray::from_dense(1, dense(vec![2], Buffer::U8(vec![1, 2]))).unwrap());
    let floats =
        Expr::input(&NdArray::from_dense(1, dense(vec![2], Buffer::F32(vec![1.0, 2.0]))).unwrap());
    let selected = Expr::select(&condition, &floats, &floats).unwrap();
    assert_eq!(selected.operation_id(), "where");
    assert_eq!(
        nmath_core::gpu::validate(&selected).unwrap_err().code,
        ErrorCode::GpuUnsupportedOperation
    );

    // NEP 50: an out-of-range Python integer fails instead of wrapping.
    assert_eq!(
        Expr::select(&condition, &ints, &Expr::scalar(Scalar::Int(-1)))
            .unwrap_err()
            .code,
        ErrorCode::DTypeUnsupported
    );
    let three = Expr::input(
        &NdArray::from_dense(1, dense(vec![3], Buffer::F32(vec![1.0, 2.0, 3.0]))).unwrap(),
    );
    assert_eq!(
        Expr::select(&condition, &floats, &three).unwrap_err().code,
        ErrorCode::ShapeBroadcast
    );
    let other_context =
        Expr::input(&NdArray::from_dense(2, dense(vec![2], Buffer::F32(vec![1.0, 2.0]))).unwrap());
    assert_eq!(
        Expr::select(&condition, &floats, &other_context)
            .unwrap_err()
            .code,
        ErrorCode::ContextMismatch
    );
    let strings = Expr::input(
        &NdArray::from_dense(
            1,
            dense(
                vec![2],
                Buffer::Unicode {
                    width: 1,
                    values: vec![b'a' as u32, b'b' as u32],
                },
            ),
        )
        .unwrap(),
    );
    for (c, x, y) in [
        (&strings, &floats, &floats),
        (&condition, &strings, &strings),
        (&condition, &floats, &strings),
    ] {
        assert_eq!(
            Expr::select(c, x, y).unwrap_err().code,
            ErrorCode::DTypeUnsupported
        );
    }
}

#[test]
fn where_budgets_snapshots_and_cancellation() {
    let condition = dense(vec![2, 1], Buffer::I32(vec![0, 3]));
    let x = dense(vec![3], Buffer::I16(vec![1, 2, 3]));
    let y = dense(vec![1], Buffer::I64(vec![-1]));
    let token = CancellationToken::new();
    // The output needs 48 bytes; the bool condition copy and the int64 x copy
    // must fit as well.
    for limit in [0, 47, 48 + 2, 48 + 2 + 23] {
        let budget = MemoryBudget::new(Device::Cpu, limit);
        assert_eq!(
            kernels::select(&condition, &x, &y, DType::Int64, &budget, &token)
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded,
            "limit {limit}"
        );
        assert_eq!(budget.usage().used, 0);
    }
    let budget = MemoryBudget::new(Device::Cpu, 48 + 2 + 24);
    let out = kernels::select(&condition, &x, &y, DType::Int64, &budget, &token).unwrap();
    assert_eq!(out.buffer, Buffer::I64(vec![-1, -1, -1, 1, 2, 3]));
    drop(out);
    assert_eq!(budget.usage().used, 0);

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert_eq!(
        kernels::select(&condition, &x, &y, DType::Int64, &budget, &cancelled)
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );

    // Inputs are frozen at job start; later writes do not change the result.
    let source = NdArray::from_dense(1, dense(vec![2], Buffer::Bool(vec![true, false]))).unwrap();
    let expr = Expr::select(
        &Expr::input(&source),
        &Expr::scalar(Scalar::Int(1)),
        &Expr::scalar(Scalar::Int(0)),
    )
    .unwrap();
    let budget = MemoryBudget::new(Device::Cpu, 1 << 10);
    let mut job = JobEnv::new(Device::Cpu, budget.clone(), CancellationToken::new());
    job.freeze(&expr).unwrap();
    let turn = source.write_ticket().wait(&token).unwrap();
    source
        .assign(
            &turn,
            &[Selector::Index(0)],
            &dense(vec![], Buffer::Bool(vec![false])),
            &token,
        )
        .unwrap();
    drop(turn);
    let Value::Dense(out) = job.eval(&expr).unwrap() else {
        panic!("where must produce independent storage");
    };
    assert_eq!(out.buffer, Buffer::I64(vec![1, 0]));
}
