// Compile the isolated backend before the coordinator adds its lib.rs wiring.
pub use nmath_core::{array, buffer, dtype, error, expr, kernels, memory, shape, sync};
#[path = "../src/gpu.rs"]
mod gpu;

use memory::{MemoryBudget, Reservation};
use nmath_core::{
    BinaryOp, Buffer, CancellationToken, DType, DenseArray, Device, Expr, NdArray, ReduceOp,
    Scalar, UnaryOp,
};

fn input(shape: Vec<usize>, buffer: Buffer) -> Expr {
    Expr::input(&NdArray::from_dense(
        1,
        DenseArray::new(shape, buffer, Reservation::none()).unwrap(),
    ))
}

fn execute(expr: &Expr) -> error::Result<DenseArray> {
    let cpu = MemoryBudget::new(Device::Cpu, 1 << 26);
    let gpu_budget = MemoryBudget::new(Device::Gpu, 1 << 26);
    let inputs = expr.inputs();
    let frozen: Vec<_> = inputs.iter().map(|a| a.snapshot(&cpu).unwrap()).collect();
    gpu::execute(expr, &cpu, &gpu_budget, &CancellationToken::new(), |a| {
        inputs
            .iter()
            .position(|b| a.shares_storage(b) && a.shape() == b.shape())
            .map(|i| &frozen[i])
            .ok_or_else(|| error::Error::new(error::ErrorCode::IndexInvalid, "Missing snapshot."))
    })
}

#[test]
fn static_validation_preserves_promotion() {
    let a = input(vec![2], Buffer::I32(vec![1, 2]));
    for op in [ReduceOp::Sum, ReduceOp::Mean] {
        assert_eq!(
            gpu::validate(&Expr::reduce(op, &a, None).unwrap())
                .unwrap_err()
                .code,
            error::ErrorCode::GpuUnsupportedOperation
        );
    }
    assert!(gpu::validate(&Expr::binary(BinaryOp::Div, &a, &a).unwrap()).is_err());
    assert!(gpu::validate(&Expr::scalar(Scalar::Float(1.0))).is_err());
    let f = input(vec![2], Buffer::F32(vec![1.0, 2.0]));
    assert!(
        gpu::validate(&Expr::binary(BinaryOp::Add, &f, &Expr::scalar(Scalar::Float(1.0))).unwrap())
            .is_ok()
    );
    assert!(gpu::validate(&Expr::binary(BinaryOp::Add, &a, &f).unwrap()).is_err());
    assert!(gpu::validate(&Expr::cast(&a, DType::Float32).unwrap()).is_err());
}

// Explicit opt-in: a normal test run must not silently count no-adapter skips
// as real GPU acceptance. Run with --ignored --nocapture on a supported
// machine.
#[test]
#[ignore = "requires an actual supported native GPU"]
fn native_gpu_operations() {
    for op in [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Mod,
        BinaryOp::Pow,
    ] {
        let a = input(vec![2, 1], Buffer::F32(vec![2.0, 4.0]));
        let b = input(vec![2], Buffer::F32(vec![1.0, 2.0]));
        let expr = Expr::binary(op, &a, &b).unwrap();
        let expected = match op {
            BinaryOp::Add => vec![3.0, 4.0, 5.0, 6.0],
            BinaryOp::Sub => vec![1.0, 0.0, 3.0, 2.0],
            BinaryOp::Mul => vec![2.0, 4.0, 4.0, 8.0],
            BinaryOp::Div => vec![2.0, 1.0, 4.0, 2.0],
            BinaryOp::Mod => vec![0.0, 0.0, 0.0, 0.0],
            BinaryOp::Pow => vec![2.0, 4.0, 4.0, 16.0],
        };
        assert_eq!(
            execute(&expr).unwrap().buffer,
            Buffer::F32(expected),
            "{op:?}"
        );
    }
    for op in [UnaryOp::Pos, UnaryOp::Neg] {
        let expr = Expr::unary(op, &input(vec![2], Buffer::I32(vec![i32::MIN, 3]))).unwrap();
        assert_eq!(
            execute(&expr).unwrap().buffer,
            Buffer::I32(vec![i32::MIN, if op == UnaryOp::Neg { -3 } else { 3 }])
        );
    }
    for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Min, ReduceOp::Max] {
        let expr = Expr::reduce(
            op,
            &input(vec![2, 2], Buffer::F32(vec![1.0, 2.0, 3.0, 4.0])),
            Some(vec![1]),
        )
        .unwrap();
        let expected = match op {
            ReduceOp::Sum => vec![3.0, 7.0],
            ReduceOp::Mean => vec![1.5, 3.5],
            ReduceOp::Min => vec![1.0, 3.0],
            ReduceOp::Max => vec![2.0, 4.0],
        };
        assert_eq!(execute(&expr).unwrap().buffer, Buffer::F32(expected));
    }
    let a = input(vec![2, 2], Buffer::U32(vec![1, 2, 3, 4]));
    assert_eq!(
        execute(&Expr::matmul(&a, &a).unwrap()).unwrap().buffer,
        Buffer::U32(vec![7, 10, 15, 22])
    );
}

#[test]
#[ignore = "requires an actual supported native GPU"]
fn native_integer_edges() {
    let a = input(vec![4], Buffer::I32(vec![i32::MAX, i32::MIN, -5, 5]));
    let b = input(vec![4], Buffer::I32(vec![2, -1, 3, -3]));
    for (op, expected) in [
        (BinaryOp::Add, vec![i32::MIN + 1, i32::MAX, -2, 2]),
        (BinaryOp::Sub, vec![i32::MAX - 2, i32::MIN + 1, -8, 8]),
        (BinaryOp::Mul, vec![-2, i32::MIN, -15, -15]),
        (BinaryOp::Mod, vec![1, 0, 1, -1]),
    ] {
        assert_eq!(
            execute(&Expr::binary(op, &a, &b).unwrap()).unwrap().buffer,
            Buffer::I32(expected),
            "{op:?}"
        );
    }
    let a = input(vec![3], Buffer::U32(vec![u32::MAX, 2, 0]));
    let b = input(vec![3], Buffer::U32(vec![2, 32, 0]));
    for (op, expected) in [
        (BinaryOp::Add, vec![1, 34, 0]),
        (BinaryOp::Sub, vec![u32::MAX - 2, u32::MAX - 29, 0]),
        (BinaryOp::Mul, vec![u32::MAX - 1, 64, 0]),
        (BinaryOp::Mod, vec![1, 2, 0]),
        (BinaryOp::Pow, vec![1, 0, 1]),
    ] {
        assert_eq!(
            execute(&Expr::binary(op, &a, &b).unwrap()).unwrap().buffer,
            Buffer::U32(expected),
            "{op:?}"
        );
    }
    assert_eq!(
        execute(&Expr::unary(UnaryOp::Neg, &a).unwrap())
            .unwrap()
            .buffer,
        Buffer::U32(vec![1, u32::MAX - 1, 0])
    );
    let a = input(vec![1], Buffer::I32(vec![2]));
    assert_eq!(
        execute(&Expr::binary(BinaryOp::Pow, &a, &Expr::scalar(Scalar::Int(-1))).unwrap())
            .unwrap_err()
            .code,
        error::ErrorCode::DTypeUnsupported
    );
    for dtype in [Buffer::I32(vec![-1, 5]), Buffer::U32(vec![u32::MAX, 1])] {
        let a = input(vec![2], dtype.clone());
        for op in [ReduceOp::Min, ReduceOp::Max] {
            let expected = match (&dtype, op) {
                (Buffer::I32(_), ReduceOp::Min) => Buffer::I32(vec![-1]),
                (Buffer::I32(_), _) => Buffer::I32(vec![5]),
                (_, ReduceOp::Min) => Buffer::U32(vec![1]),
                _ => Buffer::U32(vec![u32::MAX]),
            };
            assert_eq!(
                execute(&Expr::reduce(op, &a, None).unwrap())
                    .unwrap()
                    .buffer,
                expected
            );
        }
    }
}

#[test]
#[ignore = "requires an actual supported native GPU"]
fn native_empty_shapes_and_reduction_axes() {
    let a = input(vec![2, 0], Buffer::F32(vec![]));
    assert_eq!(
        execute(&Expr::reduce(ReduceOp::Sum, &a, Some(vec![1])).unwrap())
            .unwrap()
            .buffer,
        Buffer::F32(vec![0.0; 2])
    );
    let mean = execute(&Expr::reduce(ReduceOp::Mean, &a, Some(vec![1])).unwrap()).unwrap();
    let Buffer::F32(mean) = mean.buffer else {
        panic!()
    };
    assert!(mean.iter().all(|v| v.is_nan()));
    assert!(gpu::validate(&Expr::reduce(ReduceOp::Min, &a, Some(vec![1])).unwrap()).is_err());
    assert_eq!(
        execute(&Expr::binary(BinaryOp::Add, &a, &a).unwrap())
            .unwrap()
            .buffer,
        Buffer::F32(vec![])
    );
    let a = input(
        vec![2, 2, 2],
        Buffer::F32((1..=8).map(|v| v as f32).collect()),
    );
    assert_eq!(
        execute(&Expr::reduce(ReduceOp::Sum, &a, Some(vec![0, 2])).unwrap())
            .unwrap()
            .buffer,
        Buffer::F32(vec![14.0, 22.0])
    );
    assert_eq!(
        execute(&Expr::reduce(ReduceOp::Sum, &a, Some(vec![])).unwrap())
            .unwrap()
            .buffer,
        Buffer::F32((1..=8).map(|v| v as f32).collect())
    );
}

#[path = "gpu/oracle.rs"]
mod gpu_oracle;

#[test]
#[ignore = "requires an actual supported native GPU"]
fn native_numpy_float32_special_values() {
    let a = input(
        vec![9, 1],
        Buffer::F32(
            gpu_oracle::INPUT_A
                .iter()
                .map(|&w| f32::from_bits(w))
                .collect(),
        ),
    );
    let b = input(
        vec![1, 9],
        Buffer::F32(
            gpu_oracle::INPUT_B
                .iter()
                .map(|&w| f32::from_bits(w))
                .collect(),
        ),
    );
    for &(op, expected) in gpu_oracle::OUTPUTS {
        let expr = Expr::binary(BinaryOp::from_id(op).unwrap(), &a, &b).unwrap();
        let Buffer::F32(actual) = execute(&expr).unwrap().buffer else {
            panic!()
        };
        for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            if f32::from_bits(expected).is_nan() {
                assert!(actual.is_nan(), "{op}, index {i}: {actual}");
            } else {
                assert_eq!(actual.to_bits(), expected, "{op}, index {i}: {actual}");
            }
        }
    }
}

#[test]
#[ignore = "requires an actual supported native GPU"]
fn native_float32_subnormals_and_overflow() {
    let values = vec![
        f32::from_bits(1),
        f32::from_bits(0x80000001),
        f32::MIN_POSITIVE,
        f32::MAX,
    ];
    let a = input(vec![4], Buffer::F32(values));
    let out = execute(&Expr::binary(BinaryOp::Mul, &a, &Expr::scalar(Scalar::Float(2.0))).unwrap())
        .unwrap();
    let Buffer::F32(out) = out.buffer else {
        panic!()
    };
    assert_eq!(
        out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vec![2, 0x80000002, 0x01000000, 0x7f800000]
    );
}

#[test]
#[ignore = "requires an actual supported native GPU"]
fn native_float32_chunked_reductions() {
    let a = input(vec![2, 600], Buffer::F32(vec![1.0; 1200]));
    assert_eq!(
        execute(&Expr::reduce(ReduceOp::Sum, &a, Some(vec![-1])).unwrap())
            .unwrap()
            .buffer,
        Buffer::F32(vec![600.0; 2])
    );
    assert_eq!(
        execute(&Expr::reduce(ReduceOp::Mean, &a, Some(vec![-1])).unwrap())
            .unwrap()
            .buffer,
        Buffer::F32(vec![1.0; 2])
    );
    assert_eq!(
        execute(&Expr::matmul(&a, &input(vec![600], Buffer::F32(vec![2.0; 600]))).unwrap())
            .unwrap()
            .buffer,
        Buffer::F32(vec![1200.0; 2])
    );
    let a = input(vec![4101], Buffer::F32(vec![2.0; 4101]));
    assert_eq!(
        execute(&Expr::binary(BinaryOp::Mul, &a, &a).unwrap())
            .unwrap()
            .buffer,
        Buffer::F32(vec![4.0; 4101])
    );
}

#[test]
#[ignore = "requires an actual supported native GPU"]
fn native_integer_matmul_and_broadcast() {
    let a = input(vec![2, 2], Buffer::U32(vec![1, 2, 3, 4]));
    assert_eq!(
        execute(&Expr::matmul(&a, &a).unwrap()).unwrap().buffer,
        Buffer::U32(vec![7, 10, 15, 22])
    );
    let b = input(vec![2], Buffer::U32(vec![2, 3]));
    assert_eq!(
        execute(&Expr::binary(BinaryOp::Mul, &a, &b).unwrap())
            .unwrap()
            .buffer,
        Buffer::U32(vec![2, 6, 6, 12])
    );
    let a = input(vec![300], Buffer::I32(vec![2; 300]));
    assert_eq!(
        execute(&Expr::matmul(&a, &a).unwrap()).unwrap().buffer,
        Buffer::I32(vec![1200])
    );
}

#[path = "gpu/float32_edges.rs"]
mod float32_edges;

fn assert_float_words(actual: Buffer, expected: &[u32], label: &str) {
    let Buffer::F32(actual) = actual else {
        panic!("Expected float32")
    };
    assert_eq!(actual.len(), expected.len());
    for (i, (actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if f32::from_bits(expected).is_nan() {
            assert!(actual.is_nan(), "{label} at {i}: {actual:?}");
        } else {
            assert_eq!(actual.to_bits(), expected, "{label} at {i}");
        }
    }
}

#[test]
#[ignore = "requires an actual supported native GPU"]
fn native_float32_exact_edge_oracles() {
    let cases = float32_edges::CASES;
    let a = input(
        vec![cases.len()],
        Buffer::F32(cases.iter().map(|r| f32::from_bits(r[0])).collect()),
    );
    let b = input(
        vec![cases.len()],
        Buffer::F32(cases.iter().map(|r| f32::from_bits(r[1])).collect()),
    );
    for (column, op) in [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Mod,
    ]
    .into_iter()
    .enumerate()
    {
        let result = execute(&Expr::binary(op, &a, &b).unwrap()).unwrap();
        assert_float_words(
            result.buffer,
            &cases.iter().map(|r| r[column + 2]).collect::<Vec<_>>(),
            &format!("{op:?}"),
        );
    }
    for op in [UnaryOp::Pos, UnaryOp::Neg] {
        let result = execute(&Expr::unary(op, &a).unwrap()).unwrap();
        // Unary operations are bit operations, including NaN payloads.
        let Buffer::F32(actual) = result.buffer else {
            panic!()
        };
        for (v, row) in actual.iter().zip(cases) {
            assert_eq!(
                v.to_bits(),
                row[0] ^ if op == UnaryOp::Neg { 0x80000000 } else { 0 }
            );
        }
    }
}

#[test]
#[ignore = "requires an actual supported native GPU"]
fn native_float32_subnormal_reductions_and_matmul() {
    let a = input(vec![2, 600], Buffer::F32(vec![f32::from_bits(1); 1200]));
    for (op, bits) in [
        (ReduceOp::Sum, 600),
        (ReduceOp::Mean, 1),
        (ReduceOp::Min, 1),
        (ReduceOp::Max, 1),
    ] {
        assert_float_words(
            execute(&Expr::reduce(op, &a, Some(vec![1])).unwrap())
                .unwrap()
                .buffer,
            &[bits, bits],
            &format!("{op:?}"),
        );
    }
    let b = input(vec![600], Buffer::F32(vec![2.0; 600]));
    assert_float_words(
        execute(&Expr::matmul(&a, &b).unwrap()).unwrap().buffer,
        &[1200, 1200],
        "subnormal dot",
    );
    let a = input(
        vec![2, 2],
        Buffer::F32(vec![
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
        ]),
    );
    let b = input(vec![2, 2], Buffer::F32(vec![0.5, 0.0, 0.0, 0.5]));
    assert_float_words(
        execute(&Expr::matmul(&a, &b).unwrap()).unwrap().buffer,
        &[0x00400000, 0x80400000, 0x00400000, 0x80400000],
        "normal input underflow",
    );
    let a = input(
        vec![4],
        Buffer::F32(vec![
            f32::from_bits(1),
            f32::from_bits(0x80000001),
            0.0,
            -0.0,
        ]),
    );
    for (op, bits) in [(ReduceOp::Min, 0x80000001), (ReduceOp::Max, 1)] {
        assert_float_words(
            execute(&Expr::reduce(op, &a, None).unwrap())
                .unwrap()
                .buffer,
            &[bits],
            "subnormal order",
        );
    }
}

#[test]
fn float32_power_rejects_before_execution() {
    let a = input(vec![1], Buffer::F32(vec![2.0]));
    let power = Expr::binary(BinaryOp::Pow, &a, &a).unwrap();
    assert_eq!(
        gpu::validate(&power).unwrap_err().code,
        error::ErrorCode::GpuUnsupportedOperation
    );
    assert_eq!(
        gpu::validate(&Expr::unary(UnaryOp::Neg, &power).unwrap())
            .unwrap_err()
            .code,
        error::ErrorCode::GpuUnsupportedOperation
    );
}

#[test]
#[ignore = "requires an actual supported native GPU"]
fn native_float32_reduction_special_oracles() {
    // NumPy 2.5.4, float32, axis=1. In particular min/max distinguish
    // opposite signed zeros independently of their order in these rows.
    let a = input(
        vec![6, 2],
        Buffer::F32(vec![
            f32::NAN,
            1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            0.0,
            -0.0,
            f32::from_bits(1),
            f32::from_bits(0x80000001),
            f32::MAX,
            f32::MAX,
        ]),
    );
    for (op, expected) in [
        (ReduceOp::Sum, [0x7fc00000, 0x7fc00000, 0, 0, 0, 0x7f800000]),
        (
            ReduceOp::Mean,
            [0x7fc00000, 0x7fc00000, 0, 0, 0, 0x7f800000],
        ),
        (
            ReduceOp::Min,
            [
                0x7fc00000, 0xff800000, 0x80000000, 0x80000000, 0x80000001, 0x7f7fffff,
            ],
        ),
        (ReduceOp::Max, [0x7fc00000, 0x7f800000, 0, 0, 1, 0x7f7fffff]),
    ] {
        assert_float_words(
            execute(&Expr::reduce(op, &a, Some(vec![1])).unwrap())
                .unwrap()
                .buffer,
            &expected,
            &format!("{op:?}"),
        );
    }
}
