use super::*;

#[test]
fn cancellation_precedes_device_discovery() {
    let token = CancellationToken::new();
    token.cancel();
    let cpu = MemoryBudget::new(Device::Cpu, 0);
    let gpu = MemoryBudget::new(Device::Gpu, 0);
    assert!(matches!(
        Runtime::new(&cpu, &gpu, &token),
        Err(Error {
            code: ErrorCode::Cancelled,
            ..
        })
    ));
}

#[test]
#[ignore = "requires an actual supported native GPU"]
fn native_budgets_and_device_loss() {
    let cpu = MemoryBudget::new(Device::Cpu, 1024);
    let gpu = MemoryBudget::new(Device::Gpu, 32);
    let token = CancellationToken::new();
    let runtime = Runtime::new(&cpu, &gpu, &token).unwrap();
    eprintln!("Actual native adapter: {:?}", runtime.device.adapter_info());
    let a = runtime.upload(&Buffer::I32(vec![1, 2, 3, 4])).unwrap();
    assert_eq!(gpu.usage().used, 16);
    assert!(matches!(
        runtime.allocate(20, wgpu::BufferUsages::STORAGE, false),
        Err(Error {
            code: ErrorCode::MemoryBudgetExceeded,
            ..
        })
    ));
    assert_eq!(gpu.usage().used, 16);
    let _host = cpu.reserve(16).unwrap();
    assert_eq!(runtime.read_words(&a, 4).unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(gpu.usage().peak, 32);
    assert_eq!(gpu.usage().used, 16);
    drop(a);
    assert_eq!(gpu.usage().used, 0);
    runtime.device.destroy();
    let _ = runtime.device.poll(wgpu::PollType::Poll);
    assert_eq!(runtime.check().unwrap_err().code, ErrorCode::GpuDeviceLost);
}

#[test]
#[ignore = "requires an actual supported native GPU"]
fn native_cancellation_releases_all_allocations() {
    let cpu = MemoryBudget::new(Device::Cpu, 1024);
    let gpu = MemoryBudget::new(Device::Gpu, 1024);
    let token = CancellationToken::new();
    {
        let runtime = Runtime::new(&cpu, &gpu, &token).unwrap();
        let a = runtime.upload(&Buffer::I32(vec![1, 2, 3, 4])).unwrap();
        token.cancel();
        assert_eq!(
            runtime.read_words(&a, 4).unwrap_err().code,
            ErrorCode::Cancelled
        );
    }
    assert_eq!(gpu.usage().used, 0);
    assert_eq!(cpu.usage().used, 0);
}

#[test]
fn validate_all_shader_variants_without_a_driver() {
    use crate::kernels::{BinaryOp, UnaryOp};
    let mut graphs = Vec::new();
    for buffer in [
        Buffer::F32(vec![1.0; 4]),
        Buffer::I32(vec![1; 4]),
        Buffer::U32(vec![1; 4]),
    ] {
        let a = Expr::input(&NdArray::from_dense(
            1,
            DenseArray::new(vec![2, 2], buffer, Reservation::none()).unwrap(),
        ));
        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Mod,
            BinaryOp::Pow,
        ] {
            let e = Expr::binary(op, &a, &a).unwrap();
            if validate(&e).is_ok() {
                graphs.push(e);
            }
        }
        for op in [UnaryOp::Pos, UnaryOp::Neg] {
            graphs.push(Expr::unary(op, &a).unwrap());
        }
        for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Min, ReduceOp::Max] {
            let e = Expr::reduce(op, &a, Some(vec![1])).unwrap();
            if validate(&e).is_ok() {
                graphs.push(e);
            }
        }
        graphs.push(Expr::matmul(&a, &a).unwrap());
    }
    for e in graphs {
        let source = shader::generate(&e).unwrap();
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[test]
#[ignore = "requires an actual supported native GPU"]
fn native_cancellation_after_submission_drains_before_release() {
    let cpu = MemoryBudget::new(Device::Cpu, 1024);
    let gpu = MemoryBudget::new(Device::Gpu, 1024);
    let token = CancellationToken::new();
    let runtime = Runtime::new(&cpu, &gpu, &token).unwrap();
    let a = runtime.upload(&Buffer::I32(vec![1; 128])).unwrap();
    let b = runtime
        .allocate(512, wgpu::BufferUsages::COPY_DST, false)
        .unwrap();
    let mut encoder = runtime.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&a.buffer, 0, &b.buffer, 0, 512);
    runtime.queue.submit([encoder.finish()]);
    token.cancel();
    assert_eq!(runtime.drain().unwrap_err().code, ErrorCode::Cancelled);
    assert_eq!(gpu.usage().used, 1024);
    drop((a, b));
    assert_eq!(gpu.usage().used, 0);
}

#[test]
#[ignore = "requires an actual supported native GPU"]
fn native_precision_guard_rejects_observed_loss() {
    let cpu = MemoryBudget::new(Device::Cpu, 1024);
    let gpu = MemoryBudget::new(Device::Gpu, 1024);
    let token = CancellationToken::new();
    let runtime = Runtime::new(&cpu, &gpu, &token).unwrap();
    let a = runtime
        .upload(&Buffer::F32(vec![f32::from_bits(1), f32::MIN_POSITIVE]))
        .unwrap();
    let b = runtime.upload(&Buffer::F32(vec![2.0, 0.5])).unwrap();
    let expr = |values| {
        Expr::input(&NdArray::from_dense(
            0,
            DenseArray::new(vec![2], Buffer::F32(values), Reservation::none()).unwrap(),
        ))
    };
    let e = Expr::binary(
        crate::kernels::BinaryOp::Mul,
        &expr(vec![f32::from_bits(1), f32::MIN_POSITIVE]),
        &expr(vec![2.0, 0.5]),
    )
    .unwrap();
    let result = runtime.run(&e, &a, Some(&b)).unwrap();
    let words = runtime.read_words(&result, 2).unwrap();
    eprintln!("Float32 preservation probe: actual {words:08x?}, required [00000002, 00400000]");
    if words == [2, 0x00400000] {
        runtime.require_float32_preservation().unwrap();
    } else {
        assert_eq!(
            runtime.require_float32_preservation().unwrap_err().code,
            ErrorCode::GpuUnavailable
        );
    }
}
