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
fn native_precision_guard_verifies_integer_preservation() {
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
    assert_eq!(words, [2, 0x00400000]);
    runtime.require_float32_preservation().unwrap();
}

// This diagnostic deliberately uses native pow, outside the production
// integer kernels. A passing diagnostic is not power acceptance: it records
// device behavior and verifies that the unimplemented operation stays rejected.
#[test]
#[ignore = "requires an actual supported native GPU"]
fn native_power_precision_diagnostic() {
    let cpu = MemoryBudget::new(Device::Cpu, 1024);
    let gpu = MemoryBudget::new(Device::Gpu, 1024);
    let token = CancellationToken::new();
    let runtime = Runtime::new(&cpu, &gpu, &token).unwrap();
    let a = runtime
        .upload(&Buffer::F32(vec![
            f32::from_bits(1),
            1.0,
            2.0f32.powi(-70),
            2.0,
        ]))
        .unwrap();
    let out = runtime
        .allocate(
            8,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            false,
        )
        .unwrap();
    let module = runtime
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("native power diagnostic"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
            @group(0) @binding(0) var<storage, read> a: array<f32>;
            @group(0) @binding(1) var<storage, read_write> out: array<u32>;
            @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                out[id.x] = bitcast<u32>(pow(a[id.x * 2u], a[id.x * 2u + 1u]));
            }
        "#
                .into(),
            ),
        });
    let pipeline = runtime
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("native power diagnostic"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let bind = runtime
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.buffer.as_entire_binding(),
                },
            ],
        });
    let mut encoder = runtime.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(2, 1, 1);
    }
    runtime.queue.submit([encoder.finish()]);
    runtime.drain().unwrap();
    let words = runtime.read_words(&out, 2).unwrap();
    eprintln!(
        "Native pow diagnostic: actual {words:08x?}, NumPy 2.5.4 required [00000001, 00000200]"
    );
    let input = Expr::input(&NdArray::from_dense(
        0,
        DenseArray::new(vec![1], Buffer::F32(vec![2.0]), Reservation::none()).unwrap(),
    ));
    let power = Expr::binary(kernels::BinaryOp::Pow, &input, &input).unwrap();
    assert_eq!(
        validate(&power).unwrap_err().code,
        ErrorCode::GpuUnsupportedOperation
    );
    assert_eq!(
        shader::generate(&power).unwrap_err().code,
        ErrorCode::GpuUnsupportedOperation
    );
}
