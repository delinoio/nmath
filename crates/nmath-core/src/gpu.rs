//! Native GPU execution. Construction and CPU execution never discover
//! adapters.
//!
//! Call `validate` before queue admission, then `execute` on the worker after
//! freezing inputs. A job owns one device; failures never retry or fall back.

use std::sync::{Arc, Mutex, mpsc};

use crate::{
    array::NdArray,
    buffer::{Buffer, DenseArray},
    dtype::DType,
    error::{Error, ErrorCode, Result},
    expr::{Expr, ExprKind},
    kernels::{self, ReduceOp},
    memory::{Device, MemoryBudget, Reservation},
    sync::CancellationToken,
};

#[path = "gpu/shader.rs"]
mod shader;

fn unsupported(message: &str) -> Error {
    Error::new(ErrorCode::GpuUnsupportedOperation, message)
}

fn supported_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::Float32 | DType::Int32 | DType::UInt32)
}

/// The supported native backend, without loading a driver or discovering
/// devices. This is a platform restriction, not an official tested-hardware
/// allowlist.
pub fn platform_backend() -> Result<wgpu::Backends> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Ok(wgpu::Backends::METAL)
    } else if cfg!(all(
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    )) {
        Ok(wgpu::Backends::VULKAN)
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Ok(wgpu::Backends::DX12)
    } else {
        Err(Error::new(
            ErrorCode::GpuUnavailable,
            "This CPU platform has no supported GPU backend.",
        ))
    }
}

fn elements(shape: &[usize]) -> Result<u32> {
    shape.iter().try_fold(1u32, |n, &dim| {
        n.checked_mul(
            u32::try_from(dim).map_err(|_| unsupported("A GPU dimension exceeds u32 indexing."))?,
        )
        .ok_or_else(|| unsupported("The GPU array exceeds u32 indexing."))
    })
}

/// Check the complete graph before admission. A weak scalar adopts only its
/// binary parent's already-inferred compute dtype. No result dtype is narrowed.
pub fn validate(expr: &Expr) -> Result<()> {
    validate_node(expr, None)
}

fn validate_node(expr: &Expr, weak: Option<DType>) -> Result<()> {
    let dtype = weak.unwrap_or(expr.dtype());
    if !supported_dtype(dtype) {
        return Err(unsupported(
            "The inferred dtype is outside float32/int32/uint32. Convert explicitly before GPU \
             execution.",
        ));
    }
    elements(expr.shape())?;
    match expr.kind() {
        ExprKind::Input(_) => Ok(()),
        ExprKind::Scalar(_) => {
            // Expr construction already checks NEP 50 scalar bounds.
            if weak.is_none() {
                return Err(unsupported(
                    "A standalone scalar has no supported GPU dtype.",
                ));
            }
            Ok(())
        }
        ExprKind::Unary(_, a) => validate_node(a, None),
        ExprKind::Binary {
            lhs,
            rhs,
            compute_dtype,
            ..
        } => {
            if !supported_dtype(*compute_dtype) {
                return Err(unsupported(
                    "Binary promotion requires an unsupported GPU dtype.",
                ));
            }
            for a in [lhs, rhs] {
                validate_node(
                    a,
                    matches!(a.kind(), ExprKind::Scalar(_)).then_some(*compute_dtype),
                )?;
            }
            Ok(())
        }
        ExprKind::Reduce(op, a, axes) => {
            validate_node(a, None)?;
            let axes = kernels::reduce_axes(a.shape().len(), axes.as_deref())?;
            if matches!(op, ReduceOp::Min | ReduceOp::Max)
                && axes
                    .iter()
                    .enumerate()
                    .any(|(axis, &reduce)| reduce && a.shape()[axis] == 0)
            {
                return Err(Error::new(
                    ErrorCode::ShapeMismatch,
                    "A minimum or maximum reduction has an empty axis.",
                ));
            }
            Ok(())
        }
        ExprKind::MatMul(a, b) => {
            validate_node(a, None)?;
            validate_node(b, None)
        }
        _ => Err(unsupported(
            "The graph contains an operation outside the GPU subset.",
        )),
    }
}

struct Allocation {
    buffer: wgpu::Buffer,
    _charge: Reservation,
}

/// Each allocation remains charged until its last submitted use has completed.
struct Runtime<'a> {
    device: wgpu::Device,
    queue: wgpu::Queue,
    failure: Arc<Mutex<Option<Error>>>,
    cpu: &'a MemoryBudget,
    gpu: &'a MemoryBudget,
    token: &'a CancellationToken,
}

impl<'a> Runtime<'a> {
    fn new(
        cpu: &'a MemoryBudget,
        gpu: &'a MemoryBudget,
        token: &'a CancellationToken,
    ) -> Result<Self> {
        token.check()?;
        if cpu.device() != Device::Cpu || gpu.device() != Device::Gpu {
            return Err(Error::new(
                ErrorCode::ContextInvalidOptions,
                "GPU execution requires separate CPU and GPU budgets.",
            ));
        }
        let backends = platform_backend()?;
        tracing::debug!(event = "nmath.gpu.init", backend = ?backends);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|_| {
            Error::new(
                ErrorCode::GpuUnavailable,
                "No adapter is available for the required native backend.",
            )
        })?;
        if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
            return Err(Error::new(
                ErrorCode::GpuUnavailable,
                "A software adapter is not a GPU execution target.",
            ));
        }
        token.check()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("nmath"),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .map_err(|_| {
            Error::new(
                ErrorCode::GpuUnavailable,
                "The native GPU device could not be created.",
            )
        })?;
        let failure = Arc::new(Mutex::new(None));
        let errors = failure.clone();
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            let code = match error {
                wgpu::Error::OutOfMemory { .. } => ErrorCode::GpuOutOfMemory,
                wgpu::Error::Validation { .. } => ErrorCode::GpuUnsupportedOperation,
                wgpu::Error::Internal { .. } => ErrorCode::GpuDeviceLost,
            };
            let mut slot = errors.lock().unwrap();
            if slot.is_none() {
                *slot = Some(Error::new(
                    code,
                    "The native GPU reported an execution failure.",
                ));
            }
        }));
        let errors = failure.clone();
        device.set_device_lost_callback(move |_, _| {
            let mut slot = errors.lock().unwrap();
            if slot.is_none() {
                *slot = Some(Error::new(
                    ErrorCode::GpuDeviceLost,
                    "The GPU device was lost.",
                ));
            }
        });
        Ok(Self {
            device,
            queue,
            failure,
            cpu,
            gpu,
            token,
        })
    }

    /// Verify the integer preservation kernel on the selected device. This
    /// rejection guard is not full numerical acceptance or a hardware
    /// allowlist.
    fn require_float32_preservation(&self) -> Result<()> {
        let left_charge = self.cpu.reserve(8)?;
        let left = DenseArray::new(
            vec![2],
            Buffer::F32(vec![f32::from_bits(1), f32::MIN_POSITIVE]),
            left_charge,
        )?;
        let right_charge = self.cpu.reserve(8)?;
        let right = DenseArray::new(vec![2], Buffer::F32(vec![2.0, 0.5]), right_charge)?;
        let a = self.upload(&left.buffer)?;
        let b = self.upload(&right.buffer)?;
        let expression = Expr::binary(
            crate::kernels::BinaryOp::Mul,
            &Expr::input(&NdArray::from_dense(0, left)),
            &Expr::input(&NdArray::from_dense(0, right)),
        )?;
        let result = self.run(&expression, &a, Some(&b))?;
        let _host = self.cpu.reserve(8)?;
        if self.read_words(&result, 2)? != [2, 0x00400000] {
            return Err(Error::new(
                ErrorCode::GpuUnavailable,
                "The GPU does not preserve required float32 subnormal arithmetic. No result or \
                 CPU fallback is produced.",
            ));
        }
        Ok(())
    }

    fn check(&self) -> Result<()> {
        if let Some(error) = self.failure.lock().unwrap().clone() {
            return Err(error);
        }
        self.token.check()
    }

    fn allocate(&self, bytes: u64, usage: wgpu::BufferUsages, mapped: bool) -> Result<Allocation> {
        self.check()?;
        let bytes = bytes.max(4);
        let limits = self.device.limits();
        if bytes > limits.max_buffer_size
            || (usage.contains(wgpu::BufferUsages::STORAGE)
                && bytes > u64::from(limits.max_storage_buffer_binding_size))
        {
            return Err(unsupported(
                "A buffer exceeds the selected GPU's binding limits.",
            ));
        }
        let charge = self.gpu.reserve(bytes)?;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nmath buffer"),
            size: bytes,
            usage,
            mapped_at_creation: mapped,
        });
        self.check()?;
        Ok(Allocation {
            buffer,
            _charge: charge,
        })
    }

    fn upload(&self, source: &Buffer) -> Result<Allocation> {
        let allocation = self.allocate(
            source.byte_len(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            false,
        )?;
        // Explicit staging also accounts for transfers on discrete-memory GPUs.
        // Mapping a STORAGE buffer at creation can hide staging inside wgpu.
        let staging = self.allocate(
            source.byte_len(),
            wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            true,
        )?;
        {
            let mut bytes = staging
                .buffer
                .slice(..)
                .get_mapped_range_mut()
                .map_err(|_| Error::new(ErrorCode::GpuDeviceLost, "GPU upload mapping failed."))?;
            // Write directly into mapped storage: no uncharged serialization
            // copy.
            for i in 0..source.len() {
                if i % 16384 == 0 {
                    self.check()?;
                }
                let word = match source {
                    Buffer::F32(v) => v[i].to_bits(),
                    Buffer::I32(v) => v[i] as u32,
                    Buffer::U32(v) => v[i],
                    _ => return Err(unsupported("Only 32-bit numeric buffers can be uploaded.")),
                };
                bytes
                    .slice(i * 4..i * 4 + 4)
                    .copy_from_slice(&word.to_le_bytes());
            }
        }
        staging.buffer.unmap();
        self.check()?;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            &staging.buffer,
            0,
            &allocation.buffer,
            0,
            source.byte_len().max(4),
        );
        self.queue.submit([encoder.finish()]);
        self.drain()?;
        Ok(allocation)
    }

    // Wait even after cancellation so reservations are not released while the
    // driver still uses their buffers. No new work is submitted after
    // cancellation.
    fn drain(&self) -> Result<()> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|_| {
                Error::new(
                    ErrorCode::GpuDeviceLost,
                    "Waiting for GPU completion failed.",
                )
            })?;
        self.check()
    }

    fn run(&self, expr: &Expr, a: &Allocation, b: Option<&Allocation>) -> Result<Allocation> {
        self.check()?;
        let count = elements(expr.shape())?;
        let output = self.allocate(
            u64::from(count) * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            false,
        )?;
        if count == 0 {
            return Ok(output);
        }
        let status = self.allocate(
            4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            false,
        )?;
        let source = shader::generate(expr)?;
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("nmath kernel"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        self.check()?;
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("nmath kernel"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        self.check()?;
        let inner = match expr.kind() {
            ExprKind::Reduce(_, input, axes) => {
                let axes = kernels::reduce_axes(input.shape().len(), axes.as_deref())?;
                axes.iter()
                    .zip(input.shape())
                    .filter(|(reduce, _)| **reduce)
                    .map(|(_, &d)| d as u32)
                    .product()
            }
            ExprKind::MatMul(a, _) => *a.shape().last().unwrap() as u32,
            _ => 1,
        };
        // Bound each submission, so cancellation is observed between output
        // tiles and reduction/matmul chunks, without releasing in-flight
        // memory.
        for start in (0..count).step_by(4096) {
            for offset in (0..inner.max(1)).step_by(256) {
                self.check()?;
                let end = count.min(start.saturating_add(4096));
                let stop = inner.min(offset.saturating_add(256));
                let host_params = self.cpu.reserve(16)?;
                let params = self.upload(&Buffer::U32(vec![start, end, offset, stop]))?;
                drop(host_params);
                let mut entries = vec![
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: status.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params.buffer.as_entire_binding(),
                    },
                ];
                if let Some(b) = b {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b.buffer.as_entire_binding(),
                    });
                }
                let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("nmath operands"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
                });
                self.check()?;
                let mut encoder = self.device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups((end - start).div_ceil(64), 1, 1);
                }
                self.check()?;
                self.queue.submit([encoder.finish()]);
                self.drain()?;
            }
        }
        let _status_host = self.cpu.reserve(4)?;
        let result = self.read_words(&status, 1)?;
        if result[0] != 0 {
            return Err(Error::new(
                ErrorCode::DTypeUnsupported,
                "Integers to negative integer powers are not allowed.",
            ));
        }
        Ok(output)
    }

    fn read_words(&self, source: &Allocation, count: u32) -> Result<Vec<u32>> {
        self.check()?;
        let bytes = u64::from(count) * 4;
        let staging = self.allocate(
            bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            false,
        )?;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&source.buffer, 0, &staging.buffer, 0, bytes.max(4));
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = mpsc::sync_channel(1);
        staging
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.drain()?;
        rx.recv()
            .map_err(|_| {
                Error::new(
                    ErrorCode::GpuDeviceLost,
                    "GPU readback completion was lost.",
                )
            })?
            .map_err(|_| Error::new(ErrorCode::GpuDeviceLost, "GPU readback mapping failed."))?;
        let data = staging
            .buffer
            .slice(..)
            .get_mapped_range()
            .map_err(|_| Error::new(ErrorCode::GpuDeviceLost, "GPU readback range failed."))?;
        let result = data
            .chunks_exact(4)
            .take(count as usize)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        drop(data);
        staging.buffer.unmap();
        self.check()?;
        Ok(result)
    }

    fn evaluate<'s>(
        &self,
        expr: &Expr,
        weak: Option<DType>,
        snapshot: &impl Fn(&NdArray) -> Result<&'s DenseArray>,
    ) -> Result<Allocation> {
        self.check()?;
        match expr.kind() {
            ExprKind::Input(a) => self.upload(&snapshot(a)?.buffer),
            ExprKind::Scalar(value) => {
                let _host = self.cpu.reserve(4)?;
                self.upload(&Buffer::filled(weak.unwrap_or(expr.dtype()), 1, *value)?)
            }
            ExprKind::Unary(_, a) | ExprKind::Reduce(_, a, _) => {
                let a = self.evaluate(a, None, snapshot)?;
                self.run(expr, &a, None)
            }
            ExprKind::Binary {
                lhs,
                rhs,
                compute_dtype,
                ..
            } => {
                let a = self.evaluate(
                    lhs,
                    matches!(lhs.kind(), ExprKind::Scalar(_)).then_some(*compute_dtype),
                    snapshot,
                )?;
                let b = self.evaluate(
                    rhs,
                    matches!(rhs.kind(), ExprKind::Scalar(_)).then_some(*compute_dtype),
                    snapshot,
                )?;
                self.run(expr, &a, Some(&b))
            }
            ExprKind::MatMul(lhs, rhs) => {
                let a = self.evaluate(lhs, None, snapshot)?;
                let b = self.evaluate(rhs, None, snapshot)?;
                self.run(expr, &a, Some(&b))
            }
            _ => Err(unsupported("Unsupported GPU graph node.")),
        }
    }
}

/// Evaluate on the requested native GPU using actual-start snapshots supplied
/// by `JobEnv`. The returned host array is CPU charged; device buffers and
/// transfers are GPU charged. The caller must retain snapshots until this
/// function returns.
pub fn execute<'s>(
    expr: &Expr,
    cpu: &MemoryBudget,
    gpu: &MemoryBudget,
    token: &CancellationToken,
    snapshot: impl Fn(&NdArray) -> Result<&'s DenseArray>,
) -> Result<DenseArray> {
    validate(expr)?;
    let runtime = Runtime::new(cpu, gpu, token)?;
    if expr.dtype() == DType::Float32 {
        runtime.require_float32_preservation()?;
    }
    let output = runtime.evaluate(expr, None, &snapshot)?;
    let count = elements(expr.shape())?;
    let charge = cpu.reserve(u64::from(count) * 4)?;
    // Conversion allocates another typed Vec; charge it until the words are
    // gone.
    let temporary = cpu.reserve(u64::from(count) * 4)?;
    let words = runtime.read_words(&output, count)?;
    let buffer = match expr.dtype() {
        DType::Float32 => Buffer::F32(words.iter().map(|&v| f32::from_bits(v)).collect()),
        DType::Int32 => Buffer::I32(words.iter().map(|&v| v as i32).collect()),
        DType::UInt32 => Buffer::U32(words.clone()),
        _ => return Err(unsupported("Unsupported GPU result dtype.")),
    };
    drop(words);
    drop(temporary);
    runtime.check()?;
    DenseArray::new(expr.shape().to_vec(), buffer, charge)
}

#[cfg(test)]
#[path = "gpu/tests.rs"]
mod tests;
