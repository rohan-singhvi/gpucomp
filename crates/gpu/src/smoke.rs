//! M0 smoke test kernel: XOR every `u32` with a constant.
//! Also used by the benchmark harness as a trivial bandwidth-bound kernel.

use wgpu::util::DeviceExt as _;

use crate::{dispatch_grid, Context, GpuError, GpuTimer};

const SHADER: &str = include_str!("../shaders/xor.wgsl");
const WG_SIZE: u32 = 64;

/// CPU reference for [`Context::xor_u32`].
pub fn xor_u32_cpu(data: &[u32], key: u32) -> Vec<u32> {
    data.iter().map(|w| w ^ key).collect()
}

/// The compiled XOR pipeline.
pub struct XorKernel {
    pipeline: wgpu::ComputePipeline,
}

impl XorKernel {
    pub fn new(ctx: &Context) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("xor"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("xor"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("WG_SIZE", f64::from(WG_SIZE))],
                    ..Default::default()
                },
                cache: None,
            });
        Self { pipeline }
    }

    /// XORs every word of `data` (a `STORAGE` buffer) with `key` in place and
    /// submits.
    pub fn dispatch(&self, ctx: &Context, data: &wgpu::Buffer, key: u32, timer: Option<&GpuTimer>) {
        let words = (data.size() / 4) as u32;
        let params = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("xor params"),
                contents: bytemuck::cast_slice(&[key, 0, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xor"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("xor"),
                timestamp_writes: timer.map(GpuTimer::pass_writes),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let (x, y) = dispatch_grid(
                words.div_ceil(WG_SIZE),
                ctx.device_limits().max_compute_workgroups_per_dimension,
            );
            pass.dispatch_workgroups(x, y, 1);
        }
        if let Some(timer) = timer {
            timer.resolve(&mut encoder);
        }
        ctx.queue.submit([encoder.finish()]);
    }
}

impl Context {
    /// XORs every word with `key` on the GPU (upload, dispatch, readback).
    pub fn xor_u32(&self, data: &[u32], key: u32) -> Result<Vec<u32>, GpuError> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let buffer = self.upload(
            bytemuck::cast_slice(data),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        XorKernel::new(self).dispatch(self, &buffer, key, None);
        let bytes = self.read_buffer(&buffer, std::mem::size_of_val(data) as u64)?;
        Ok(bytemuck::pod_collect_to_vec(&bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_xor_flips_the_key_bits() {
        assert_eq!(xor_u32_cpu(&[0, 1, 0xFFFF_FFFF], 3), [3, 2, 0xFFFF_FFFC]);
    }
}
