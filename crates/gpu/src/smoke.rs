//! M0 smoke test kernel: XOR every `u32` with a constant.

use wgpu::util::DeviceExt as _;

use crate::{Context, GpuError};

const SHADER: &str = include_str!("../shaders/xor.wgsl");
const WG_SIZE: u32 = 64;

/// CPU reference for [`Context::xor_u32`].
pub fn xor_u32_cpu(data: &[u32], key: u32) -> Vec<u32> {
    data.iter().map(|w| w ^ key).collect()
}

impl Context {
    /// XORs every word with `key` on the GPU.
    ///
    /// Smoke test only: one 1D dispatch, so `data.len()` is limited to
    /// `WG_SIZE * max_compute_workgroups_per_dimension` words.
    pub fn xor_u32(&self, data: &[u32], key: u32) -> Result<Vec<u32>, GpuError> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let device = &self.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("xor"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("xor data"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("xor params"),
            contents: bytemuck::cast_slice(&[key, 0, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xor"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((data.len() as u32).div_ceil(WG_SIZE), 1, 1);
        }
        self.queue.submit([encoder.finish()]);

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
