//! wgpu adapter/device setup shared by every GPU pipeline.

use std::fmt::Write as _;
use std::sync::mpsc;

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no suitable GPU adapter: {0}")]
    NoAdapter(#[from] wgpu::RequestAdapterError),
    #[error("failed to create GPU device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    #[error("GPU device poll failed: {0}")]
    Poll(#[from] wgpu::PollError),
    #[error("GPU buffer readback failed: {0}")]
    Readback(String),
}

#[derive(Clone, Debug, Default)]
pub struct ContextOptions {
    /// Restrict adapter selection to these backends. `None` = Metal, D3D12 and Vulkan.
    pub backends: Option<wgpu::Backends>,
}

/// An adapter, the device created from it, and its queue.
pub struct Context {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    adapter: AdapterSummary,
    adapter_limits: wgpu::Limits,
    device_limits: wgpu::Limits,
}

impl Context {
    /// Picks a high-performance adapter and creates a device that requests the
    /// adapter's actual buffer and workgroup-storage limits (see [`device_limits`]).
    pub fn new(options: &ContextOptions) -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async(options))
    }

    async fn new_async(options: &ContextOptions) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: options.backends.unwrap_or(wgpu::Backends::PRIMARY),
            ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await?;
        let info = adapter.get_info();
        let summary = AdapterSummary {
            name: info.name,
            backend: info.backend,
            device_type: info.device_type,
            driver: format!("{} {}", info.driver, info.driver_info)
                .trim()
                .to_string(),
        };
        let adapter_limits = adapter.limits();
        let required_limits = device_limits(&adapter_limits);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gpucomp"),
                required_limits: required_limits.clone(),
                ..Default::default()
            })
            .await?;
        log::info!(
            "using adapter {:?} ({}, {:?})",
            summary.name,
            summary.backend,
            summary.device_type
        );
        log::debug!("device limits:\n{}", report(&summary, &required_limits));
        Ok(Self {
            device,
            queue,
            adapter: summary,
            adapter_limits,
            device_limits: required_limits,
        })
    }

    pub fn adapter(&self) -> &AdapterSummary {
        &self.adapter
    }

    /// What the adapter supports.
    pub fn adapter_limits(&self) -> &wgpu::Limits {
        &self.adapter_limits
    }

    /// What the device was created with; pipelines must stay within these.
    pub fn device_limits(&self) -> &wgpu::Limits {
        &self.device_limits
    }

    /// Report of the adapter and the limits the device was created with.
    pub fn report(&self) -> String {
        report(&self.adapter, &self.device_limits)
    }

    /// Copies `size` bytes from the start of `src` (which needs `COPY_SRC`) back to the host.
    pub(crate) fn read_buffer(&self, src: &wgpu::Buffer, size: u64) -> Result<Vec<u8>, GpuError> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);

        let (tx, rx) = mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| drop(tx.send(r)));
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()
            .map_err(|e| GpuError::Readback(e.to_string()))?
            .map_err(|e| GpuError::Readback(e.to_string()))?;
        let bytes = staging
            .slice(..)
            .get_mapped_range()
            .map_err(|e| GpuError::Readback(e.to_string()))?
            .to_vec();
        staging.unmap();
        Ok(bytes)
    }
}

/// The subset of adapter information we log and print.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterSummary {
    pub name: String,
    pub backend: wgpu::Backend,
    pub device_type: wgpu::DeviceType,
    pub driver: String,
}

/// Limits to request when creating the device.
///
/// Starts from the WebGPU defaults (or the downlevel defaults if the adapter
/// can't meet them), then raises the limits that bound how much data a single
/// dispatch can touch to the adapter's actual maxima.
pub fn device_limits(adapter: &wgpu::Limits) -> wgpu::Limits {
    let defaults = wgpu::Limits::defaults();
    let base = if defaults.check_limits(adapter) {
        defaults
    } else {
        wgpu::Limits::downlevel_defaults()
    };
    wgpu::Limits {
        max_storage_buffer_binding_size: adapter.max_storage_buffer_binding_size,
        max_buffer_size: adapter.max_buffer_size,
        max_compute_workgroup_storage_size: adapter.max_compute_workgroup_storage_size,
        ..base
    }
}

/// Human-readable adapter, backend and limits report for `gpucomp info`.
pub fn report(adapter: &AdapterSummary, limits: &wgpu::Limits) -> String {
    let mut s = String::new();
    // Writing to a String can't fail.
    let _ = writeln!(s, "adapter:     {}", adapter.name);
    let _ = writeln!(s, "backend:     {}", adapter.backend);
    let _ = writeln!(s, "device type: {:?}", adapter.device_type);
    let _ = writeln!(s, "driver:      {}", adapter.driver);
    let _ = writeln!(s, "limits:");
    let rows: [(&str, u64); 6] = [
        (
            "max_storage_buffer_binding_size",
            limits.max_storage_buffer_binding_size,
        ),
        ("max_buffer_size", limits.max_buffer_size),
        (
            "max_compute_workgroup_storage_size",
            limits.max_compute_workgroup_storage_size.into(),
        ),
        (
            "max_compute_invocations_per_workgroup",
            limits.max_compute_invocations_per_workgroup.into(),
        ),
        (
            "max_compute_workgroups_per_dimension",
            limits.max_compute_workgroups_per_dimension.into(),
        ),
        (
            "max_storage_buffers_per_shader_stage",
            limits.max_storage_buffers_per_shader_stage.into(),
        ),
    ];
    for (name, value) in rows {
        let _ = writeln!(s, "  {name:<38} {value}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big_adapter() -> wgpu::Limits {
        wgpu::Limits {
            max_storage_buffer_binding_size: 4 << 30,
            max_buffer_size: 8 << 30,
            max_compute_workgroup_storage_size: 32 << 10,
            ..wgpu::Limits::defaults()
        }
    }

    #[test]
    fn device_limits_request_adapters_storage_binding_size() {
        assert_eq!(
            device_limits(&big_adapter()).max_storage_buffer_binding_size,
            4 << 30
        );
    }

    #[test]
    fn device_limits_request_adapters_buffer_size() {
        assert_eq!(device_limits(&big_adapter()).max_buffer_size, 8 << 30);
    }

    #[test]
    fn device_limits_request_adapters_workgroup_storage_size() {
        assert_eq!(
            device_limits(&big_adapter()).max_compute_workgroup_storage_size,
            32 << 10
        );
    }

    #[test]
    fn device_limits_never_exceed_a_downlevel_adapter() {
        let adapter = wgpu::Limits::downlevel_defaults();
        assert!(device_limits(&adapter).check_limits(&adapter));
    }

    #[test]
    fn report_names_adapter_backend_and_key_limits() {
        let summary = AdapterSummary {
            name: "Test GPU".into(),
            backend: wgpu::Backend::Metal,
            device_type: wgpu::DeviceType::IntegratedGpu,
            driver: "drv 1.0".into(),
        };
        let text = report(&summary, &big_adapter());
        for needle in [
            "Test GPU",
            "metal",
            "IntegratedGpu",
            "drv 1.0",
            "max_storage_buffer_binding_size",
            &(4u64 << 30).to_string(),
            "max_buffer_size",
            &(8u64 << 30).to_string(),
            "max_compute_workgroup_storage_size",
            "32768",
            "max_compute_workgroups_per_dimension",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
        }
    }
}
