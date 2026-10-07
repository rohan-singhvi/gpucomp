//! Kernel-only timing with GPU timestamp queries.

use std::time::Duration;

use crate::{Context, GpuError};

/// Times one compute pass. Only available when the device has
/// `TIMESTAMP_QUERY` and its timestamps actually work (checked with a probe
/// pass); callers fall back to wall-clock timing otherwise.
pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
}

impl GpuTimer {
    pub fn new(ctx: &Context) -> Option<Self> {
        if !ctx
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return None;
        }
        let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("timer"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timer resolve"),
            size: 2 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let timer = Self { query_set, resolve };
        if !timer.probe(ctx) {
            log::warn!(
                "adapter advertises TIMESTAMP_QUERY but timestamps read back unusable; \
                 falling back to wall-clock timing"
            );
            return None;
        }
        Some(timer)
    }

    /// Times an empty compute pass and checks the samples are real.
    fn probe(&self, ctx: &Context) -> bool {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        drop(encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("timer probe"),
            timestamp_writes: Some(self.pass_writes()),
        }));
        self.resolve(&mut encoder);
        ctx.queue.submit([encoder.finish()]);
        self.read_ticks(ctx).is_ok_and(timestamps_usable)
    }

    fn read_ticks(&self, ctx: &Context) -> Result<[u64; 2], GpuError> {
        let bytes = ctx.read_buffer(&self.resolve, 2 * 8)?;
        Ok(bytemuck::pod_read_unaligned(&bytes))
    }

    /// Pass this as `ComputePassDescriptor::timestamp_writes`.
    pub fn pass_writes(&self) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }
    }

    /// Records the resolve of both timestamps; call after the timed pass ends.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..2, &self.resolve, 0);
    }

    /// Time between the start and end of the most recently timed pass.
    pub fn read(&self, ctx: &Context) -> Result<Duration, GpuError> {
        let ticks = self.read_ticks(ctx)?;
        let period_ns = f64::from(ctx.queue.get_timestamp_period());
        let ns = ticks[1].saturating_sub(ticks[0]) as f64 * period_ns;
        Ok(Duration::from_nanos(ns as u64))
    }
}

/// Whether a resolved `[begin, end]` timestamp pair looks real. Some drivers
/// advertise `TIMESTAMP_QUERY` but never write samples (seen on Apple M4 Pro /
/// Metal with wgpu 30), which resolves to zeros.
pub fn timestamps_usable(ticks: [u64; 2]) -> bool {
    ticks[0] != 0 && ticks[1] >= ticks[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_zero_timestamps_are_unusable() {
        assert!(!timestamps_usable([0, 0]));
    }

    #[test]
    fn end_before_begin_is_unusable() {
        assert!(!timestamps_usable([500, 400]));
    }

    #[test]
    fn increasing_nonzero_timestamps_are_usable() {
        assert!(timestamps_usable([1_000, 1_500]));
    }
}
