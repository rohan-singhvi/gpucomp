//! M7: byte-shuffle and delta filters on the GPU, forward and inverse
//! (plan §4a; `cpu::filter` is the reference).
//!
//! A [`FilterJob`] transforms `len` bytes at `src_offset` of a source buffer
//! into `len` bytes at `dst_offset` of a *different* destination buffer.
//! Typical uses: undoing filters on decoded chunks (the decoder), and making
//! filtered copies of input chunks for candidate compression (an encoder):
//!
//! ```text
//! let kernels = FilterKernels::new(ctx);                       // once; reuse
//! let pass = kernels.prepare(ctx, Direction::Forward, &input, &scratch, &jobs)?;
//! kernels.record(ctx, &mut encoder, &pass, None);              // any number of times
//! ctx.queue.submit([encoder.finish()]);
//! ```
//!
//! Rules: offsets are 4-aligned, ranges lie inside their buffers, destination
//! ranges don't overlap, and jobs with [`Filter::None`] are skipped. Bytes of
//! the destination outside every job are left untouched, including the bytes
//! past the end of a job in its final word.

use bytemuck::{Pod, Zeroable};
use format::Filter;
use wgpu::BufferUsages;

use crate::{dispatch_grid, Context, GpuError};

const SHADER: &str = include_str!("../shaders/filter.wgsl");
/// Invocations per workgroup (one workgroup per job).
const WG_SIZE: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Apply the filter (before compression).
    Forward,
    /// Undo it (after decompression).
    Inverse,
}

/// One range to transform. See the module docs for the rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilterJob {
    pub filter: Filter,
    /// Start in the source buffer, in bytes (4-aligned).
    pub src_offset: u32,
    /// Start in the destination buffer, in bytes (4-aligned).
    pub dst_offset: u32,
    /// Bytes to transform (any length; the `len % width` tail is copied).
    pub len: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum FilterError {
    #[error("filter job {job}: offsets must be 4-byte aligned")]
    Misaligned { job: usize },
    #[error("filter job {job}: range lies outside its buffer")]
    OutOfBounds { job: usize },
    #[error("filter job {job}: destination overlaps another job's")]
    Overlap { job: usize },
    #[error("filter job {job}: width must be 1, 2, 4 or 8")]
    BadWidth { job: usize },
    #[error(transparent)]
    Gpu(#[from] GpuError),
}

/// A job as the shader sees it (`Job` in `filter.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct JobDesc {
    src_offset: u32,
    dst_offset: u32,
    len: u32,
    width: u32,
}

/// The compiled filter pipelines; create once and reuse.
pub struct FilterKernels {
    layout: wgpu::BindGroupLayout,
    shuffle_forward: wgpu::ComputePipeline,
    shuffle_inverse: wgpu::ComputePipeline,
    delta_forward: wgpu::ComputePipeline,
    delta_inverse: wgpu::ComputePipeline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Shuffle,
    Delta,
}

/// Jobs uploaded and bound to their buffers, ready to [`record`](FilterKernels::record).
pub struct FilterPass {
    direction: Direction,
    /// One dispatch per filter kind that has jobs: (kind, bind group, job count).
    dispatches: Vec<(Kind, wgpu::BindGroup, u32)>,
}

impl FilterPass {
    /// True when no job needs any work (all `None`, or no jobs).
    pub fn is_empty(&self) -> bool {
        self.dispatches.is_empty()
    }
}

impl FilterKernels {
    pub fn new(ctx: &Context) -> Self {
        let device = &ctx.device;
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("filter"),
            entries: &[storage(0, true), storage(1, true), storage(2, false)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("filter"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("filter"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("WG_SIZE", f64::from(WG_SIZE))],
                    ..Default::default()
                },
                cache: None,
            })
        };
        FilterKernels {
            shuffle_forward: pipeline("shuffle_forward"),
            shuffle_inverse: pipeline("shuffle_inverse"),
            delta_forward: pipeline("delta_forward"),
            delta_inverse: pipeline("delta_inverse"),
            layout,
        }
    }

    /// Validates `jobs` against the buffers and uploads them. `src` and `dst`
    /// need `STORAGE` usage and must be different buffers.
    pub fn prepare(
        &self,
        ctx: &Context,
        direction: Direction,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        jobs: &[FilterJob],
    ) -> Result<FilterPass, FilterError> {
        validate(jobs, src.size(), dst.size())?;
        let mut dispatches = Vec::new();
        for kind in [Kind::Shuffle, Kind::Delta] {
            let descs: Vec<JobDesc> = jobs
                .iter()
                .filter_map(|j| {
                    let width = match (kind, j.filter) {
                        (Kind::Shuffle, Filter::Shuffle { width })
                        | (Kind::Delta, Filter::Delta { width }) => width,
                        _ => return None,
                    };
                    Some(JobDesc {
                        src_offset: j.src_offset,
                        dst_offset: j.dst_offset,
                        len: j.len,
                        width: u32::from(width),
                    })
                })
                .collect();
            if descs.is_empty() {
                continue;
            }
            let job_buf = ctx.upload(bytemuck::cast_slice(&descs), BufferUsages::STORAGE);
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("filter"),
                layout: &self.layout,
                entries: &[(0, src), (1, &job_buf), (2, dst)].map(|(binding, buffer)| {
                    wgpu::BindGroupEntry {
                        binding,
                        resource: buffer.as_entire_binding(),
                    }
                }),
            });
            dispatches.push((kind, bind_group, descs.len() as u32));
        }
        Ok(FilterPass {
            direction,
            dispatches,
        })
    }

    /// Records the prepared jobs as one compute pass into `encoder` (nothing
    /// if the pass is empty), with optional timestamp writes for that pass
    /// (e.g. [`GpuTimer::pass_writes`](crate::GpuTimer::pass_writes)).
    pub fn record(
        &self,
        ctx: &Context,
        encoder: &mut wgpu::CommandEncoder,
        pass: &FilterPass,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        if pass.is_empty() {
            return;
        }
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("filter"),
            timestamp_writes,
        });
        for (kind, bind_group, count) in &pass.dispatches {
            cpass.set_pipeline(match (kind, pass.direction) {
                (Kind::Shuffle, Direction::Forward) => &self.shuffle_forward,
                (Kind::Shuffle, Direction::Inverse) => &self.shuffle_inverse,
                (Kind::Delta, Direction::Forward) => &self.delta_forward,
                (Kind::Delta, Direction::Inverse) => &self.delta_inverse,
            });
            cpass.set_bind_group(0, bind_group, &[]);
            let (x, y) = dispatch_grid(
                *count,
                ctx.device_limits().max_compute_workgroups_per_dimension,
            );
            cpass.dispatch_workgroups(x, y, 1);
        }
    }

    /// Convenience: uploads `src`, and `dst` as the destination's initial
    /// contents, runs `jobs` and reads the destination back.
    pub fn run(
        &self,
        ctx: &Context,
        direction: Direction,
        src: &[u8],
        jobs: &[FilterJob],
        dst: &[u8],
    ) -> Result<Vec<u8>, FilterError> {
        // Buffers are whole words, at least one.
        let words = |bytes: &[u8]| {
            let mut v = bytes.to_vec();
            v.resize((bytes.len().div_ceil(4) * 4).max(4), 0);
            v
        };
        let src_buf = ctx.upload(&words(src), BufferUsages::STORAGE);
        let dst_buf = ctx.upload(&words(dst), BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        validate(jobs, src.len() as u64, dst.len() as u64)?;
        let pass = self.prepare(ctx, direction, &src_buf, &dst_buf, jobs)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.record(ctx, &mut encoder, &pass, None);
        ctx.queue.submit([encoder.finish()]);
        let mut out = ctx.read_buffer(&dst_buf, dst_buf.size())?;
        out.truncate(dst.len());
        Ok(out)
    }
}

/// Checks alignment, widths, bounds (`src_len`/`dst_len` bytes) and that
/// destination ranges are disjoint.
fn validate(jobs: &[FilterJob], src_len: u64, dst_len: u64) -> Result<(), FilterError> {
    let mut ranges = Vec::with_capacity(jobs.len());
    for (job, j) in jobs.iter().enumerate() {
        match j.filter {
            Filter::None => continue,
            Filter::Shuffle { width } | Filter::Delta { width } => {
                if !matches!(width, 1 | 2 | 4 | 8) {
                    return Err(FilterError::BadWidth { job });
                }
            }
        }
        if j.src_offset % 4 != 0 || j.dst_offset % 4 != 0 {
            return Err(FilterError::Misaligned { job });
        }
        let len = u64::from(j.len);
        if u64::from(j.src_offset) + len > src_len || u64::from(j.dst_offset) + len > dst_len {
            return Err(FilterError::OutOfBounds { job });
        }
        ranges.push((u64::from(j.dst_offset), u64::from(j.dst_offset) + len, job));
    }
    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        let ((_, end, _), (start, _, job)) = (pair[0], pair[1]);
        if start < end {
            return Err(FilterError::Overlap { job });
        }
    }
    Ok(())
}
