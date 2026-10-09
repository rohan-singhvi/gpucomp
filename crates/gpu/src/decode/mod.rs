//! GPU decompression.

pub mod plan;

use std::borrow::Cow;
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Range;

use format::{Filter, Index};
use wgpu::BufferUsages;

use crate::filter::{Direction, FilterJob, FilterKernels, FilterPass};
use crate::{dispatch_grid, Context, GpuError};

const NAIVE_SHADER: &str = include_str!("../../shaders/lz4_decode_naive.wgsl");
const COOP_SHADER: &str = include_str!("../../shaders/lz4_decode_coop.wgsl");
const GLZ_SHADER: &str = include_str!("../../shaders/glz_decode.wgsl");
/// Invocations per workgroup for GLZ (one sequence each). At most 64: the
/// shader tracks pending matches in a 64-bit mask.
const GLZ_WORKGROUP: u32 = 64;

/// Per-chunk result codes written by the decode shader (0 = ok).
/// They mirror `cpu::lz4::decode::DecodeError`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkStatus {
    Truncated = 1,
    ZeroOffset = 2,
    OffsetBeforeStart = 3,
    OutputOverflow = 4,
    SizeMismatch = 5,
    /// GLZ only: invalid sequence count, extension count or final sequence.
    BadSequence = 6,
    /// The shader read an unknown code; shouldn't happen.
    Unknown = u32::MAX,
}

#[derive(Debug, thiserror::Error)]
pub enum GpuDecodeError {
    #[error(transparent)]
    Read(#[from] format::ReadError),
    #[error(transparent)]
    Format(#[from] format::FormatError),
    #[error(transparent)]
    TooLarge(#[from] plan::TooLarge),
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error("chunk {chunk}: {status:?}")]
    Chunk { chunk: usize, status: ChunkStatus },
    #[error("chunk {chunk}: checksum mismatch")]
    Checksum { chunk: usize },
    #[error(transparent)]
    Filter(#[from] crate::filter::FilterError),
    #[error("decoder configuration not supported: {0}")]
    Unsupported(String),
    #[error("writing decoded output: {0}")]
    Write(std::io::Error),
}

impl ChunkStatus {
    fn from_code(code: u32) -> Option<Self> {
        Some(match code {
            0 => return None,
            1 => ChunkStatus::Truncated,
            2 => ChunkStatus::ZeroOffset,
            3 => ChunkStatus::OffsetBeforeStart,
            4 => ChunkStatus::OutputOverflow,
            5 => ChunkStatus::SizeMismatch,
            6 => ChunkStatus::BadSequence,
            _ => ChunkStatus::Unknown,
        })
    }
}

/// Which decode kernel to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeKernel {
    /// M2: one invocation decodes one whole chunk, byte by byte.
    Naive,
    /// M5: one workgroup per chunk; invocation 0 decodes serially and hands
    /// copies of at least `long_copy` bytes to the whole workgroup.
    Cooperative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecoderConfig {
    pub kernel: DecodeKernel,
    /// Invocations per workgroup.
    pub workgroup: u32,
    /// Cooperative kernel only: copies at least this long are done by the
    /// whole workgroup; shorter ones by invocation 0.
    pub long_copy: u32,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        DecoderConfig {
            kernel: DecodeKernel::Cooperative,
            workgroup: 32,
            long_copy: 16,
        }
    }
}

/// GPU decoder for LZ4 and GLZ files (the file's codec picks the kernel).
pub struct GpuDecoder {
    pipeline: wgpu::ComputePipeline,
    glz_pipeline: wgpu::ComputePipeline,
    filters: FilterKernels,
    config: DecoderConfig,
    memory_budget: u64,
}

fn compute_pipeline(
    ctx: &Context,
    label: &str,
    source: &str,
    constants: &[(&str, f64)],
) -> wgpu::ComputePipeline {
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    ctx.device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants,
                ..Default::default()
            },
            cache: None,
        })
}

impl GpuDecoder {
    /// The default decoder (see [`DecoderConfig::default`]).
    pub fn new(ctx: &Context) -> Self {
        Self::with_config(ctx, DecoderConfig::default()).expect("default decoder config is valid")
    }

    /// Compiles the decode pipeline; reuse the decoder across calls.
    pub fn with_config(ctx: &Context, config: DecoderConfig) -> Result<Self, GpuDecodeError> {
        let limits = ctx.device_limits();
        if config.workgroup == 0
            || config.workgroup > limits.max_compute_invocations_per_workgroup
            || config.workgroup > limits.max_compute_workgroup_size_x
        {
            return Err(GpuDecodeError::Unsupported(format!(
                "workgroup size {}",
                config.workgroup
            )));
        }
        let (label, source) = match config.kernel {
            DecodeKernel::Naive => ("lz4 decode (naive)", NAIVE_SHADER),
            DecodeKernel::Cooperative => ("lz4 decode (cooperative)", COOP_SHADER),
        };
        let mut constants = vec![("WG_SIZE", f64::from(config.workgroup))];
        if config.kernel == DecodeKernel::Cooperative {
            constants.push(("LONG_COPY", f64::from(config.long_copy.max(1))));
        }
        let pipeline = compute_pipeline(ctx, label, source, &constants);
        let glz_pipeline = compute_pipeline(
            ctx,
            "glz decode",
            GLZ_SHADER,
            &[("WG_SIZE", f64::from(GLZ_WORKGROUP))],
        );
        Ok(GpuDecoder {
            pipeline,
            glz_pipeline,
            filters: FilterKernels::new(ctx),
            config,
            memory_budget: crate::encode::DEFAULT_GPU_MEMORY,
        })
    }

    fn pipeline_for(&self, codec: format::Codec) -> &wgpu::ComputePipeline {
        match codec {
            format::Codec::Glz | format::Codec::GlzE => &self.glz_pipeline,
            format::Codec::Stored | format::Codec::Lz4 => &self.pipeline,
        }
    }

    /// Caps the GPU memory one batch may use (default
    /// [`crate::encode::DEFAULT_GPU_MEMORY`]); large files decode in batches.
    pub fn with_memory_budget(mut self, bytes: u64) -> Self {
        self.memory_budget = bytes;
        self
    }

    /// Decompresses a whole `.gpcz` file from `reader` into `writer`, a batch
    /// at a time, and returns the bytes written.
    pub fn decompress_stream<R: Read + Seek, W: Write>(
        &self,
        ctx: &Context,
        reader: &mut R,
        writer: &mut W,
        verify: bool,
    ) -> Result<u64, GpuDecodeError> {
        let index = format::read_index(reader)?;
        let data_offset = index.data_offset();
        let mut written = 0u64;
        self.decode_batches(
            ctx,
            &index,
            0..index.chunks.len(),
            verify,
            &mut |span| read_span(reader, data_offset, span).map(Cow::Owned),
            &mut |bytes| {
                writer.write_all(&bytes).map_err(GpuDecodeError::Write)?;
                written += bytes.len() as u64;
                Ok(())
            },
        )?;
        Ok(written)
    }

    /// The batch loop behind every decode: splits `chunks` into batches that
    /// fit the memory budget and binding limit, and pipelines them two at a
    /// time (batch i + 1 is uploaded and dispatched before batch i is read
    /// back). `read` returns a span of the data section; `sink` receives each
    /// batch's output in order.
    fn decode_batches<'a>(
        &self,
        ctx: &Context,
        index: &Index,
        chunks: Range<usize>,
        verify: bool,
        read: &mut dyn FnMut(Range<u64>) -> Result<Cow<'a, [u8]>, GpuDecodeError>,
        sink: &mut dyn FnMut(Vec<u8>) -> Result<(), GpuDecodeError>,
    ) -> Result<(), GpuDecodeError> {
        let limit = batch_limit(ctx);
        // Two batches are in flight at once.
        let budget = self.memory_budget / 2;
        let batches = plan::batches(index, chunks, budget, limit);
        log::debug!(
            "GPU decoder: {} batch(es), budget {} MiB",
            batches.len(),
            self.memory_budget >> 20
        );
        let mut pending: Option<PreparedDecode> = None;
        for range in batches {
            let plan = plan::plan(index, range.clone(), limit)?;
            let src = read(plan.src.clone())?;
            let prepared = self.prepare(ctx, index.clone(), range, &plan, &src)?;
            if let Some(p) = &prepared {
                self.dispatch(ctx, p, None);
            }
            if let Some(previous) = pending.take() {
                sink(self.finish(ctx, &previous, verify)?)?;
            }
            pending = prepared;
        }
        if let Some(previous) = pending {
            sink(self.finish(ctx, &previous, verify)?)?;
        }
        Ok(())
    }

    /// Decompresses a whole `.gpcz` file held in memory.
    pub fn decompress(
        &self,
        ctx: &Context,
        file: &[u8],
        verify: bool,
    ) -> Result<Vec<u8>, GpuDecodeError> {
        let index = Index::parse(file)?;
        let data = &file[index.data_offset() as usize..];
        index.validate(data.len() as u64)?;
        let mut out = Vec::new();
        self.decode_batches(
            ctx,
            &index,
            0..index.chunks.len(),
            verify,
            &mut |span| Ok(Cow::Borrowed(&data[span.start as usize..span.end as usize])),
            &mut |bytes| {
                append_batch(&mut out, bytes);
                Ok(())
            },
        )?;
        Ok(out)
    }

    /// Decompresses `[offset, offset + len)`, reading only the index and the
    /// chunks that overlap it, and uploading only those.
    pub fn decompress_range<R: Read + Seek>(
        &self,
        ctx: &Context,
        reader: &mut R,
        offset: u64,
        len: u64,
        verify: bool,
    ) -> Result<Vec<u8>, GpuDecodeError> {
        let index = format::read_index(reader)?;
        let chunks = index.chunks_for_range(offset, len)?;
        let data_offset = index.data_offset();
        let mut out = Vec::new();
        self.decode_batches(
            ctx,
            &index,
            chunks.clone(),
            verify,
            &mut |span| read_span(reader, data_offset, span).map(Cow::Owned),
            &mut |bytes| {
                append_batch(&mut out, bytes);
                Ok(())
            },
        )?;
        let first = chunks.start as u64 * u64::from(index.header.chunk_size);
        out.drain(..(offset.saturating_sub(first) as usize).min(out.len()));
        out.truncate(len as usize);
        Ok(out)
    }

    /// Uploads a whole in-memory file for decoding; `None` if it has no chunks.
    /// With [`dispatch`](Self::dispatch) and [`finish`](Self::finish), this
    /// lets benchmarks time the kernel apart from transfers.
    pub fn prepare_file(
        &self,
        ctx: &Context,
        file: &[u8],
    ) -> Result<Option<PreparedDecode>, GpuDecodeError> {
        let index = Index::parse(file)?;
        let data = &file[index.data_offset() as usize..];
        index.validate(data.len() as u64)?;
        let chunks = 0..index.chunks.len();
        let plan = plan::plan(&index, chunks.clone(), batch_limit(ctx))?;
        let src = &data[plan.src.start as usize..plan.src.end as usize];
        self.prepare(ctx, index, chunks, &plan, src)
    }

    fn prepare(
        &self,
        ctx: &Context,
        index: Index,
        chunks: Range<usize>,
        plan: &plan::DecodePlan,
        src: &[u8],
    ) -> Result<Option<PreparedDecode>, GpuDecodeError> {
        if index.header.codec == format::Codec::GlzE {
            return Err(GpuDecodeError::Unsupported(
                "GLZ-E decoding on the GPU is not implemented yet".into(),
            ));
        }
        if plan.chunks.is_empty() {
            return Ok(None);
        }
        let device = &ctx.device;
        // Bindings can't be empty; a zero-byte payload still gets one word.
        let src_buf = if src.is_empty() {
            ctx.upload(&[0; 4], BufferUsages::STORAGE)
        } else {
            ctx.upload(src, BufferUsages::STORAGE)
        };
        let desc_buf = ctx.upload(bytemuck::cast_slice(&plan.chunks), BufferUsages::STORAGE);
        let dst_size = format::pad4(plan.dst_len).max(4);
        let dst_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("decode output"),
            size: dst_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let status_size = 4 * plan.chunks.len() as u64;
        let status_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("decode status"),
            size: status_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz4 decode (naive)"),
            layout: &self
                .pipeline_for(index.header.codec)
                .get_bind_group_layout(0),
            entries: &[
                (0, &src_buf),
                (1, &desc_buf),
                (2, &dst_buf),
                (3, &status_buf),
            ]
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }),
        });
        let unfilter = self.prepare_unfilter(ctx, &index, chunks.clone(), plan, &dst_buf)?;
        Ok(Some(PreparedDecode {
            index,
            chunks,
            dst_len: plan.dst_len,
            dst_buf,
            status_buf,
            bind_group,
            unfilter,
        }))
    }

    /// For chunks with a filter: the decoded (still filtered) bytes are copied
    /// to a compact scratch buffer, then the inverse filter writes them back
    /// into the output. `None` if no chunk is filtered.
    fn prepare_unfilter(
        &self,
        ctx: &Context,
        index: &Index,
        chunks: Range<usize>,
        plan: &plan::DecodePlan,
        dst_buf: &wgpu::Buffer,
    ) -> Result<Option<Unfilter>, GpuDecodeError> {
        let mut jobs = Vec::new();
        let mut copies: Vec<(u64, u64, u64)> = Vec::new(); // (output, scratch, bytes)
        let mut scratch_len = 0u64;
        for (entry, desc) in index.chunks[chunks].iter().zip(&plan.chunks) {
            if entry.filter == Filter::None {
                continue;
            }
            let bytes = format::pad4(u64::from(desc.uncomp_size));
            jobs.push(FilterJob {
                filter: entry.filter,
                src_offset: scratch_len as u32,
                dst_offset: desc.dst_offset,
                len: desc.uncomp_size,
            });
            let from = u64::from(desc.dst_offset);
            match copies.last_mut() {
                // Adjacent chunks go in one copy.
                Some((at, _, n)) if *at + *n == from => *n += bytes,
                _ => copies.push((from, scratch_len, bytes)),
            }
            scratch_len += bytes;
        }
        if jobs.is_empty() {
            return Ok(None);
        }
        let scratch = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("unfilter scratch"),
            size: scratch_len,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pass = self
            .filters
            .prepare(ctx, Direction::Inverse, &scratch, dst_buf, &jobs)?;
        Ok(Some(Unfilter {
            scratch,
            copies,
            pass,
        }))
    }

    /// Records and submits the decode. The output buffer must be zero (fresh
    /// from [`prepare_file`](Self::prepare_file)), so dispatching again first
    /// clears it.
    pub fn dispatch(&self, ctx: &Context, p: &PreparedDecode, timer: Option<&crate::GpuTimer>) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.clear_buffer(&p.dst_buf, 0, None);
        // With an unfilter stage, the timer spans decode, copies and filters.
        let decode_writes = match (timer, &p.unfilter) {
            (Some(t), Some(_)) => Some(t.begin_writes()),
            (Some(t), None) => Some(t.pass_writes()),
            (None, _) => None,
        };
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lz4 decode"),
                timestamp_writes: decode_writes,
            });
            pass.set_pipeline(self.pipeline_for(p.index.header.codec));
            pass.set_bind_group(0, &p.bind_group, &[]);
            let chunks = p.chunks.len() as u32;
            let (x, y) = dispatch_grid(
                match (p.index.header.codec, self.config.kernel) {
                    (format::Codec::Glz, _) => chunks,
                    (_, DecodeKernel::Naive) => chunks.div_ceil(self.config.workgroup),
                    (_, DecodeKernel::Cooperative) => chunks,
                },
                ctx.device_limits().max_compute_workgroups_per_dimension,
            );
            pass.dispatch_workgroups(x, y, 1);
        }
        if let Some(u) = &p.unfilter {
            for &(from, to, bytes) in &u.copies {
                encoder.copy_buffer_to_buffer(&p.dst_buf, from, &u.scratch, to, bytes);
            }
            let writes = timer.map(crate::GpuTimer::end_writes);
            self.filters.record(ctx, &mut encoder, &u.pass, writes);
        }
        if let Some(timer) = timer {
            timer.resolve(&mut encoder);
        }
        ctx.queue.submit([encoder.finish()]);
    }

    /// Reads back statuses and output, then checks checksums if asked.
    pub fn finish(
        &self,
        ctx: &Context,
        p: &PreparedDecode,
        verify: bool,
    ) -> Result<Vec<u8>, GpuDecodeError> {
        let status: Vec<u32> =
            bytemuck::pod_collect_to_vec(&ctx.read_buffer(&p.status_buf, p.status_buf.size())?);
        if let Some((k, s)) = status
            .iter()
            .enumerate()
            .find_map(|(k, &code)| ChunkStatus::from_code(code).map(|s| (k, s)))
        {
            return Err(GpuDecodeError::Chunk {
                chunk: p.chunks.start + k,
                status: s,
            });
        }
        let mut out = ctx.read_buffer(&p.dst_buf, p.dst_buf.size())?;
        out.truncate(p.dst_len as usize);

        if verify && p.index.header.checksums {
            let chunk_size = p.index.header.chunk_size as usize;
            for (k, (bytes, entry)) in out
                .chunks(chunk_size)
                .zip(&p.index.chunks[p.chunks.clone()])
                .enumerate()
            {
                if format::checksum(bytes) != entry.checksum {
                    return Err(GpuDecodeError::Checksum {
                        chunk: p.chunks.start + k,
                    });
                }
            }
        }
        Ok(out)
    }
}

/// GPU buffers for one decode, ready to [`dispatch`](GpuDecoder::dispatch).
pub struct PreparedDecode {
    index: Index,
    chunks: Range<usize>,
    dst_len: u64,
    dst_buf: wgpu::Buffer,
    status_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    unfilter: Option<Unfilter>,
}

/// The inverse-filter stage of a decode (see `GpuDecoder::prepare_unfilter`).
struct Unfilter {
    scratch: wgpu::Buffer,
    /// `(output offset, scratch offset, bytes)` copies, all 4-aligned.
    copies: Vec<(u64, u64, u64)>,
    pass: FilterPass,
}

impl PreparedDecode {
    /// Uncompressed bytes this decode produces.
    pub fn output_len(&self) -> u64 {
        self.dst_len
    }
}

/// Reads `span` (relative to the data section at `data_offset`) from `reader`.
fn read_span<R: Read + Seek>(
    reader: &mut R,
    data_offset: u64,
    span: Range<u64>,
) -> Result<Vec<u8>, GpuDecodeError> {
    let mut bytes = vec![0u8; (span.end - span.start) as usize];
    reader
        .seek(SeekFrom::Start(data_offset + span.start))
        .map_err(format::ReadError::from)?;
    reader
        .read_exact(&mut bytes)
        .map_err(format::ReadError::from)?;
    Ok(bytes)
}

/// Largest span one decode may bind: the storage binding limit, and u32 shader addressing.
fn batch_limit(ctx: &Context) -> u64 {
    ctx.device_limits()
        .max_storage_buffer_binding_size
        .min(u64::from(u32::MAX) & !3)
}

/// Appends a decoded batch to `out`, taking it over without a copy when `out`
/// is still empty (a single-batch decode returns the readback as is).
fn append_batch(out: &mut Vec<u8>, batch: Vec<u8>) {
    if out.is_empty() {
        *out = batch;
    } else {
        out.extend_from_slice(&batch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_batch_is_taken_over_without_a_copy() {
        let mut out = Vec::new();
        let batch = vec![1u8, 2, 3];
        let ptr = batch.as_ptr();
        append_batch(&mut out, batch);
        assert_eq!(out, [1, 2, 3]);
        assert_eq!(out.as_ptr(), ptr, "first batch was copied");
    }

    #[test]
    fn later_batches_are_appended_in_order() {
        let mut out = Vec::new();
        append_batch(&mut out, vec![1, 2]);
        append_batch(&mut out, vec![]);
        append_batch(&mut out, vec![3]);
        assert_eq!(out, [1, 2, 3]);
    }
}
