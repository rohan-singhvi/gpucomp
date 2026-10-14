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
const GLZE_SHADER: &str = include_str!("../../shaders/glze_decode.wgsl");
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
    /// GLZ-E only: an entropy-coded stream ends early.
    StreamTruncated = 7,
    /// GLZ-E only: unknown stream mode or nonzero reserved bits.
    StreamBadMode = 8,
    /// GLZ-E only: code lengths over 11 or not a complete prefix code.
    StreamBadTable = 9,
    /// GLZ-E only: a Huffman lane decodes past its words.
    StreamLaneOverrun = 10,
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
    #[error(transparent)]
    Cpu(#[from] cpu::container::CpuError),
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
            7 => ChunkStatus::StreamTruncated,
            8 => ChunkStatus::StreamBadMode,
            9 => ChunkStatus::StreamBadTable,
            10 => ChunkStatus::StreamLaneOverrun,
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
    /// GLZ-E entropy decode (rebuilds GLZ blocks for `glz_pipeline`).
    glze_pipeline: wgpu::ComputePipeline,
    filters: FilterKernels,
    config: DecoderConfig,
    memory_budget: u64,
    /// Reusable batch buffers: two sets, as two batches are in flight.
    buffers: std::sync::Mutex<[DecodeBuffers; 2]>,
}

/// GPU buffers a decode batch reuses, grown when a bigger batch arrives.
/// Stale contents are harmless: inputs and descriptors are rewritten, the
/// output and images are cleared by every dispatch, and statuses are written
/// for every chunk.
#[derive(Default)]
struct DecodeBuffers {
    upload: Option<wgpu::Buffer>,
    src: Option<wgpu::Buffer>,
    desc: Option<wgpu::Buffer>,
    dst: Option<wgpu::Buffer>,
    status: Option<wgpu::Buffer>,
    readback: Option<wgpu::Buffer>,
    image: Option<wgpu::Buffer>,
    glze_desc: Option<wgpu::Buffer>,
    glze_status: Option<wgpu::Buffer>,
    status_read: Option<wgpu::Buffer>,
    glze_status_read: Option<wgpu::Buffer>,
    /// The submission of the last copy out of `upload`: mapping it again
    /// waits for that one only, not for later batches' kernels.
    upload_done: Option<wgpu::SubmissionIndex>,
    /// Buffers created so far.
    allocations: u32,
}

#[derive(Clone, Copy)]
enum Slot {
    Upload,
    Src,
    Desc,
    Dst,
    Status,
    Readback,
    Image,
    GlzeDesc,
    GlzeStatus,
    StatusRead,
    GlzeStatusRead,
}

/// Where a batch's buffers come from: a reusable set, or fresh ones
/// ([`GpuDecoder::prepare_file`], whose buffers outlive the call).
struct Alloc<'a>(Option<&'a mut DecodeBuffers>);

impl Alloc<'_> {
    /// A buffer of at least `size` bytes (at least 4) with `usage`.
    fn get(
        &mut self,
        ctx: &Context,
        slot: Slot,
        label: &str,
        size: u64,
        usage: BufferUsages,
    ) -> wgpu::Buffer {
        let size = format::pad4(size).max(4);
        let create = || {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let Some(set) = self.0.as_deref_mut() else {
            return create();
        };
        let DecodeBuffers {
            upload,
            src,
            desc,
            dst,
            status,
            readback,
            image,
            glze_desc,
            glze_status,
            status_read,
            glze_status_read,
            upload_done: _,
            allocations,
        } = set;
        let cached = match slot {
            Slot::Upload => upload,
            Slot::Src => src,
            Slot::Desc => desc,
            Slot::Dst => dst,
            Slot::Status => status,
            Slot::Readback => readback,
            Slot::Image => image,
            Slot::GlzeDesc => glze_desc,
            Slot::GlzeStatus => glze_status,
            Slot::StatusRead => status_read,
            Slot::GlzeStatusRead => glze_status_read,
        };
        match cached {
            Some(buf) if buf.size() >= size => buf.clone(),
            _ => {
                // Drop the old one first so both never coexist.
                *cached = None;
                *allocations += 1;
                let buf = create();
                *cached = Some(buf.clone());
                buf
            }
        }
    }

    /// `bytes` in a storage buffer: through a reused mapped staging buffer
    /// and a GPU copy (fresh mapped memory would page-fault on every call),
    /// or a fresh buffer.
    fn upload(
        &mut self,
        ctx: &Context,
        staging: Slot,
        slot: Slot,
        label: &str,
        bytes: &[u8],
    ) -> Result<wgpu::Buffer, GpuError> {
        let size = format::pad4(bytes.len() as u64).max(4);
        if self.0.is_none() {
            let mut padded = bytes.to_vec();
            padded.resize(size as usize, 0);
            return Ok(ctx.upload(&padded, BufferUsages::STORAGE));
        }
        let up = self.get(
            ctx,
            staging,
            "decode upload",
            size,
            BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
        );
        // Wait only for this buffer's last copy (long done in a pipeline), not
        // for the kernels queued since.
        let last = self.0.as_ref().and_then(|set| set.upload_done.clone());
        map_after(ctx, &[(&up, size)], wgpu::MapMode::Write, last)?;
        let written = up.get_mapped_range_mut(..size).map(|mut view| {
            view.slice(..bytes.len()).copy_from_slice(bytes);
            view.slice(bytes.len()..).fill(0);
        });
        up.unmap();
        written.map_err(|e| GpuError::Readback(e.to_string()))?;
        let buf = self.get(
            ctx,
            slot,
            label,
            size,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&up, 0, &buf, 0, size);
        let done = ctx.queue.submit([encoder.finish()]);
        if let Some(set) = self.0.as_deref_mut() {
            set.upload_done = Some(done);
        }
        Ok(buf)
    }
}

/// Maps `buffers` (each its first `len` bytes) once submission `after` (or,
/// if None, everything submitted so far) has finished.
fn map_after(
    ctx: &Context,
    buffers: &[(&wgpu::Buffer, u64)],
    mode: wgpu::MapMode,
    after: Option<wgpu::SubmissionIndex>,
) -> Result<(), GpuError> {
    let (tx, rx) = std::sync::mpsc::channel();
    for &(buffer, len) in buffers {
        let tx = tx.clone();
        buffer.map_async(mode, ..len, move |r| drop(tx.send(r)));
    }
    drop(tx);
    ctx.device.poll(wgpu::PollType::Wait {
        submission_index: after,
        timeout: None,
    })?;
    let mut result = Ok(());
    for r in rx.iter() {
        if let Err(e) = r {
            result = Err(GpuError::Readback(e.to_string()));
        }
    }
    result
}

/// A scoped thread's result, its panic passed on.
fn joined<T>(handle: std::thread::ScopedJoinHandle<'_, T>) -> T {
    handle
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

/// GPU bytes per decode batch at most ([`plan::batch_bytes`]): about 150 MB
/// of LZ4 output. Smaller batches let one batch's readback overlap the next
/// one's kernels; much smaller ones don't fill the GPU, since every pass has
/// a latency floor (DECISIONS "Pipelined GPU batches").
pub const DECODE_BATCH_BYTES: u64 = 512 << 20;

/// The GPU bytes one decode batch may use under `memory_budget`.
fn batch_budget(memory_budget: u64) -> u64 {
    // Two batches are in flight at once.
    (memory_budget / 2).min(DECODE_BATCH_BYTES)
}

/// `size` bytes of `buffer` from its start, as a binding.
fn binding(buffer: &wgpu::Buffer, size: u64) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset: 0,
        size: wgpu::BufferSize::new(format::pad4(size).max(4)),
    })
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
        // Compiled in parallel: one at a time took ~5 ms on an M4 Pro.
        let glz_constants = [("WG_SIZE", f64::from(GLZ_WORKGROUP))];
        let (pipeline, glz_pipeline, glze_pipeline, filters) = std::thread::scope(|scope| {
            let glz =
                scope.spawn(|| compute_pipeline(ctx, "glz decode", GLZ_SHADER, &glz_constants));
            let glze =
                scope.spawn(|| compute_pipeline(ctx, "glz-e entropy decode", GLZE_SHADER, &[]));
            let filters = scope.spawn(|| FilterKernels::new(ctx));
            let pipeline = compute_pipeline(ctx, label, source, &constants);
            (pipeline, joined(glz), joined(glze), joined(filters))
        });
        Ok(GpuDecoder {
            pipeline,
            glz_pipeline,
            glze_pipeline,
            filters,
            config,
            memory_budget: crate::encode::DEFAULT_GPU_MEMORY,
            buffers: std::sync::Mutex::default(),
        })
    }

    fn pipeline_for(&self, codec: format::Codec) -> &wgpu::ComputePipeline {
        match codec {
            format::Codec::Glz | format::Codec::GlzE => &self.glz_pipeline,
            format::Codec::Stored | format::Codec::Lz4 => &self.pipeline,
        }
    }

    /// GPU buffers the decoder has created for its reusable batch buffers.
    pub fn buffer_allocations(&self) -> u32 {
        self.buffers().iter().map(|b| b.allocations).sum()
    }

    /// The reusable buffers. A panic while they were held may have left a
    /// staging buffer mapped, so poisoned buffers start over.
    fn buffers(&self) -> std::sync::MutexGuard<'_, [DecodeBuffers; 2]> {
        self.buffers.lock().unwrap_or_else(|poisoned| {
            let mut guard = poisoned.into_inner();
            *guard = Default::default();
            guard
        })
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
                writer.write_all(bytes).map_err(GpuDecodeError::Write)?;
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
        sink: &mut dyn FnMut(&[u8]) -> Result<(), GpuDecodeError>,
    ) -> Result<(), GpuDecodeError> {
        let limit = batch_limit(ctx);
        let budget = batch_budget(self.memory_budget);
        let batches = plan::batches(index, chunks, budget, limit);
        log::debug!(
            "GPU decoder: {} batch(es), budget {} MiB",
            batches.len(),
            self.memory_budget >> 20
        );
        let mut sets = self.buffers();
        let mut pending: Option<PreparedDecode> = None;
        for (k, range) in batches.into_iter().enumerate() {
            let plan = plan::plan(index, range.clone(), limit)?;
            let src = read(plan.src.clone())?;
            // Batch k - 1 may still be in flight on the other set.
            let alloc = Alloc(Some(&mut sets[k % 2]));
            let prepared = self.prepare(ctx, index.clone(), range, &plan, &src, alloc)?;
            if let Some(p) = &prepared {
                self.dispatch(ctx, p, None);
            }
            if let Some(previous) = pending.take() {
                self.finish_with(ctx, &previous, verify, sink)?;
            }
            pending = prepared;
        }
        if let Some(previous) = pending {
            self.finish_with(ctx, &previous, verify, sink)?;
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
        let mut out = Vec::with_capacity(index.header.total_size as usize);
        self.decode_batches(
            ctx,
            &index,
            0..index.chunks.len(),
            verify,
            &mut |span| Ok(Cow::Borrowed(&data[span.start as usize..span.end as usize])),
            &mut |bytes| {
                out.extend_from_slice(bytes);
                Ok(())
            },
        )?;
        Ok(out)
    }

    /// Decodes chunks `chunks` of the in-memory `file`, handing the output to
    /// `sink` in order, a batch at a time. For callers that split a file's
    /// chunks between decoders (`gpu::hybrid`).
    pub fn decompress_chunks(
        &self,
        ctx: &Context,
        file: &[u8],
        chunks: Range<usize>,
        verify: bool,
        sink: &mut dyn FnMut(&[u8]) -> Result<(), GpuDecodeError>,
    ) -> Result<(), GpuDecodeError> {
        let index = Index::parse(file)?;
        let data = &file[index.data_offset() as usize..];
        index.validate(data.len() as u64)?;
        if chunks.end > index.chunks.len() || chunks.start > chunks.end {
            return Err(GpuDecodeError::Unsupported(format!(
                "chunks {chunks:?} of a {}-chunk file",
                index.chunks.len()
            )));
        }
        self.decode_batches(
            ctx,
            &index,
            chunks,
            verify,
            &mut |span| Ok(Cow::Borrowed(&data[span.start as usize..span.end as usize])),
            &mut |bytes| sink(bytes),
        )
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
                out.extend_from_slice(bytes);
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
        self.prepare(ctx, index, chunks, &plan, src, Alloc(None))
    }

    fn prepare(
        &self,
        ctx: &Context,
        index: Index,
        chunks: Range<usize>,
        plan: &plan::DecodePlan,
        src: &[u8],
        mut alloc: Alloc,
    ) -> Result<Option<PreparedDecode>, GpuDecodeError> {
        if plan.chunks.is_empty() {
            return Ok(None);
        }
        let device = &ctx.device;
        let src_buf = alloc.upload(ctx, Slot::Upload, Slot::Src, "decode input", src)?;
        let src_size = format::pad4(src.len() as u64).max(4);
        // GLZ-E: the entropy stage rebuilds each chunk's GLZ block in an image
        // buffer, and the GLZ kernel decodes the images.
        let glze = if index.header.codec == format::Codec::GlzE {
            Some(self.prepare_glze(ctx, plan, src, (&src_buf, src_size), &mut alloc)?)
        } else {
            None
        };
        let descs: &[u8] = match &glze {
            Some(g) => bytemuck::cast_slice(&g.glz_descs),
            None => bytemuck::cast_slice(&plan.chunks),
        };
        let desc_buf = alloc.get(
            ctx,
            Slot::Desc,
            "decode chunks",
            descs.len() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        ctx.queue.write_buffer(&desc_buf, 0, descs);
        let (glz_src, glz_src_size) = match &glze {
            Some(g) => (&g.image_buf, g.image_size),
            None => (&src_buf, src_size),
        };
        let dst_size = format::pad4(plan.dst_len).max(4);
        let dst_buf = alloc.get(
            ctx,
            Slot::Dst,
            "decode output",
            dst_size,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        let status_size = 4 * plan.chunks.len() as u64;
        let status_buf = alloc.get(
            ctx,
            Slot::Status,
            "decode status",
            status_size,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        // Readback staging for the output and statuses, copied in the batch's
        // own submission (see `dispatch`), so `finish` waits for this batch
        // only.
        let readback = if alloc.0.is_some() {
            let out = alloc.get(
                ctx,
                Slot::Readback,
                "decode readback",
                dst_size,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            );
            let status = alloc.get(
                ctx,
                Slot::StatusRead,
                "decode status readback",
                status_size,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            );
            let glze_status = glze.as_ref().map(|g| {
                alloc.get(
                    ctx,
                    Slot::GlzeStatusRead,
                    "glz-e status readback",
                    g.status_size,
                    BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                )
            });
            Some(Readback {
                out,
                status,
                glze_status,
            })
        } else {
            None
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("decode"),
            layout: &self
                .pipeline_for(index.header.codec)
                .get_bind_group_layout(0),
            entries: &[
                (0, binding(glz_src, glz_src_size)),
                (1, binding(&desc_buf, descs.len() as u64)),
                (2, binding(&dst_buf, dst_size)),
                (3, binding(&status_buf, status_size)),
            ]
            .map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource }),
        });
        let unfilter = self.prepare_unfilter(ctx, &index, chunks.clone(), plan, &dst_buf)?;
        Ok(Some(PreparedDecode {
            index,
            chunks,
            dst_len: plan.dst_len,
            dst_size,
            dst_buf,
            status_buf,
            status_size,
            readback,
            submitted: std::sync::Mutex::new(None),
            bind_group,
            unfilter,
            glze,
        }))
    }

    /// The GLZ-E entropy stage: image layout from the chunk headers, the
    /// image and status buffers, and the GLZ descriptors that point at the
    /// images.
    fn prepare_glze(
        &self,
        ctx: &Context,
        plan: &plan::DecodePlan,
        src: &[u8],
        src_buf: (&wgpu::Buffer, u64),
        alloc: &mut Alloc,
    ) -> Result<GlzeStage, GpuDecodeError> {
        let (descs, image_len) = plan::glze_layout(&plan.chunks, src);
        let limit = batch_limit(ctx);
        if image_len > limit {
            return Err(plan::TooLarge {
                what: "GLZ-E image",
                bytes: image_len,
                limit,
            }
            .into());
        }
        let image_size = image_len.max(4);
        let image_buf = alloc.get(
            ctx,
            Slot::Image,
            "glz-e images",
            image_size,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let status_size = 4 * descs.len() as u64;
        let status_buf = alloc.get(
            ctx,
            Slot::GlzeStatus,
            "glz-e status",
            status_size,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let desc_bytes: &[u8] = bytemuck::cast_slice(&descs);
        let desc_buf = alloc.get(
            ctx,
            Slot::GlzeDesc,
            "glz-e chunks",
            desc_bytes.len() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        ctx.queue.write_buffer(&desc_buf, 0, desc_bytes);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("glz-e entropy decode"),
            layout: &self.glze_pipeline.get_bind_group_layout(0),
            entries: &[
                (0, binding(src_buf.0, src_buf.1)),
                (1, binding(&desc_buf, desc_bytes.len() as u64)),
                (2, binding(&image_buf, image_size)),
                (3, binding(&status_buf, status_size)),
            ]
            .map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource }),
        });
        let glz_descs = plan
            .chunks
            .iter()
            .zip(&descs)
            .map(|(c, g)| plan::ChunkDesc {
                src_offset: g.image_offset,
                comp_size: g.image_size | (c.comp_size & format::STORED_BIT),
                ..*c
            })
            .collect();
        Ok(GlzeStage {
            image_buf,
            image_size,
            status_buf,
            status_size,
            bind_group,
            glz_descs,
        })
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

    /// The names of the stages [`dispatch`](Self::dispatch) runs for `p`, in
    /// order, for profiling with [`dispatch_stages`](Self::dispatch_stages):
    /// GLZ-E's entropy kernel, the decode kernel, then the inverse filters
    /// (with their copies) when any chunk is filtered.
    pub fn stages(&self, p: &PreparedDecode) -> Vec<&'static str> {
        let mut stages = Vec::new();
        if p.glze.is_some() {
            stages.push("glze-entropy");
        }
        stages.push("decode");
        if p.unfilter.is_some() {
            stages.push("unfilter");
        }
        stages
    }

    /// [`dispatch`](Self::dispatch) of only the first `count`
    /// [`stages`](Self::stages), without a timer. Timing successive prefixes
    /// gives each stage's time; the output is meaningful only for all of them.
    pub fn dispatch_stages(&self, ctx: &Context, p: &PreparedDecode, count: usize) {
        self.dispatch_with(ctx, p, None, count);
    }

    /// Records and submits the decode. The output buffer must be zero (fresh
    /// from [`prepare_file`](Self::prepare_file)), so dispatching again first
    /// clears it.
    pub fn dispatch(&self, ctx: &Context, p: &PreparedDecode, timer: Option<&crate::GpuTimer>) {
        self.dispatch_with(ctx, p, timer, usize::MAX);
    }

    /// [`dispatch`](Self::dispatch) of the first `count` stages.
    fn dispatch_with(
        &self,
        ctx: &Context,
        p: &PreparedDecode,
        timer: Option<&crate::GpuTimer>,
        count: usize,
    ) {
        // Stage indices: the entropy kernel (GLZ-E only), decode, unfilter.
        let decode_stage = usize::from(p.glze.is_some());
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.clear_buffer(&p.dst_buf, 0, Some(p.dst_size));
        if let Some(g) = &p.glze {
            encoder.clear_buffer(&g.image_buf, 0, Some(g.image_size));
        }
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
            let chunks = p.chunks.len() as u32;
            let max = ctx.device_limits().max_compute_workgroups_per_dimension;
            if let Some(g) = p.glze.as_ref().filter(|_| count > 0) {
                pass.set_pipeline(&self.glze_pipeline);
                pass.set_bind_group(0, &g.bind_group, &[]);
                let (x, y) = dispatch_grid(chunks, max);
                pass.dispatch_workgroups(x, y, 1);
            }
            if count > decode_stage {
                pass.set_pipeline(self.pipeline_for(p.index.header.codec));
                pass.set_bind_group(0, &p.bind_group, &[]);
                let (x, y) = dispatch_grid(
                    match (p.index.header.codec, self.config.kernel) {
                        (format::Codec::Glz | format::Codec::GlzE, _) => chunks,
                        (_, DecodeKernel::Naive) => chunks.div_ceil(self.config.workgroup),
                        (_, DecodeKernel::Cooperative) => chunks,
                    },
                    ctx.device_limits().max_compute_workgroups_per_dimension,
                );
                pass.dispatch_workgroups(x, y, 1);
            }
        }
        if let Some(u) = p.unfilter.as_ref().filter(|_| count > decode_stage + 1) {
            for &(from, to, bytes) in &u.copies {
                encoder.copy_buffer_to_buffer(&p.dst_buf, from, &u.scratch, to, bytes);
            }
            let writes = timer.map(crate::GpuTimer::end_writes);
            self.filters.record(ctx, &mut encoder, &u.pass, writes);
        }
        if let Some(timer) = timer {
            timer.resolve(&mut encoder);
        }
        if let Some(r) = &p.readback {
            encoder.copy_buffer_to_buffer(&p.dst_buf, 0, &r.out, 0, p.dst_size);
            encoder.copy_buffer_to_buffer(&p.status_buf, 0, &r.status, 0, p.status_size);
            if let (Some(g), Some(gs)) = (&p.glze, &r.glze_status) {
                encoder.copy_buffer_to_buffer(&g.status_buf, 0, gs, 0, g.status_size);
            }
        }
        let index = ctx.queue.submit([encoder.finish()]);
        *p.submitted.lock().unwrap_or_else(|e| e.into_inner()) = Some(index);
    }

    /// Reads back statuses and output, then checks checksums if asked.
    pub fn finish(
        &self,
        ctx: &Context,
        p: &PreparedDecode,
        verify: bool,
    ) -> Result<Vec<u8>, GpuDecodeError> {
        let mut out = Vec::new();
        self.finish_with(ctx, p, verify, &mut |bytes| {
            out.extend_from_slice(bytes);
            Ok(())
        })?;
        Ok(out)
    }

    /// [`finish`](Self::finish), handing the checked output to `consume`
    /// straight from the readback buffer instead of copying it out first.
    pub fn finish_with(
        &self,
        ctx: &Context,
        p: &PreparedDecode,
        verify: bool,
        consume: &mut dyn FnMut(&[u8]) -> Result<(), GpuDecodeError>,
    ) -> Result<(), GpuDecodeError> {
        match &p.readback {
            Some(r) => {
                // Wait for this batch's submission only: the next batch's
                // kernel keeps running meanwhile.
                let after = p
                    .submitted
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                let mut maps = vec![(&r.out, p.dst_size), (&r.status, p.status_size)];
                if let (Some(gs), Some(g)) = (&r.glze_status, &p.glze) {
                    maps.push((gs, g.status_size));
                }
                let mapped = map_after(ctx, &maps, wgpu::MapMode::Read, after);
                let read = |buf: &wgpu::Buffer, len: u64| -> Result<Vec<u8>, GpuError> {
                    buf.get_mapped_range(..len)
                        .map(|view| view.to_vec())
                        .map_err(|e| GpuError::Readback(e.to_string()))
                };
                let result = mapped.map_err(GpuDecodeError::from).and_then(|()| {
                    let status: Vec<u32> =
                        bytemuck::pod_collect_to_vec(&read(&r.status, p.status_size)?);
                    let entropy = match (&r.glze_status, &p.glze) {
                        (Some(gs), Some(g)) => Some(bytemuck::pod_collect_to_vec::<u8, u32>(
                            &read(gs, g.status_size)?,
                        )),
                        _ => None,
                    };
                    let view = r
                        .out
                        .get_mapped_range(..p.dst_size)
                        .map_err(|e| GpuError::Readback(e.to_string()))?;
                    self.check(p, status, entropy, &view[..p.dst_len as usize], verify)?;
                    consume(&view[..p.dst_len as usize])
                });
                // Unmap on every path, so the reused buffers map again.
                for (buf, _) in &maps {
                    buf.unmap();
                }
                result
            }
            None => {
                let status: Vec<u32> =
                    bytemuck::pod_collect_to_vec(&ctx.read_buffer(&p.status_buf, p.status_size)?);
                let entropy = match &p.glze {
                    Some(g) => Some(bytemuck::pod_collect_to_vec::<u8, u32>(
                        &ctx.read_buffer(&g.status_buf, g.status_size)?,
                    )),
                    None => None,
                };
                let out = ctx.read_buffer(&p.dst_buf, p.dst_size)?;
                self.check(p, status, entropy, &out[..p.dst_len as usize], verify)?;
                consume(&out[..p.dst_len as usize])
            }
        }
    }

    /// Fails on the first chunk whose status is an error (an entropy-stage
    /// error first), then, if asked, on the first checksum mismatch in `out`.
    fn check(
        &self,
        p: &PreparedDecode,
        mut status: Vec<u32>,
        entropy: Option<Vec<u32>>,
        out: &[u8],
        verify: bool,
    ) -> Result<(), GpuDecodeError> {
        if let Some(entropy) = entropy {
            // An entropy-stage error comes first: the GLZ kernel then decoded
            // a partial image.
            for (s, &e) in status.iter_mut().zip(&entropy) {
                if e != 0 {
                    *s = e;
                }
            }
        }
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
        Ok(())
    }
}

/// GPU buffers for one decode, ready to [`dispatch`](GpuDecoder::dispatch).
pub struct PreparedDecode {
    index: Index,
    chunks: Range<usize>,
    dst_len: u64,
    /// Bytes of each (possibly larger, reused) buffer this batch uses.
    dst_size: u64,
    dst_buf: wgpu::Buffer,
    status_buf: wgpu::Buffer,
    status_size: u64,
    /// Reused readback staging (None: fresh reads after the fact).
    readback: Option<Readback>,
    /// The submission `dispatch` made: `finish` waits for it alone.
    submitted: std::sync::Mutex<Option<wgpu::SubmissionIndex>>,
    bind_group: wgpu::BindGroup,
    unfilter: Option<Unfilter>,
    glze: Option<GlzeStage>,
}

/// Staging buffers a batch's results are copied into by its own submission.
struct Readback {
    out: wgpu::Buffer,
    status: wgpu::Buffer,
    glze_status: Option<wgpu::Buffer>,
}

/// The entropy stage of a GLZ-E decode (see `GpuDecoder::prepare_glze`).
struct GlzeStage {
    image_buf: wgpu::Buffer,
    image_size: u64,
    status_buf: wgpu::Buffer,
    status_size: u64,
    bind_group: wgpu::BindGroup,
    /// Descriptors for the GLZ kernel, pointing at the images.
    glz_descs: Vec<plan::ChunkDesc>,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> Option<Context> {
        match Context::new(&crate::ContextOptions::default()) {
            Ok(ctx) => Some(ctx),
            Err(GpuError::NoAdapter(e)) => {
                eprintln!("skipping GPU test: no adapter ({e})");
                None
            }
            Err(e) => panic!("GPU context creation failed: {e}"),
        }
    }

    #[test]
    fn batches_are_capped_so_two_overlap_but_each_fills_the_gpu() {
        // Two batches in flight share the budget.
        assert_eq!(batch_budget(64 << 20), 32 << 20);
        // A large budget still decodes in several batches of the cap.
        assert_eq!(
            batch_budget(crate::encode::DEFAULT_GPU_MEMORY),
            DECODE_BATCH_BYTES
        );
        assert_eq!(batch_budget(2 * DECODE_BATCH_BYTES), DECODE_BATCH_BYTES);
    }

    #[test]
    fn repeated_decompress_calls_reuse_buffers() {
        let Some(ctx) = gpu() else { return };
        let input: Vec<u8> = (0..300_000u32)
            .flat_map(|i| format!("{} ", (i * 7) % 1999).into_bytes())
            .take(1 << 20)
            .collect();
        for codec in [format::Codec::Lz4, format::Codec::GlzE] {
            let encoder = match codec {
                format::Codec::Lz4 => cpu::container::Encoder::Lz4Flex,
                _ => cpu::container::Encoder::Glz(Default::default()),
            };
            let file = cpu::container::compress(
                &input,
                &cpu::container::CompressOptions {
                    codec,
                    encoder,
                    checksums: true,
                    ..Default::default()
                },
            )
            .unwrap();
            // One batch, then several (both buffer sets in use).
            for budget in [crate::encode::DEFAULT_GPU_MEMORY, 1 << 20] {
                let decoder = GpuDecoder::new(&ctx).with_memory_budget(budget);
                assert!(decoder.decompress(&ctx, &file, true).unwrap() == input);
                let allocations = decoder.buffer_allocations();
                assert!(allocations > 0);
                for _ in 0..2 {
                    assert!(decoder.decompress(&ctx, &file, true).unwrap() == input);
                }
                assert_eq!(
                    decoder.buffer_allocations(),
                    allocations,
                    "{codec:?}, budget {budget}"
                );
            }
        }
    }
}
