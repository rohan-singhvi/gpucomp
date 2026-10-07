//! GPU decompression.

pub mod plan;

use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;

use format::{Filter, Index};
use wgpu::BufferUsages;

use crate::{dispatch_grid, Context, GpuError};

const SHADER: &str = include_str!("../../shaders/lz4_decode_naive.wgsl");
const WG_SIZE: u32 = 64;

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
    #[error("chunk {chunk}: filter not supported by this decoder")]
    UnsupportedFilter { chunk: usize },
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
            _ => ChunkStatus::Unknown,
        })
    }
}

/// Naive GPU LZ4 decoder (M2): one invocation decodes one whole chunk.
pub struct Lz4GpuDecoder {
    pipeline: wgpu::ComputePipeline,
}

impl Lz4GpuDecoder {
    /// Compiles the decode pipeline; reuse the decoder across calls.
    pub fn new(ctx: &Context) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("lz4 decode (naive)"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("lz4 decode (naive)"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("WG_SIZE", f64::from(WG_SIZE))],
                    ..Default::default()
                },
                cache: None,
            });
        Lz4GpuDecoder { pipeline }
    }

    /// Decompresses a whole `.gpcz` file held in memory.
    pub fn decompress(
        &self,
        ctx: &Context,
        file: &[u8],
        verify: bool,
    ) -> Result<Vec<u8>, GpuDecodeError> {
        match self.prepare_file(ctx, file)? {
            None => Ok(Vec::new()),
            Some(prepared) => {
                self.dispatch(ctx, &prepared, None);
                self.finish(ctx, &prepared, verify)
            }
        }
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
        let plan = plan::plan(&index, chunks.clone(), batch_limit(ctx))?;
        let mut src = vec![0u8; (plan.src.end - plan.src.start) as usize];
        reader
            .seek(SeekFrom::Start(index.data_offset() + plan.src.start))
            .map_err(format::ReadError::from)?;
        reader
            .read_exact(&mut src)
            .map_err(format::ReadError::from)?;
        let mut out = self.decode(ctx, &index, chunks.clone(), &plan, &src, verify)?;
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

    /// Decodes `chunks` of `index` from `src` (the plan's payload span).
    fn decode(
        &self,
        ctx: &Context,
        index: &Index,
        chunks: Range<usize>,
        plan: &plan::DecodePlan,
        src: &[u8],
        verify: bool,
    ) -> Result<Vec<u8>, GpuDecodeError> {
        match self.prepare(ctx, index.clone(), chunks, plan, src)? {
            None => Ok(Vec::new()),
            Some(prepared) => {
                self.dispatch(ctx, &prepared, None);
                self.finish(ctx, &prepared, verify)
            }
        }
    }

    fn prepare(
        &self,
        ctx: &Context,
        index: Index,
        chunks: Range<usize>,
        plan: &plan::DecodePlan,
        src: &[u8],
    ) -> Result<Option<PreparedDecode>, GpuDecodeError> {
        for (chunk, entry) in index.chunks[chunks.clone()].iter().enumerate() {
            if entry.filter != Filter::None {
                return Err(GpuDecodeError::UnsupportedFilter {
                    chunk: chunks.start + chunk,
                });
            }
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
            layout: &self.pipeline.get_bind_group_layout(0),
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
        Ok(Some(PreparedDecode {
            index,
            chunks,
            dst_len: plan.dst_len,
            dst_buf,
            status_buf,
            bind_group,
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
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lz4 decode (naive)"),
                timestamp_writes: timer.map(crate::GpuTimer::pass_writes),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &p.bind_group, &[]);
            let (x, y) = dispatch_grid(
                (p.chunks.len() as u32).div_ceil(WG_SIZE),
                ctx.device_limits().max_compute_workgroups_per_dimension,
            );
            pass.dispatch_workgroups(x, y, 1);
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

/// GPU buffers for one decode, ready to [`dispatch`](Lz4GpuDecoder::dispatch).
pub struct PreparedDecode {
    index: Index,
    chunks: Range<usize>,
    dst_len: u64,
    dst_buf: wgpu::Buffer,
    status_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl PreparedDecode {
    /// Uncompressed bytes this decode produces.
    pub fn output_len(&self) -> u64 {
        self.dst_len
    }
}

/// Largest span one decode may bind: the storage binding limit, and u32 shader addressing.
fn batch_limit(ctx: &Context) -> u64 {
    ctx.device_limits()
        .max_storage_buffer_binding_size
        .min(u64::from(u32::MAX) & !3)
}
