//! GPU LZ4 compression (M3): one workgroup per chunk, running the same
//! algorithm as `cpu::lz4::encode` so outputs are byte-identical.

use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::{mpsc, Mutex, MutexGuard};

use wgpu::BufferUsages;

use crate::{dispatch_grid, Context, GpuError};

const COMMON_SHADER: &str = include_str!("../../shaders/encode_common.wgsl");
const MATCHES_SHADER: &str = include_str!("../../shaders/encode_matches.wgsl");
const PARSE_SHADER: &str = include_str!("../../shaders/encode_parse.wgsl");
const PARSE_SEG_SHADER: &str = include_str!("../../shaders/encode_parse_seg.wgsl");
const EMIT_SHARED: &str = include_str!("../../shaders/encode_emit_shared.wgsl");
const LZ4_EMIT: &str = include_str!("../../shaders/lz4_emit.wgsl");
const GLZ_EMIT: &str = include_str!("../../shaders/glz_emit.wgsl");
const PACK_SHADER: &str = include_str!("../../shaders/encode_pack.wgsl");
/// Bytes of the `Params` uniform shared by all encode kernels.
const PARAMS_SIZE: u64 = 32;
/// Pack entry flag: this chunk isn't packed by this pass (encode_pack.wgsl).
const SKIP_BIT: u32 = 0x4000_0000;
/// Invocations per emit workgroup (independent of the match-finding block).
pub const EMIT_WG: u32 = 64;
/// Chunks per parse workgroup (one invocation each); matches the shader.
const PARSE_WG: u32 = 64;

/// Match-finding parameters; same meaning as `cpu::lz4::encode::Params`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeParams {
    /// Positions per match-finding block = workgroup size.
    pub block: u32,
    /// The workgroup hash table has `1 << hash_log` buckets.
    pub hash_log: u32,
    /// Phase 1 extends matches at most this far.
    pub probe_len: u32,
    /// Lazy parse (see `cpu::lz4::encode::Params::lazy`).
    pub lazy: bool,
    /// Hash-chain candidates per position (see `cpu::lz4::encode::Params::depth`).
    pub depth: u32,
}

impl EncodeParams {
    /// The encoder for compression `level`; must equal
    /// `cpu::lz4::encode::Params::for_level`.
    pub fn for_level(level: u8) -> Self {
        EncodeParams {
            depth: match level {
                0 | 1 => 1,
                2 => 4,
                _ => 16,
            },
            ..EncodeParams::default()
        }
    }
}

impl Default for EncodeParams {
    fn default() -> Self {
        // Level 1: block 128 trades 1.2% ratio (Silesia 1.943x vs 1.966x at
        // 64) for ~24% kernel throughput; see DECISIONS.md.
        EncodeParams {
            block: 128,
            hash_log: 12,
            probe_len: 16,
            lazy: true,
            depth: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuCompressOptions {
    /// `Lz4` or `Glz`.
    pub codec: format::Codec,
    pub chunk_size: u32,
    pub checksums: bool,
    pub level: u8,
    /// GLZ only: dependency elimination over groups of this many sequences
    /// (1..=`MAX_GROUP`).
    pub independent_groups: Option<u32>,
    /// Per-chunk filter selection (plan §4a); `level` picks the candidates.
    pub filters: FilterMode,
}

/// Whether the encoder tries filters on each chunk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FilterMode {
    #[default]
    None,
    /// Candidates compete on each chunk's leading sample ([`sample_len`]);
    /// the winner encodes the whole chunk. Same rule as the CPU's `Auto`.
    Auto,
    /// Try [`filter_candidates`]`(level)` on whole chunks and keep the
    /// smallest block.
    Exhaustive,
}

/// Bytes of each chunk's leading sample for [`FilterMode::Auto`]: an eighth
/// of the chunk, at least 4 KiB. Must equal `cpu::filter::sample_len`.
pub fn sample_len(chunk_size: u32) -> u32 {
    (chunk_size / 8).max(format::MIN_CHUNK_SIZE)
}

/// Filters tried per chunk at `level`, in tie-break order. Must equal
/// `cpu::filter::candidates`, so GPU and CPU files are identical.
pub fn filter_candidates(level: u8) -> &'static [format::Filter] {
    use format::Filter::{Delta, None, Shuffle};
    const LEVEL1: [format::Filter; 3] = [None, Shuffle { width: 4 }, Delta { width: 4 }];
    const LEVEL2: [format::Filter; 7] = [
        None,
        Shuffle { width: 2 },
        Shuffle { width: 4 },
        Shuffle { width: 8 },
        Delta { width: 2 },
        Delta { width: 4 },
        Delta { width: 8 },
    ];
    if level <= 1 {
        &LEVEL1
    } else {
        &LEVEL2
    }
}

impl Default for GpuCompressOptions {
    fn default() -> Self {
        GpuCompressOptions {
            codec: format::Codec::Lz4,
            chunk_size: format::DEFAULT_CHUNK_SIZE,
            checksums: false,
            level: 1,
            independent_groups: None,
            filters: FilterMode::None,
        }
    }
}

/// Largest dependency-elimination group the GPU parse supports. The GLZ
/// decoder resolves 64 sequences per step, so larger groups can't help it.
pub const MAX_GROUP: u32 = 64;

/// Largest hash-chain depth (candidates per position).
pub const MAX_DEPTH: u32 = 64;

/// Segments per chunk in the parallel parse (one lane each); emit finds a
/// sequence through the per-chunk table of where each segment's run starts.
pub const PARSE_SEGMENTS: u32 = 32;

#[derive(Debug, thiserror::Error)]
pub enum GpuEncodeError {
    #[error(transparent)]
    Format(#[from] format::FormatError),
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error("encoder parameters don't fit this device: {0}")]
    Unsupported(String),
    #[error("input too large: {0} chunks")]
    TooManyChunks(u64),
    #[error(transparent)]
    Filter(#[from] crate::filter::FilterError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// One chunk's LZ4 block, as reported by [`GpuEncoder::encode_blocks`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodedBlock {
    Compressed(Vec<u8>),
    /// The block would be `size` bytes, no smaller than the chunk, so the GPU
    /// skipped writing it (the container stores such chunks raw).
    Incompressible {
        size: u32,
    },
}

/// Bytes reserved per chunk for its compressed output: the LZ4 worst case
/// (`n + n/255 + 16`), rounded up to whole words.
pub fn slot_size(chunk_size: u32) -> u32 {
    format::pad4(u64::from(chunk_size) + u64::from(chunk_size) / 255 + 16) as u32
}

/// Default GPU memory budget for one encoder or decoder batch: enough to
/// encode 256 MiB (with or without sampled filter selection) in one batch.
pub const DEFAULT_GPU_MEMORY: u64 = 3 << 30;

/// GPU bytes the encoder needs per input chunk: input, upload staging,
/// match scratch (4 bytes per byte), output slot, packed data and its readback,
/// plus per-chunk sizes, info and pack entries (and, with dependency
/// elimination, the group lists). Filter selection keeps one extra input-sized
/// buffer (sampled) or two per extra candidate (exhaustive).
pub fn encoder_bytes_per_chunk(options: &GpuCompressOptions, params: &EncodeParams) -> u64 {
    let c = u64::from(options.chunk_size);
    let slot = |n: u64| u64::from(slot_size(n as u32));
    let mut bytes = 8 * c + slot(c) + 32 + 4 * u64::from(PARSE_SEGMENTS);
    if params.depth > 1 {
        // Hash-chain links, one per input byte.
        bytes += 4 * c;
    }
    if options.independent_groups.is_some() {
        bytes += 8 * u64::from(MAX_GROUP);
    }
    let extra = filter_candidates(options.level).len() as u64 - 1;
    match options.filters {
        FilterMode::None => {}
        FilterMode::Auto => {
            let s = u64::from(sample_len(options.chunk_size));
            bytes += c + extra * (s + slot(s));
        }
        FilterMode::Exhaustive => bytes += extra * (c + slot(c)),
    }
    bytes
}

/// Workgroup memory the encoder needs: the larger of the match-finding
/// kernel's hash table and the emit kernel's scans (the parse uses none).
pub fn workgroup_bytes(params: &EncodeParams) -> u32 {
    (4 << params.hash_log).max(16 * EMIT_WG + 16 + 4 * PARSE_SEGMENTS)
}

/// How many chunks one dispatch may encode so that the input, match scratch
/// and output slots each fit in `binding_limit` bytes.
pub fn chunks_per_batch(chunk_size: u32, binding_limit: u64) -> u64 {
    // Match scratch (one u32 per input byte) is the largest buffer.
    (binding_limit / (4 * u64::from(chunk_size))).max(1)
}

pub struct GpuEncoder {
    params: EncodeParams,
    layout: wgpu::BindGroupLayout,
    matches: wgpu::ComputePipeline,
    lz4: CodecPipelines,
    glz: CodecPipelines,
    pack_layout: wgpu::BindGroupLayout,
    pack: wgpu::ComputePipeline,
    /// Forward filters for per-chunk filter selection.
    filters: crate::filter::FilterKernels,
    /// Caps the chunks per batch below what the device allows (tests).
    max_batch_chunks: Option<u64>,
    memory_budget: u64,
    /// Buffers reused across `compress`/`encode_blocks` calls.
    buffers: Mutex<BufferCache>,
}

/// The codec-specific kernels: parse (size accounting) and emit.
struct CodecPipelines {
    /// Segmented parallel parse, one workgroup per chunk.
    parse: wgpu::ComputePipeline,
    /// One invocation per chunk: GLZ dependency elimination depends on the
    /// parse's history, so it can't be split into segments.
    parse_serial: wgpu::ComputePipeline,
    emit: wgpu::ComputePipeline,
}

/// Raw output of one or more encode dispatches: per-chunk compressed sizes and
/// the chunks' output slots, `slot_size` bytes each.
struct Encoded {
    sizes: Vec<u32>,
    slots: Vec<u8>,
    slot_size: usize,
}

impl Encoded {
    fn block(&self, i: usize) -> &[u8] {
        let start = i * self.slot_size;
        &self.slots[start..start + self.sizes[i] as usize]
    }
}

/// GPU buffers kept between calls and grown when a bigger batch arrives.
/// Stale contents are harmless: the input and params are rewritten, the
/// output slots are cleared by every encode, and the kernels write every
/// scratch, info, group and size entry before reading it.
#[derive(Default)]
struct BufferCache {
    input: Option<wgpu::Buffer>,
    upload: Option<wgpu::Buffer>,
    scratch: Option<wgpu::Buffer>,
    output: Option<wgpu::Buffer>,
    sizes: Option<wgpu::Buffer>,
    params: Option<wgpu::Buffer>,
    info: Option<wgpu::Buffer>,
    groups: Option<wgpu::Buffer>,
    segs: Option<wgpu::Buffer>,
    chain: Option<wgpu::Buffer>,
    entries: Option<wgpu::Buffer>,
    packed: Option<wgpu::Buffer>,
    sizes_read: Option<wgpu::Buffer>,
    data_read: Option<wgpu::Buffer>,
    /// Buffers created so far.
    allocations: u32,
    /// Bytes read back to the host since the last reset.
    read_bytes: u64,
}

impl BufferCache {
    /// The buffer in `slot` if it has at least `size` bytes, else a new one.
    fn ensure(
        slot: &mut Option<wgpu::Buffer>,
        allocations: &mut u32,
        device: &wgpu::Device,
        label: &str,
        size: u64,
        usage: BufferUsages,
    ) -> wgpu::Buffer {
        match slot {
            Some(buf) if buf.size() >= size => buf.clone(),
            _ => {
                // Drop the old one first so both never coexist.
                *slot = None;
                *allocations += 1;
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                });
                *slot = Some(buf.clone());
                buf
            }
        }
    }
}

/// `size` bytes of `buffer` from its start, as a binding.
fn binding(buffer: &wgpu::Buffer, size: u64) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset: 0,
        size: wgpu::BufferSize::new(size),
    })
}

/// Maps the first `len` bytes of `buffer`, waiting for the GPU to finish
/// with it.
fn map(
    ctx: &Context,
    buffer: &wgpu::Buffer,
    mode: wgpu::MapMode,
    len: u64,
) -> Result<(), GpuError> {
    let (tx, rx) = mpsc::channel();
    buffer.map_async(mode, ..len, move |r| drop(tx.send(r)));
    ctx.device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()
        .map_err(|e| GpuError::Readback(e.to_string()))?
        .map_err(|e| GpuError::Readback(e.to_string()))
}

/// Maps the first `len` bytes of `buffer` (MAP_READ, already filled by a
/// submitted copy), hands them to `f` and unmaps.
fn read_mapped(
    ctx: &Context,
    buffer: &wgpu::Buffer,
    len: u64,
    f: impl FnOnce(&[u8]),
) -> Result<(), GpuError> {
    map(ctx, buffer, wgpu::MapMode::Read, len)?;
    // Unmap even on error, so the (cached) buffer can be mapped again.
    let result = buffer.get_mapped_range(..len).map(|view| f(&view));
    buffer.unmap();
    result.map_err(|e| GpuError::Readback(e.to_string()))
}

impl GpuEncoder {
    /// Compiles the encoder; fails if `params` need more workgroup memory or a
    /// bigger workgroup than the device allows.
    pub fn new(ctx: &Context, params: EncodeParams) -> Result<Self, GpuEncodeError> {
        let limits = ctx.device_limits();
        let unsupported = |why: String| Err(GpuEncodeError::Unsupported(why));
        if !(1..=16).contains(&params.hash_log) {
            return unsupported(format!("hash_log {} not in 1..=16", params.hash_log));
        }
        if workgroup_bytes(&params) > limits.max_compute_workgroup_storage_size {
            return unsupported(format!(
                "{} bytes of workgroup memory needed, device allows {}",
                workgroup_bytes(&params),
                limits.max_compute_workgroup_storage_size
            ));
        }
        if params.block == 0
            || params.block > limits.max_compute_invocations_per_workgroup
            || params.block > limits.max_compute_workgroup_size_x
        {
            return unsupported(format!("workgroup size {} not supported", params.block));
        }
        if !(1..=MAX_DEPTH).contains(&params.depth) {
            return unsupported(format!("depth {} not in 1..={MAX_DEPTH}", params.depth));
        }
        // The parse marks visited positions with bit 31 of the match word, so
        // probe-capped lengths (bits 16..) must stay below 1 << 15.
        if !(4..=32_767).contains(&params.probe_len) {
            return unsupported(format!("probe_len {} not in 4..=32767", params.probe_len));
        }
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
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("encode"),
                entries: &[
                    storage(0, true),  // input
                    storage(1, false), // scratch
                    storage(2, false), // output
                    storage(3, false), // sizes
                    uniform(4),        // params
                    storage(5, false), // chunk_info
                    storage(6, false), // group_buf
                    storage(7, false), // segs
                    storage(8, false), // chain
                ],
            });
        let pack_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("encode pack"),
                entries: &[
                    storage(0, true),  // input
                    storage(1, true),  // output slots
                    storage(2, true),  // payload layout
                    storage(3, false), // packed data section
                    uniform(4),        // params
                ],
            });
        let pipeline_layout = |layout| {
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("encode"),
                    bind_group_layouts: &[Some(layout)],
                    immediate_size: 0,
                })
        };
        let encode_layout = pipeline_layout(&layout);
        let pipeline = |label: &str,
                        layout: &wgpu::PipelineLayout,
                        sources: &[&str],
                        constants: &[(&str, f64)]| {
            let module = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(sources.join("\n").into()),
                });
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants,
                        ..Default::default()
                    },
                    cache: None,
                })
        };
        let wg = ("WG_SIZE", f64::from(params.block));
        let codec = |name: &str, emit: &str, id: u32| CodecPipelines {
            parse: pipeline(
                &format!("{name} parse"),
                &encode_layout,
                &[COMMON_SHADER, PARSE_SEG_SHADER],
                &[("CODEC", f64::from(id))],
            ),
            parse_serial: pipeline(
                &format!("{name} serial parse"),
                &encode_layout,
                &[COMMON_SHADER, PARSE_SHADER],
                &[("CODEC", f64::from(id))],
            ),
            emit: pipeline(
                &format!("{name} emit"),
                &encode_layout,
                &[COMMON_SHADER, EMIT_SHARED, emit],
                &[("WG_SIZE", f64::from(EMIT_WG))],
            ),
        };
        Ok(GpuEncoder {
            filters: crate::filter::FilterKernels::new(ctx),
            params,
            matches: pipeline(
                "encode matches",
                &encode_layout,
                &[COMMON_SHADER, MATCHES_SHADER],
                &[
                    wg,
                    ("HASH_LOG", f64::from(params.hash_log)),
                    ("DEPTH", f64::from(params.depth)),
                ],
            ),
            lz4: codec("lz4", LZ4_EMIT, 0),
            glz: codec("glz", GLZ_EMIT, 1),
            pack: pipeline(
                "encode pack",
                &pipeline_layout(&pack_layout),
                &[PACK_SHADER],
                &[],
            ),
            layout,
            pack_layout,
            max_batch_chunks: None,
            memory_budget: DEFAULT_GPU_MEMORY,
            buffers: Mutex::default(),
        })
    }

    /// Limits every dispatch to at most `chunks` chunks (at least one), below
    /// what the device's binding limit allows. Output is unchanged; this
    /// exercises the multi-batch path on small inputs, or caps GPU memory use.
    pub fn with_max_batch_chunks(mut self, chunks: u64) -> Self {
        self.max_batch_chunks = Some(chunks.max(1));
        self
    }

    /// Caps the GPU memory one batch may use (default [`DEFAULT_GPU_MEMORY`]):
    /// inputs are encoded in batches of at most
    /// `bytes / encoder_bytes_per_chunk` chunks (at least one).
    pub fn with_memory_budget(mut self, bytes: u64) -> Self {
        self.memory_budget = bytes;
        self
    }

    /// Compresses `len` bytes from `reader` into a `.gpcz` file written to
    /// `writer` (from its current position), a batch at a time: the header and
    /// chunk table are reserved first and filled in at the end. Returns the
    /// bytes written.
    pub fn compress_stream<R: Read, W: Write + Seek>(
        &self,
        ctx: &Context,
        reader: &mut R,
        len: u64,
        writer: &mut W,
        options: &GpuCompressOptions,
    ) -> Result<u64, GpuEncodeError> {
        let start = writer.stream_position()?;
        let table_len = table_len(len, options)? as u64;
        // Reserve the header and chunk table; chunk_count is known from `len`.
        std::io::copy(&mut std::io::repeat(0).take(table_len), writer)?;
        let mut sink = WriterSink {
            writer: &mut *writer,
            buffer: Vec::new(),
            written: 0,
        };
        let index = self.compress_batches(
            ctx,
            &mut ReaderSource {
                reader,
                buffer: Vec::new(),
            },
            len,
            &mut sink,
            options,
        )?;
        let data_len = sink.written;
        writer.seek(SeekFrom::Start(start))?;
        writer.write_all(&index.to_bytes())?;
        writer.seek(SeekFrom::Start(start + table_len + data_len))?;
        Ok(table_len + data_len)
    }

    pub fn params(&self) -> EncodeParams {
        self.params
    }

    /// Chunks per dispatch: the device's binding limit, or the test cap.
    fn batch_chunks(&self, ctx: &Context, chunk_size: u32) -> u64 {
        let limit = ctx
            .device_limits()
            .max_storage_buffer_binding_size
            .min(u64::from(u32::MAX) & !3);
        chunks_per_batch(chunk_size, limit).min(self.max_batch_chunks.unwrap_or(u64::MAX))
    }

    /// The buffer cache. A panic while it was held may have left a staging
    /// buffer mapped, so a poisoned cache starts over.
    fn buffers(&self) -> MutexGuard<'_, BufferCache> {
        self.buffers.lock().unwrap_or_else(|poisoned| {
            let mut guard = poisoned.into_inner();
            *guard = BufferCache::default();
            guard
        })
    }

    /// Encodes every `chunk_size` chunk of `input` as a block of
    /// `options.codec`. For tests and debugging.
    pub fn encode_blocks(
        &self,
        ctx: &Context,
        input: &[u8],
        options: &GpuCompressOptions,
    ) -> Result<Vec<EncodedBlock>, GpuEncodeError> {
        let chunk_size = options.chunk_size;
        let encoded = self.encode(ctx, input, options)?;
        Ok(input
            .chunks(chunk_size as usize)
            .enumerate()
            .map(|(i, chunk)| {
                let size = encoded.sizes[i];
                if size as usize >= chunk.len() {
                    EncodedBlock::Incompressible { size }
                } else {
                    EncodedBlock::Compressed(encoded.block(i).to_vec())
                }
            })
            .collect())
    }

    /// Compresses `input` into a `.gpcz` container (LZ4 or GLZ).
    ///
    /// Per batch: encode, read back only the per-chunk sizes, lay out the
    /// payloads on the host, pack them on the GPU into one contiguous data
    /// section and read back exactly that. The header and chunk table are
    /// built here; checksums are computed while the GPU encodes.
    pub fn compress(
        &self,
        ctx: &Context,
        input: &[u8],
        options: &GpuCompressOptions,
    ) -> Result<Vec<u8>, GpuEncodeError> {
        let table_len = table_len(input.len() as u64, options)?;
        let mut sink = VecSink {
            out: vec![0; table_len],
        };
        sink.out.reserve(input.len() + input.len() / 16 + 64);
        let index = self.compress_batches(
            ctx,
            &mut SliceSource {
                data: input,
                pos: 0,
            },
            input.len() as u64,
            &mut sink,
            options,
        )?;
        let mut out = sink.out;
        out[..table_len].copy_from_slice(&index.to_bytes());
        Ok(out)
    }

    /// The batch loop shared by [`compress`](Self::compress) and
    /// [`compress_stream`](Self::compress_stream): encodes `len` bytes from
    /// `source` a batch at a time (bounded by the binding limit and the memory
    /// budget), appending each batch's data section to `sink`, and returns the
    /// header and chunk table.
    fn compress_batches(
        &self,
        ctx: &Context,
        source: &mut impl BatchSource,
        len: u64,
        sink: &mut impl DataSink,
        options: &GpuCompressOptions,
    ) -> Result<format::Index, GpuEncodeError> {
        check_options(options)?;
        let chunk_size = options.chunk_size;
        let chunk_count = format::chunk_count_for(len, chunk_size.max(1));
        let header = format::Header {
            codec: options.codec,
            chunk_size,
            chunk_count: u32::try_from(chunk_count)
                .map_err(|_| GpuEncodeError::TooManyChunks(chunk_count))?,
            total_size: len,
            checksums: options.checksums,
            level: options.level,
        };
        let mut chunks = Vec::with_capacity(chunk_count as usize);
        let mut checksums = Vec::with_capacity(chunk_count as usize);

        let mut cache = self.buffers();
        cache.read_bytes = 0;
        let per_batch = self.batch_chunks_for(ctx, options) as usize * chunk_size as usize;
        let mut data_len = 0u64;
        let mut remaining = len;
        while remaining > 0 {
            let take = (per_batch as u64).min(remaining) as usize;
            remaining -= take as u64;
            let batch = source.next(take)?;
            let out = sink.buffer();
            let before = out.len();
            let (layout, filters) = if options.filters != FilterMode::None {
                if options.checksums {
                    checksums.extend(batch.chunks(chunk_size as usize).map(format::checksum));
                }
                match options.filters {
                    FilterMode::Auto => {
                        self.encode_sampled_batch(ctx, &mut cache, batch, options, out)?
                    }
                    _ => self.encode_filtered_batch(ctx, &mut cache, batch, options, out)?,
                }
            } else {
                let (prepared, sizes) =
                    self.encode_batch(ctx, &mut cache, batch, options, || {
                        if options.checksums {
                            checksums
                                .extend(batch.chunks(chunk_size as usize).map(format::checksum));
                        }
                    })?;
                let layout = layout_payloads(&sizes, batch.len(), chunk_size);
                self.pack(ctx, &mut cache, &prepared, &layout, out)?;
                let filters = vec![format::Filter::None; layout.entries.len()];
                (layout, filters)
            };
            let written = (out.len() - before) as u64;
            let first = chunks.len();
            chunks.extend(
                layout
                    .entries
                    .iter()
                    .enumerate()
                    .map(|(i, &[offset, size])| format::ChunkEntry {
                        comp_offset: data_len + u64::from(offset),
                        comp_size: size & !format::STORED_BIT,
                        stored: size & format::STORED_BIT != 0,
                        uncomp_size: header.uncomp_size_of((first + i) as u32),
                        checksum: 0,
                        filter: filters[i],
                    }),
            );
            data_len += written;
            sink.batch_done()?;
        }
        drop(cache);
        for (chunk, checksum) in chunks.iter_mut().zip(checksums) {
            chunk.checksum = checksum;
        }
        Ok(format::Index { header, chunks })
    }

    /// Chunks per batch: within the binding limit (with filter selection,
    /// shared by every candidate) and the memory budget.
    fn batch_chunks_for(&self, ctx: &Context, options: &GpuCompressOptions) -> u64 {
        let candidates = match options.filters {
            FilterMode::None => 1,
            FilterMode::Auto | FilterMode::Exhaustive => {
                filter_candidates(options.level).len() as u64
            }
        };
        let by_binding = (self.batch_chunks(ctx, options.chunk_size) / candidates).max(1);
        let by_budget =
            (self.memory_budget / encoder_bytes_per_chunk(options, &self.params)).max(1);
        let chunks = by_binding.min(by_budget);
        log::debug!(
            "GPU encoder: {chunks} chunks of {} KiB per batch (budget {} MiB)",
            options.chunk_size >> 10,
            self.memory_budget >> 20
        );
        chunks
    }

    /// Runs the encoder over all chunks, in as many dispatches as the device's
    /// binding limit requires, reading back sizes and whole output slots.
    fn encode(
        &self,
        ctx: &Context,
        input: &[u8],
        options: &GpuCompressOptions,
    ) -> Result<Encoded, GpuEncodeError> {
        check_options(options)?;
        let chunk_size = options.chunk_size;
        let slot = slot_size(chunk_size) as usize;
        let chunk_count = input.len().div_ceil(chunk_size as usize);
        let mut encoded = Encoded {
            sizes: Vec::with_capacity(chunk_count),
            slots: Vec::with_capacity(chunk_count * slot),
            slot_size: slot,
        };
        let mut cache = self.buffers();
        let per_batch = self.batch_chunks(ctx, chunk_size) as usize * chunk_size as usize;
        for batch in input.chunks(per_batch) {
            let (prepared, sizes) = self.encode_batch(ctx, &mut cache, batch, options, || ())?;
            encoded.sizes.extend(sizes);
            encoded
                .slots
                .extend(ctx.read_buffer(&prepared.output_buf, prepared.output_size)?);
        }
        Ok(encoded)
    }

    /// Encodes one batch and reads back its per-chunk sizes, running
    /// `while_gpu_runs` on the host between submitting and waiting.
    fn encode_batch(
        &self,
        ctx: &Context,
        cache: &mut BufferCache,
        batch: &[u8],
        options: &GpuCompressOptions,
        while_gpu_runs: impl FnOnce(),
    ) -> Result<(PreparedEncode, Vec<u32>), GpuEncodeError> {
        let prepared = self.prepare_batch(ctx, cache, batch, options)?;
        let sizes_len = u64::from(prepared.chunks) * 4;
        let sizes_read = BufferCache::ensure(
            &mut cache.sizes_read,
            &mut cache.allocations,
            &ctx.device,
            "encode sizes readback",
            sizes_len,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.record_encode(ctx, &mut encoder, &prepared, None);
        encoder.copy_buffer_to_buffer(&prepared.sizes_buf, 0, &sizes_read, 0, sizes_len);
        ctx.queue.submit([encoder.finish()]);
        while_gpu_runs();
        let mut sizes = Vec::new();
        read_mapped(ctx, &sizes_read, sizes_len, |bytes| {
            sizes = bytemuck::pod_collect_to_vec(bytes);
        })?;
        cache.read_bytes += sizes_len;
        Ok((prepared, sizes))
    }

    /// Packs an encoded batch's payloads (`layout`) into one contiguous data
    /// section on the GPU and appends it to `out`.
    fn pack(
        &self,
        ctx: &Context,
        cache: &mut BufferCache,
        p: &PreparedEncode,
        layout: &PayloadLayout,
        out: &mut Vec<u8>,
    ) -> Result<(), GpuEncodeError> {
        if layout.len == 0 {
            return Ok(());
        }
        let from = PackSource {
            input: (&p.input_buf, p.input_size),
            slots: (&p.output_buf, p.output_size),
        };
        self.pack_pass(ctx, cache, p, from, &layout.entries, layout.len);
        self.read_packed(ctx, cache, layout.len, out)
    }

    /// Submits one pack pass: copies the payloads of `entries` (those without
    /// [`SKIP_BIT`]) from `slots` (or `input`, for stored chunks) into the
    /// batch's packed data section.
    fn pack_pass(
        &self,
        ctx: &Context,
        cache: &mut BufferCache,
        p: &PreparedEncode,
        from: PackSource,
        entries: &[[u32; 2]],
        len: u32,
    ) {
        let PackSource { input, slots } = from;
        let device = &ctx.device;
        let entries_len = u64::from(p.chunks) * 8;
        let entries_buf = BufferCache::ensure(
            &mut cache.entries,
            &mut cache.allocations,
            device,
            "encode payload layout",
            entries_len,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        // Ordered before this pass's submission (and after earlier ones).
        ctx.queue
            .write_buffer(&entries_buf, 0, bytemuck::cast_slice(entries));
        let packed = BufferCache::ensure(
            &mut cache.packed,
            &mut cache.allocations,
            device,
            "encode packed data",
            u64::from(len),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("encode pack"),
            layout: &self.pack_layout,
            entries: &[
                (0, binding(input.0, input.1)),
                (1, binding(slots.0, slots.1)),
                (2, binding(&entries_buf, entries_len)),
                (3, binding(&packed, u64::from(len))),
                (4, binding(&p.params_buf, PARAMS_SIZE)),
            ]
            .map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource }),
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("encode pack"),
                timestamp_writes: None,
            });
            let max = ctx.device_limits().max_compute_workgroups_per_dimension;
            let (x, y) = dispatch_grid(p.chunks, max);
            pass.set_pipeline(&self.pack);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        ctx.queue.submit([encoder.finish()]);
    }

    /// Reads back the first `len` bytes of the packed data section into `out`.
    fn read_packed(
        &self,
        ctx: &Context,
        cache: &mut BufferCache,
        len: u32,
        out: &mut Vec<u8>,
    ) -> Result<(), GpuEncodeError> {
        let len = u64::from(len);
        let device = &ctx.device;
        let packed = BufferCache::ensure(
            &mut cache.packed,
            &mut cache.allocations,
            device,
            "encode packed data",
            len,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let data_read = BufferCache::ensure(
            &mut cache.data_read,
            &mut cache.allocations,
            device,
            "encode data readback",
            len,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&packed, 0, &data_read, 0, len);
        ctx.queue.submit([encoder.finish()]);
        read_mapped(ctx, &data_read, len, |bytes| out.extend_from_slice(bytes))?;
        cache.read_bytes += len;
        Ok(())
    }

    /// Encodes `input` (one batch) under every filter candidate of
    /// `options.level`: the unfiltered one in the cached buffers, the others
    /// from a GPU-filtered copy of the input into their own output slots.
    /// Returns the prepared batch, each candidate's (filtered input, slots)
    /// (`None` = the cached ones) and each candidate's per-chunk block sizes.
    #[allow(clippy::type_complexity)]
    fn run_candidates(
        &self,
        ctx: &Context,
        cache: &mut BufferCache,
        input: &[u8],
        options: &GpuCompressOptions,
    ) -> Result<
        (
            PreparedEncode,
            Vec<Option<(wgpu::Buffer, wgpu::Buffer)>>,
            Vec<Vec<u32>>,
        ),
        GpuEncodeError,
    > {
        use crate::filter::{Direction, FilterJob};
        let chunk_size = options.chunk_size as usize;
        let p = self.prepare_batch(ctx, cache, input, options)?;
        let mut runs = Vec::new();
        let mut sizes = Vec::new();
        for &filter in filter_candidates(options.level) {
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let run = if filter == format::Filter::None {
                self.record_encode(ctx, &mut encoder, &p, None);
                None
            } else {
                let filtered = self.candidate_buffer(ctx, "encode filtered input", p.input_size);
                let output = self.candidate_buffer(ctx, "encode candidate output", p.output_size);
                let jobs: Vec<FilterJob> = input
                    .chunks(chunk_size)
                    .enumerate()
                    .map(|(i, chunk)| FilterJob {
                        filter,
                        src_offset: (i * chunk_size) as u32,
                        dst_offset: (i * chunk_size) as u32,
                        len: chunk.len() as u32,
                    })
                    .collect();
                let pass = self.filters.prepare(
                    ctx,
                    Direction::Forward,
                    &p.input_buf,
                    &filtered,
                    &jobs,
                )?;
                self.filters.record(ctx, &mut encoder, &pass, None);
                let bind_group = self.bind_candidate(ctx, &p, &filtered, &output);
                self.record_encode_with(ctx, &mut encoder, &p, &bind_group, &output, None);
                Some((filtered, output))
            };
            ctx.queue.submit([encoder.finish()]);
            sizes.push(bytemuck::pod_collect_to_vec(
                &ctx.read_buffer(&p.sizes_buf, u64::from(p.chunks) * 4)?,
            ));
            runs.push(run);
        }
        Ok((p, runs, sizes))
    }

    fn candidate_buffer(&self, ctx: &Context, label: &str, size: u64) -> wgpu::Buffer {
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// `FilterMode::Auto`: candidates compete on each chunk's leading sample
    /// (encoded as chunks of [`sample_len`]); the batch is then filtered per
    /// chunk with its winner and encoded once. Same rule as the CPU's `Auto`.
    fn encode_sampled_batch(
        &self,
        ctx: &Context,
        cache: &mut BufferCache,
        batch: &[u8],
        options: &GpuCompressOptions,
        out: &mut Vec<u8>,
    ) -> Result<(PayloadLayout, Vec<format::Filter>), GpuEncodeError> {
        use crate::filter::{Direction, FilterJob};
        let candidates = filter_candidates(options.level);
        let chunk_size = options.chunk_size as usize;
        let sample = sample_len(options.chunk_size) as usize;
        let samples: Vec<u8> = batch
            .chunks(chunk_size)
            .flat_map(|chunk| &chunk[..sample.min(chunk.len())])
            .copied()
            .collect();
        let sample_options = GpuCompressOptions {
            chunk_size: sample as u32,
            ..*options
        };
        let (_, _, sizes) = self.run_candidates(ctx, cache, &samples, &sample_options)?;
        let winners: Vec<format::Filter> = (0..sizes[0].len())
            .map(|c| {
                let mut k_best = 0;
                for k in 1..candidates.len() {
                    if sizes[k][c] < sizes[k_best][c] {
                        k_best = k;
                    }
                }
                candidates[k_best]
            })
            .collect();

        // The whole batch, each chunk filtered with its winner, encoded once.
        let p = self.prepare_batch(ctx, cache, batch, options)?;
        let filtered = self.candidate_buffer(ctx, "encode filtered input", p.input_size);
        let jobs: Vec<FilterJob> = batch
            .chunks(chunk_size)
            .zip(&winners)
            .enumerate()
            .map(|(i, (chunk, &filter))| FilterJob {
                filter,
                src_offset: (i * chunk_size) as u32,
                dst_offset: (i * chunk_size) as u32,
                len: chunk.len() as u32,
            })
            .collect();
        let pass = self
            .filters
            .prepare(ctx, Direction::Forward, &p.input_buf, &filtered, &jobs)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        // Unfiltered chunks (skipped by the filter pass) keep the input bytes.
        encoder.copy_buffer_to_buffer(&p.input_buf, 0, &filtered, 0, p.input_size);
        if !pass.is_empty() {
            self.filters.record(ctx, &mut encoder, &pass, None);
        }
        let bind_group = self.bind_candidate(ctx, &p, &filtered, &p.output_buf);
        self.record_encode_with(ctx, &mut encoder, &p, &bind_group, &p.output_buf, None);
        ctx.queue.submit([encoder.finish()]);
        let sizes: Vec<u32> =
            bytemuck::pod_collect_to_vec(&ctx.read_buffer(&p.sizes_buf, u64::from(p.chunks) * 4)?);
        let layout = layout_payloads(&sizes, batch.len(), options.chunk_size);
        // Stored chunks come from the unfiltered input; the rest from the slots.
        self.pack(ctx, cache, &p, &layout, out)?;
        let filters = layout
            .entries
            .iter()
            .zip(winners)
            .map(|(&[_, len], filter)| {
                if len & format::STORED_BIT != 0 {
                    format::Filter::None
                } else {
                    filter
                }
            })
            .collect();
        Ok((layout, filters))
    }

    /// `FilterMode::Exhaustive`: encodes one batch under every candidate,
    /// keeps each chunk's smallest block (ties to the earlier candidate; a
    /// chunk whose best block doesn't shrink it is stored raw, unfiltered) and
    /// packs the winners into `out`. Same rule as the CPU's `Exhaustive`.
    fn encode_filtered_batch(
        &self,
        ctx: &Context,
        cache: &mut BufferCache,
        batch: &[u8],
        options: &GpuCompressOptions,
        out: &mut Vec<u8>,
    ) -> Result<(PayloadLayout, Vec<format::Filter>), GpuEncodeError> {
        let candidates = filter_candidates(options.level);
        let chunk_lens: Vec<u32> = batch
            .chunks(options.chunk_size as usize)
            .map(|c| c.len() as u32)
            .collect();
        let (p, runs, sizes) = self.run_candidates(ctx, cache, batch, options)?;

        // Choose: strictly smallest block, earlier candidate on ties.
        let mut winner = vec![0usize; chunk_lens.len()];
        let mut best = vec![0u32; chunk_lens.len()];
        for c in 0..chunk_lens.len() {
            let mut k_best = 0;
            for k in 1..candidates.len() {
                if sizes[k][c] < sizes[k_best][c] {
                    k_best = k;
                }
            }
            winner[c] = k_best;
            best[c] = sizes[k_best][c];
        }
        let layout = layout_payloads(&best, batch.len(), options.chunk_size);
        let stored = |c: usize| layout.entries[c][1] & format::STORED_BIT != 0;

        // Pack: one pass per candidate that won anything; stored chunks (raw,
        // unfiltered input) go with the unfiltered candidate's pass.
        if layout.len > 0 {
            for (k, run) in runs.iter().enumerate() {
                let mine = |c: usize| if stored(c) { k == 0 } else { winner[c] == k };
                if !(0..chunk_lens.len()).any(mine) {
                    continue;
                }
                let entries: Vec<[u32; 2]> = layout
                    .entries
                    .iter()
                    .enumerate()
                    .map(|(c, &[offset, len])| [offset, if mine(c) { len } else { SKIP_BIT }])
                    .collect();
                let from = match run {
                    None => PackSource {
                        input: (&p.input_buf, p.input_size),
                        slots: (&p.output_buf, p.output_size),
                    },
                    Some((filtered, output)) => PackSource {
                        input: (filtered, p.input_size),
                        slots: (output, p.output_size),
                    },
                };
                self.pack_pass(ctx, cache, &p, from, &entries, layout.len);
            }
            self.read_packed(ctx, cache, layout.len, out)?;
        }
        let filters = (0..chunk_lens.len())
            .map(|c| {
                if stored(c) {
                    format::Filter::None
                } else {
                    candidates[winner[c]]
                }
            })
            .collect();
        Ok((layout, filters))
    }

    /// The encode bind group of `p`, with `input` and `output` replaced.
    fn bind_candidate(
        &self,
        ctx: &Context,
        p: &PreparedEncode,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("encode (filter candidate)"),
            layout: &self.layout,
            entries: &[
                (0, binding(input, p.input_size)),
                (1, binding(&p.scratch.0, p.scratch.1)),
                (2, binding(output, p.output_size)),
                (3, binding(&p.sizes_buf, u64::from(p.chunks) * 4)),
                (4, binding(&p.params_buf, PARAMS_SIZE)),
                (5, binding(&p.info.0, p.info.1)),
                (6, binding(&p.groups.0, p.groups.1)),
                (7, binding(&p.segs.0, p.segs.1)),
                (8, binding(&p.chain.0, p.chain.1)),
            ]
            .map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource }),
        })
    }

    /// Uploads `input` as a single batch, ready to [`dispatch`](Self::dispatch),
    /// so benchmarks can time the kernel apart from transfers. Fails if the
    /// input needs more than one batch on this device. The buffers are its
    /// own, not the encoder's reusable ones.
    pub fn prepare(
        &self,
        ctx: &Context,
        input: &[u8],
        options: &GpuCompressOptions,
    ) -> Result<PreparedEncode, GpuEncodeError> {
        check_options(options)?;
        let chunk_size = options.chunk_size;
        let chunks = (input.len() as u64).div_ceil(u64::from(chunk_size)).max(1);
        if chunks > self.batch_chunks(ctx, chunk_size) {
            return Err(GpuEncodeError::Unsupported(format!(
                "{chunks} chunks don't fit one batch on this device"
            )));
        }
        self.prepare_batch(ctx, &mut BufferCache::default(), input, options)
    }

    /// Uploads `batch` and its parameters into `cache`'s buffers (growing
    /// them if needed) and binds them.
    fn prepare_batch(
        &self,
        ctx: &Context,
        cache: &mut BufferCache,
        batch: &[u8],
        options: &GpuCompressOptions,
    ) -> Result<PreparedEncode, GpuEncodeError> {
        let device = &ctx.device;
        let chunk_size = options.chunk_size;
        let chunks = (batch.len() as u64).div_ceil(u64::from(chunk_size)).max(1);
        let slot = slot_size(chunk_size);
        let c = &mut *cache;
        let mut ensure = |slot: &mut Option<wgpu::Buffer>, label, size, usage| {
            BufferCache::ensure(slot, &mut c.allocations, device, label, size, usage)
        };

        // The input padded with zeros to whole words (plus one), so unaligned
        // reads stay in bounds. Written into a reusable mapped staging buffer
        // (`Queue::write_buffer` would allocate, and page-fault, a fresh one
        // per call), then copied on the GPU.
        let input_size = format::pad4(batch.len() as u64) + 4;
        let input_buf = ensure(
            &mut c.input,
            "encode input",
            input_size,
            // COPY_SRC: sampled filter selection copies it into a filtered buffer.
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );
        let upload = ensure(
            &mut c.upload,
            "encode input upload",
            input_size,
            BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
        );
        map(ctx, &upload, wgpu::MapMode::Write, input_size)?;
        let written = upload.get_mapped_range_mut(..input_size).map(|mut view| {
            let whole = batch.len() & !3;
            view.slice(..whole).copy_from_slice(&batch[..whole]);
            let mut tail = [0u8; 8];
            tail[..batch.len() - whole].copy_from_slice(&batch[whole..]);
            view.slice(whole..)
                .copy_from_slice(&tail[..input_size as usize - whole]);
        });
        upload.unmap();
        written.map_err(|e| GpuError::Readback(e.to_string()))?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&upload, 0, &input_buf, 0, input_size);
        ctx.queue.submit([encoder.finish()]);

        let scratch_size = chunks * u64::from(chunk_size) * 4;
        let scratch_buf = ensure(
            &mut c.scratch,
            "encode scratch",
            scratch_size,
            BufferUsages::STORAGE,
        );
        let output_size = chunks * u64::from(slot);
        let output_buf = ensure(
            &mut c.output,
            "encode output",
            output_size,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        let sizes_buf = ensure(
            &mut c.sizes,
            "encode sizes",
            chunks * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let groups = options.independent_groups.unwrap_or(0);
        let params = [
            chunk_size,
            batch.len() as u32,
            slot,
            self.params.probe_len,
            groups,
            u32::from(self.params.lazy),
            0,
            0,
        ];
        let params_buf = ensure(
            &mut c.params,
            "encode params",
            PARAMS_SIZE,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );
        ctx.queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));
        let info_size = chunks * 16;
        let info_buf = ensure(
            &mut c.info,
            "encode chunk info",
            info_size,
            BufferUsages::STORAGE,
        );
        // Only the GLZ parse with dependency elimination uses the group lists.
        let group_size = if groups > 0 {
            chunks * u64::from(MAX_GROUP) * 8
        } else {
            8
        };
        let group_buf = ensure(
            &mut c.groups,
            "encode group lists",
            group_size,
            BufferUsages::STORAGE,
        );
        // Per chunk, where each parse segment's sequences start (see emit).
        let segs_size = chunks * u64::from(PARSE_SEGMENTS) * 4;
        let segs_buf = ensure(
            &mut c.segs,
            "encode segment table",
            segs_size,
            BufferUsages::STORAGE,
        );
        // Hash-chain links, one u32 per input byte, only with depth > 1.
        let chain_size = if self.params.depth > 1 {
            chunks * u64::from(chunk_size) * 4
        } else {
            4
        };
        let chain_buf = ensure(
            &mut c.chain,
            "encode chain links",
            chain_size,
            BufferUsages::STORAGE,
        );
        // Bind exactly the sizes this batch needs, as if the buffers were fresh.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("encode"),
            layout: &self.layout,
            entries: &[
                (0, binding(&input_buf, input_size)),
                (1, binding(&scratch_buf, scratch_size)),
                (2, binding(&output_buf, output_size)),
                (3, binding(&sizes_buf, chunks * 4)),
                (4, binding(&params_buf, PARAMS_SIZE)),
                (5, binding(&info_buf, info_size)),
                (6, binding(&group_buf, group_size)),
                (7, binding(&segs_buf, segs_size)),
                (8, binding(&chain_buf, chain_size)),
            ]
            .map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource }),
        });
        Ok(PreparedEncode {
            scratch: (scratch_buf.clone(), scratch_size),
            info: (info_buf.clone(), info_size),
            groups: (group_buf.clone(), group_size),
            segs: (segs_buf.clone(), segs_size),
            chain: (chain_buf.clone(), chain_size),
            serial_parse: groups > 0,
            codec: options.codec,
            chunks: chunks as u32,
            input_buf,
            input_size,
            output_buf,
            output_size,
            sizes_buf,
            params_buf,
            bind_group,
        })
    }

    fn kernels(&self, codec: format::Codec) -> &CodecPipelines {
        match codec {
            format::Codec::Glz => &self.glz,
            format::Codec::Lz4 | format::Codec::Stored => &self.lz4,
        }
    }

    /// Clears the output slots and submits the encode.
    pub fn dispatch(&self, ctx: &Context, p: &PreparedEncode, timer: Option<&crate::GpuTimer>) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.record_encode(ctx, &mut encoder, p, timer);
        if let Some(timer) = timer {
            timer.resolve(&mut encoder);
        }
        ctx.queue.submit([encoder.finish()]);
    }

    /// Records clearing the output slots and the three encode kernels.
    fn record_encode(
        &self,
        ctx: &Context,
        encoder: &mut wgpu::CommandEncoder,
        p: &PreparedEncode,
        timer: Option<&crate::GpuTimer>,
    ) {
        self.record_encode_with(ctx, encoder, p, &p.bind_group, &p.output_buf, timer);
    }

    /// [`record_encode`](Self::record_encode) with another bind group and
    /// output buffer (a filter candidate's).
    fn record_encode_with(
        &self,
        ctx: &Context,
        encoder: &mut wgpu::CommandEncoder,
        p: &PreparedEncode,
        bind_group: &wgpu::BindGroup,
        output: &wgpu::Buffer,
        timer: Option<&crate::GpuTimer>,
    ) {
        encoder.clear_buffer(output, 0, Some(p.output_size));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("encode"),
            timestamp_writes: timer.map(crate::GpuTimer::pass_writes),
        });
        // Three kernels; wgpu orders their storage accesses.
        let max = ctx.device_limits().max_compute_workgroups_per_dimension;
        let per_chunk = dispatch_grid(p.chunks, max);
        let kernels = self.kernels(p.codec);
        let parse = if p.serial_parse {
            (
                &kernels.parse_serial,
                dispatch_grid(p.chunks.div_ceil(PARSE_WG), max),
            )
        } else {
            (&kernels.parse, per_chunk)
        };
        pass.set_bind_group(0, bind_group, &[]);
        for (pipeline, (x, y)) in [
            (&self.matches, per_chunk),
            parse,
            (&kernels.emit, per_chunk),
        ] {
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(x, y, 1);
        }
    }
}

/// GPU buffers for one encode batch, ready to [`dispatch`](GpuEncoder::dispatch).
pub struct PreparedEncode {
    codec: format::Codec,
    chunks: u32,
    input_buf: wgpu::Buffer,
    /// Bytes of each buffer this batch uses (cached buffers may be larger).
    input_size: u64,
    output_buf: wgpu::Buffer,
    output_size: u64,
    sizes_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// The remaining bindings, kept so a filter candidate can be bound with
    /// its own input and output (see [`GpuEncoder::bind_candidate`]).
    scratch: (wgpu::Buffer, u64),
    info: (wgpu::Buffer, u64),
    groups: (wgpu::Buffer, u64),
    segs: (wgpu::Buffer, u64),
    chain: (wgpu::Buffer, u64),
    /// GLZ with dependency elimination: the one-invocation-per-chunk parse.
    serial_parse: bool,
}

/// Header plus chunk-table bytes for `len` input bytes.
fn table_len(len: u64, options: &GpuCompressOptions) -> Result<usize, GpuEncodeError> {
    let chunks = format::chunk_count_for(len, options.chunk_size.max(1));
    u32::try_from(chunks).map_err(|_| GpuEncodeError::TooManyChunks(chunks))?;
    Ok(format::HEADER_SIZE + format::ENTRY_SIZE * chunks as usize)
}

/// Where the batch loop gets its input from.
trait BatchSource {
    /// The next `n` input bytes.
    fn next(&mut self, n: usize) -> Result<&[u8], GpuEncodeError>;
}

struct SliceSource<'a> {
    data: &'a [u8],
    pos: usize,
}

impl BatchSource for SliceSource<'_> {
    fn next(&mut self, n: usize) -> Result<&[u8], GpuEncodeError> {
        let batch = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(batch)
    }
}

struct ReaderSource<'a, R> {
    reader: &'a mut R,
    buffer: Vec<u8>,
}

impl<R: Read> BatchSource for ReaderSource<'_, R> {
    fn next(&mut self, n: usize) -> Result<&[u8], GpuEncodeError> {
        self.buffer.resize(n, 0);
        self.reader.read_exact(&mut self.buffer)?;
        Ok(&self.buffer)
    }
}

/// Where the batch loop puts each batch's data section.
trait DataSink {
    /// The buffer the batch's data section is appended to.
    fn buffer(&mut self) -> &mut Vec<u8>;
    /// Called after each batch.
    fn batch_done(&mut self) -> Result<(), GpuEncodeError>;
}

/// The whole file in memory (header and table reserved at the front).
struct VecSink {
    out: Vec<u8>,
}

impl DataSink for VecSink {
    fn buffer(&mut self) -> &mut Vec<u8> {
        &mut self.out
    }

    fn batch_done(&mut self) -> Result<(), GpuEncodeError> {
        Ok(())
    }
}

/// Writes each batch's data section out as soon as it's packed.
struct WriterSink<'a, W> {
    writer: &'a mut W,
    buffer: Vec<u8>,
    written: u64,
}

impl<W: Write> DataSink for WriterSink<'_, W> {
    fn buffer(&mut self) -> &mut Vec<u8> {
        &mut self.buffer
    }

    fn batch_done(&mut self) -> Result<(), GpuEncodeError> {
        self.writer.write_all(&self.buffer)?;
        self.written += self.buffer.len() as u64;
        self.buffer.clear();
        Ok(())
    }
}

/// What a pack pass copies from: the (filtered) input for stored chunks and
/// the encode output slots for compressed ones, each with its bound size.
struct PackSource<'a> {
    input: (&'a wgpu::Buffer, u64),
    slots: (&'a wgpu::Buffer, u64),
}

/// Where each chunk of a batch goes in that batch's data section, as uploaded
/// to the pack kernel: `[offset, len | STORED_BIT]` per chunk, with `offset`
/// in bytes from the start of the batch's data section.
#[derive(Debug, PartialEq, Eq)]
struct PayloadLayout {
    entries: Vec<[u32; 2]>,
    /// Bytes in the batch's data section (payloads padded to 4 bytes).
    len: u32,
}

/// Lays out a batch's payloads from the kernel's `sizes`: a chunk is stored
/// raw when its block would be no smaller than the chunk (the CPU encoders'
/// rule), and every payload is padded to 4 bytes.
fn layout_payloads(sizes: &[u32], batch_len: usize, chunk_size: u32) -> PayloadLayout {
    let mut len = 0u32;
    let entries = sizes
        .iter()
        .enumerate()
        .map(|(i, &size)| {
            let chunk_len = (batch_len - i * chunk_size as usize).min(chunk_size as usize) as u32;
            let entry = if size >= chunk_len {
                [len, chunk_len | format::STORED_BIT]
            } else {
                [len, size]
            };
            len += (entry[1] & !format::STORED_BIT).next_multiple_of(4);
            entry
        })
        .collect();
    PayloadLayout { entries, len }
}

fn check_options(options: &GpuCompressOptions) -> Result<(), GpuEncodeError> {
    check_chunk_size(options.chunk_size)?;
    if options.codec == format::Codec::Stored {
        return Err(GpuEncodeError::Unsupported(
            "the GPU encodes LZ4 or GLZ".into(),
        ));
    }
    match options.independent_groups {
        Some(_) if options.codec != format::Codec::Glz => Err(GpuEncodeError::Unsupported(
            "independent groups need the GLZ codec".into(),
        )),
        Some(g) if !(1..=MAX_GROUP).contains(&g) => Err(GpuEncodeError::Unsupported(format!(
            "independent groups of {g}; must be 1..={MAX_GROUP}"
        ))),
        _ => Ok(()),
    }
}

fn check_chunk_size(chunk_size: u32) -> Result<(), format::FormatError> {
    if chunk_size.is_power_of_two()
        && (format::MIN_CHUNK_SIZE..=format::MAX_CHUNK_SIZE).contains(&chunk_size)
    {
        Ok(())
    } else {
        Err(format::FormatError::BadChunkSize(chunk_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GPU context, or `None` when the machine has no adapter.
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

    fn pattern(n: usize, seed: u8) -> Vec<u8> {
        (0..n)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    fn text(n: usize) -> Vec<u8> {
        b"the quick brown fox jumps over the lazy dog. "
            .iter()
            .copied()
            .cycle()
            .take(n)
            .collect()
    }

    #[test]
    fn pack_copies_blocks_and_raw_chunks_into_padded_payloads() {
        let Some(ctx) = gpu() else { return };
        let encoder = GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
        let options = GpuCompressOptions {
            chunk_size: 4096,
            ..GpuCompressOptions::default()
        };
        let input = pattern(2 * 4096 + 10, 1);
        let mut cache = BufferCache::default();
        let p = encoder
            .prepare_batch(&ctx, &mut cache, &input, &options)
            .unwrap();
        // Fake slots, with junk past each block that must not be copied.
        let slot = slot_size(4096) as usize;
        let slots = pattern(3 * slot, 7);
        ctx.queue.write_buffer(&p.output_buf, 0, &slots);
        // Chunk 0: a 5-byte block; chunk 1: stored; chunk 2: a 3-byte block.
        let layout = layout_payloads(&[5, 4096, 3], input.len(), 4096);
        let mut out = vec![0xEE; 3];
        encoder
            .pack(&ctx, &mut cache, &p, &layout, &mut out)
            .unwrap();
        let expected = [
            &[0xEE; 3][..],
            &slots[..5],
            &[0; 3],
            &input[4096..8192],
            &slots[2 * slot..2 * slot + 3],
            &[0],
        ]
        .concat();
        assert_eq!(out, expected);
    }

    #[test]
    fn compress_reads_back_only_the_sizes_and_the_data_section() {
        let Some(ctx) = gpu() else { return };
        let encoder = GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
        let options = GpuCompressOptions {
            chunk_size: 4096,
            ..GpuCompressOptions::default()
        };
        let input = [text(40_000), pattern(9_000, 3)].concat();
        let file = encoder.compress(&ctx, &input, &options).unwrap();
        let chunks = input.len().div_ceil(4096) as u64;
        let data = file.len() as u64 - (format::HEADER_SIZE as u64 + 24 * chunks);
        assert!(data < input.len() as u64 / 2, "text should compress");
        assert_eq!(encoder.buffers().read_bytes, 4 * chunks + data);
    }

    #[test]
    fn repeated_compress_calls_reuse_buffers() {
        let Some(ctx) = gpu() else { return };
        let encoder = GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
        let options = GpuCompressOptions {
            chunk_size: 4096,
            ..GpuCompressOptions::default()
        };
        let big = text(100_000);
        let first = encoder.compress(&ctx, &big, &options).unwrap();
        let allocations = encoder.buffers().allocations;
        assert!(allocations > 0);
        for input in [&big[..], &big[..50_000], &big[..1]] {
            encoder.compress(&ctx, input, &options).unwrap();
        }
        assert_eq!(encoder.compress(&ctx, &big, &options).unwrap(), first);
        assert_eq!(encoder.buffers().allocations, allocations);
    }

    #[test]
    fn payloads_are_laid_out_back_to_back_in_whole_words() {
        // Chunks of 4096 bytes, the last one 10 bytes long.
        let layout = layout_payloads(&[100, 4096, 7, 12], 3 * 4096 + 10, 4096);
        let stored = format::STORED_BIT;
        assert_eq!(
            layout,
            PayloadLayout {
                entries: vec![
                    [0, 100],             // compressed
                    [100, 4096 | stored], // size == chunk: stored raw
                    [4196, 7],            // compressed, padded to 8
                    [4204, 10 | stored],  // last chunk: 12 >= 10, stored raw
                ],
                len: 4216,
            }
        );
    }

    #[test]
    fn slots_hold_the_lz4_worst_case_in_whole_words() {
        assert_eq!(slot_size(4096), 4096 + 16 + 16);
        assert_eq!(slot_size(65_536), 65_536 + 257 + 16 + 3);
        assert_eq!(slot_size(65_536) % 4, 0);
    }

    #[test]
    fn workgroup_memory_is_the_larger_of_the_table_and_the_emit_scans() {
        // Match finding holds the hash table; emit holds two scans (u32 and
        // vec2 per invocation), sequence addresses, counters and the segment
        // table; the parse
        // holds a few words per segment, less than emit.
        let p = EncodeParams::default();
        assert_eq!(workgroup_bytes(&p), 4 * 4096);
        // The emit kernel's workgroup is EMIT_WG whatever the match block.
        let small_table = EncodeParams {
            hash_log: 7,
            block: 256,
            ..p
        };
        assert_eq!(
            workgroup_bytes(&small_table),
            16 * EMIT_WG + 16 + 4 * PARSE_SEGMENTS
        );
    }

    #[test]
    fn encoder_memory_per_chunk_counts_every_buffer() {
        let c = 65_536u64;
        let base = GpuCompressOptions::default();
        // input, upload staging, scratch (4×), packed, packed readback = 8c,
        // plus the output slot, 32 bytes (sizes + readback, info, entry) and
        // the parse's segment table (a u32 per segment).
        let plain = 8 * c + u64::from(slot_size(65_536)) + 32 + 4 * u64::from(PARSE_SEGMENTS);
        assert_eq!(
            encoder_bytes_per_chunk(&base, &EncodeParams::default()),
            plain
        );
        // Hash chains keep a link (u32) per input byte.
        let chained = EncodeParams {
            depth: 4,
            ..EncodeParams::default()
        };
        assert_eq!(encoder_bytes_per_chunk(&base, &chained), plain + 4 * c);
        let groups = GpuCompressOptions {
            codec: format::Codec::Glz,
            independent_groups: Some(8),
            ..base
        };
        assert_eq!(
            encoder_bytes_per_chunk(&groups, &EncodeParams::default()),
            plain + 8 * u64::from(MAX_GROUP)
        );
        let auto = GpuCompressOptions {
            filters: FilterMode::Auto,
            ..base
        };
        // One filtered copy of the input, plus each extra candidate's filtered
        // sample and sample slot.
        let sample = u64::from(sample_len(65_536));
        let extra = filter_candidates(1).len() as u64 - 1;
        assert_eq!(
            encoder_bytes_per_chunk(&auto, &EncodeParams::default()),
            plain + c + extra * (sample + u64::from(slot_size(sample as u32)))
        );
        let exhaustive = GpuCompressOptions {
            filters: FilterMode::Exhaustive,
            ..base
        };
        assert_eq!(
            encoder_bytes_per_chunk(&exhaustive, &EncodeParams::default()),
            plain + extra * (c + u64::from(slot_size(65_536)))
        );
    }

    #[test]
    fn the_default_budget_encodes_256_mib_in_one_batch() {
        // Each extra batch costs about one whole parse kernel (DECISIONS.md, M8),
        // so benchmark-sized inputs should stay in a single batch.
        for filters in [FilterMode::None, FilterMode::Auto] {
            let options = GpuCompressOptions {
                filters,
                ..Default::default()
            };
            let chunks =
                DEFAULT_GPU_MEMORY / encoder_bytes_per_chunk(&options, &EncodeParams::default());
            assert!(
                chunks * u64::from(options.chunk_size) >= 256 << 20,
                "{filters:?}: {chunks} chunks per batch"
            );
        }
    }

    #[test]
    fn groups_are_capped_at_the_glz_decode_block() {
        assert_eq!(MAX_GROUP, 64);
    }

    #[test]
    fn batches_are_bounded_by_the_largest_buffer() {
        // Match scratch is 4 bytes per input byte: the binding limit / (4 × chunk).
        assert_eq!(chunks_per_batch(65_536, 1 << 30), 4096);
        assert_eq!(chunks_per_batch(65_536, 4 << 20), 16);
        // Always at least one chunk.
        assert_eq!(chunks_per_batch(1 << 20, 1 << 20), 1);
    }
}
