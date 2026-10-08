//! GPU LZ4 compression (M3): one workgroup per chunk, running the same
//! algorithm as `cpu::lz4::encode` so outputs are byte-identical.

use wgpu::util::DeviceExt as _;
use wgpu::BufferUsages;

use crate::{dispatch_grid, Context, GpuError};

const COMMON_SHADER: &str = include_str!("../../shaders/encode_common.wgsl");
const MATCHES_SHADER: &str = include_str!("../../shaders/encode_matches.wgsl");
const PARSE_SHADER: &str = include_str!("../../shaders/encode_parse.wgsl");
const EMIT_SHARED: &str = include_str!("../../shaders/encode_emit_shared.wgsl");
const LZ4_EMIT: &str = include_str!("../../shaders/lz4_emit.wgsl");
const GLZ_EMIT: &str = include_str!("../../shaders/glz_emit.wgsl");
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
}

impl Default for EncodeParams {
    fn default() -> Self {
        EncodeParams {
            block: 64,
            hash_log: 12,
            probe_len: 16,
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
}

impl Default for GpuCompressOptions {
    fn default() -> Self {
        GpuCompressOptions {
            codec: format::Codec::Lz4,
            chunk_size: format::DEFAULT_CHUNK_SIZE,
            checksums: false,
            level: 1,
            independent_groups: None,
        }
    }
}

/// Largest dependency-elimination group the GPU parse supports. The GLZ
/// decoder resolves 64 sequences per step, so larger groups can't help it.
pub const MAX_GROUP: u32 = 64;

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

/// Workgroup memory the encoder needs: the larger of the match-finding
/// kernel's hash table and the emit kernel's scans (the parse uses none).
pub fn workgroup_bytes(params: &EncodeParams) -> u32 {
    (4 << params.hash_log).max(12 * params.block + 16)
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
}

/// The codec-specific kernels: parse (size accounting) and emit.
struct CodecPipelines {
    parse: wgpu::ComputePipeline,
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
        if !(4..=65_535).contains(&params.probe_len) {
            return unsupported(format!("probe_len {} not in 4..=65535", params.probe_len));
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
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("encode"),
                entries: &[
                    storage(0, true),  // input
                    storage(1, false), // scratch
                    storage(2, false), // output
                    storage(3, false), // sizes
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    storage(5, false), // chunk_info
                    storage(6, false), // group_buf
                ],
            });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("encode"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = |label: &str, kernel: &[&str], constants: &[(&str, f64)]| {
            let source = std::iter::once(COMMON_SHADER)
                .chain(kernel.iter().copied())
                .collect::<Vec<_>>()
                .join("\n");
            let module = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
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
                &[PARSE_SHADER],
                &[("CODEC", f64::from(id))],
            ),
            emit: pipeline(&format!("{name} emit"), &[EMIT_SHARED, emit], &[wg]),
        };
        Ok(GpuEncoder {
            params,
            matches: pipeline(
                "encode matches",
                &[MATCHES_SHADER],
                &[wg, ("HASH_LOG", f64::from(params.hash_log))],
            ),
            lz4: codec("lz4", LZ4_EMIT, 0),
            glz: codec("glz", GLZ_EMIT, 1),
            layout,
        })
    }

    pub fn params(&self) -> EncodeParams {
        self.params
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
    pub fn compress(
        &self,
        ctx: &Context,
        input: &[u8],
        options: &GpuCompressOptions,
    ) -> Result<Vec<u8>, GpuEncodeError> {
        let chunk_size = options.chunk_size;
        let chunk_count = format::chunk_count_for(input.len() as u64, chunk_size.max(1));
        let header = format::Header {
            codec: options.codec,
            chunk_size,
            chunk_count: u32::try_from(chunk_count)
                .map_err(|_| GpuEncodeError::TooManyChunks(chunk_count))?,
            total_size: input.len() as u64,
            checksums: options.checksums,
            level: options.level,
        };
        let encoded = self.encode(ctx, input, options)?;
        // Same stored fallback as the CPU encoders: keep a block only if it shrank.
        let payloads: Vec<format::ChunkPayload> = input
            .chunks(chunk_size as usize)
            .enumerate()
            .map(|(i, chunk)| {
                // Blocks that don't shrink their chunk were never written.
                let block = encoded.block(i);
                let stored = block.len() >= chunk.len();
                format::ChunkPayload {
                    bytes: if stored { chunk } else { block },
                    stored,
                    checksum: if options.checksums {
                        format::checksum(chunk)
                    } else {
                        0
                    },
                }
            })
            .collect();
        Ok(format::assemble(header, &payloads))
    }

    /// Runs the encoder over all chunks, in as many dispatches as the device's
    /// binding limit requires.
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
        let limit = ctx
            .device_limits()
            .max_storage_buffer_binding_size
            .min(u64::from(u32::MAX) & !3);
        let per_batch = chunks_per_batch(chunk_size, limit) as usize * chunk_size as usize;
        for batch in input.chunks(per_batch.max(1)) {
            self.encode_batch(ctx, batch, options, &mut encoded)?;
        }
        Ok(encoded)
    }

    fn encode_batch(
        &self,
        ctx: &Context,
        batch: &[u8],
        options: &GpuCompressOptions,
        encoded: &mut Encoded,
    ) -> Result<(), GpuEncodeError> {
        let prepared = self.prepare_batch(ctx, batch, options);
        self.dispatch(ctx, &prepared, None);
        let sizes: Vec<u32> = bytemuck::pod_collect_to_vec(
            &ctx.read_buffer(&prepared.sizes_buf, prepared.sizes_buf.size())?,
        );
        encoded.sizes.extend(sizes);
        encoded
            .slots
            .extend(ctx.read_buffer(&prepared.output_buf, prepared.output_buf.size())?);
        Ok(())
    }

    /// Uploads `input` as a single batch, ready to [`dispatch`](Self::dispatch),
    /// so benchmarks can time the kernel apart from transfers. Fails if the
    /// input needs more than one batch on this device.
    pub fn prepare(
        &self,
        ctx: &Context,
        input: &[u8],
        options: &GpuCompressOptions,
    ) -> Result<PreparedEncode, GpuEncodeError> {
        check_options(options)?;
        let chunk_size = options.chunk_size;
        let limit = ctx
            .device_limits()
            .max_storage_buffer_binding_size
            .min(u64::from(u32::MAX) & !3);
        let chunks = (input.len() as u64).div_ceil(u64::from(chunk_size)).max(1);
        if chunks > chunks_per_batch(chunk_size, limit) {
            return Err(GpuEncodeError::Unsupported(format!(
                "{chunks} chunks don't fit one batch on this device"
            )));
        }
        Ok(self.prepare_batch(ctx, input, options))
    }

    fn prepare_batch(
        &self,
        ctx: &Context,
        batch: &[u8],
        options: &GpuCompressOptions,
    ) -> PreparedEncode {
        let device = &ctx.device;
        let chunk_size = options.chunk_size;
        let chunks = (batch.len() as u64).div_ceil(u64::from(chunk_size)).max(1);
        let slot = slot_size(chunk_size);

        // Pad the input to whole words (plus one) so unaligned reads stay in bounds.
        let mut words = batch.to_vec();
        words.resize(format::pad4(batch.len() as u64) as usize + 4, 0);
        let input_buf = ctx.upload(&words, BufferUsages::STORAGE);
        let scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("encode scratch"),
            size: chunks * u64::from(chunk_size) * 4,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("encode output"),
            size: chunks * u64::from(slot),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sizes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("encode sizes"),
            size: chunks * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let groups = options.independent_groups.unwrap_or(0);
        let params = [
            chunk_size,
            batch.len() as u32,
            slot,
            self.params.probe_len,
            groups,
            0,
            0,
            0,
        ];
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("encode params"),
            contents: bytemuck::cast_slice(&params),
            usage: BufferUsages::UNIFORM,
        });
        let info_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("encode chunk info"),
            size: chunks * 16,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // Only the GLZ parse with dependency elimination uses the group lists.
        let group_bytes = if groups > 0 {
            chunks * u64::from(MAX_GROUP) * 8
        } else {
            8
        };
        let group_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("encode group lists"),
            size: group_bytes,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("encode"),
            layout: &self.layout,
            entries: &[
                (0, &input_buf),
                (1, &scratch_buf),
                (2, &output_buf),
                (3, &sizes_buf),
                (4, &params_buf),
                (5, &info_buf),
                (6, &group_buf),
            ]
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }),
        });
        PreparedEncode {
            codec: options.codec,
            chunks: chunks as u32,
            output_buf,
            sizes_buf,
            bind_group,
        }
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
        encoder.clear_buffer(&p.output_buf, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("encode"),
                timestamp_writes: timer.map(crate::GpuTimer::pass_writes),
            });
            // Three kernels; wgpu orders their storage accesses.
            let max = ctx.device_limits().max_compute_workgroups_per_dimension;
            let per_chunk = dispatch_grid(p.chunks, max);
            let parse = dispatch_grid(p.chunks.div_ceil(PARSE_WG), max);
            let kernels = self.kernels(p.codec);
            pass.set_bind_group(0, &p.bind_group, &[]);
            for (pipeline, (x, y)) in [
                (&self.matches, per_chunk),
                (&kernels.parse, parse),
                (&kernels.emit, per_chunk),
            ] {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(x, y, 1);
            }
        }
        if let Some(timer) = timer {
            timer.resolve(&mut encoder);
        }
        ctx.queue.submit([encoder.finish()]);
    }
}

/// GPU buffers for one encode batch, ready to [`dispatch`](GpuEncoder::dispatch).
pub struct PreparedEncode {
    codec: format::Codec,
    chunks: u32,
    output_buf: wgpu::Buffer,
    sizes_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
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

    #[test]
    fn slots_hold_the_lz4_worst_case_in_whole_words() {
        assert_eq!(slot_size(4096), 4096 + 16 + 16);
        assert_eq!(slot_size(65_536), 65_536 + 257 + 16 + 3);
        assert_eq!(slot_size(65_536) % 4, 0);
    }

    #[test]
    fn workgroup_memory_is_the_larger_of_the_table_and_the_emit_scans() {
        // Match finding holds the hash table; emit holds two scans (u32 and
        // vec2 per invocation) and counters; the parse holds nothing.
        let p = EncodeParams::default();
        assert_eq!(workgroup_bytes(&p), 4 * 4096);
        let small_table = EncodeParams {
            hash_log: 8,
            block: 256,
            ..p
        };
        assert_eq!(workgroup_bytes(&small_table), 12 * 256 + 16);
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
