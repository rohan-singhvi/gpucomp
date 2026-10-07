//! GPU LZ4 compression (M3): one workgroup per chunk, running the same
//! algorithm as `cpu::lz4::encode` so outputs are byte-identical.

use wgpu::util::DeviceExt as _;
use wgpu::BufferUsages;

use crate::{dispatch_grid, Context, GpuError};

const SHADER: &str = include_str!("../../shaders/lz4_encode.wgsl");

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
    pub chunk_size: u32,
    pub checksums: bool,
    pub level: u8,
}

impl Default for GpuCompressOptions {
    fn default() -> Self {
        GpuCompressOptions {
            chunk_size: format::DEFAULT_CHUNK_SIZE,
            checksums: false,
            level: 1,
        }
    }
}

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

/// One chunk's LZ4 block, as reported by [`Lz4GpuEncoder::encode_blocks`].
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

/// Workgroup memory the encoder needs: the hash table plus the scan scratch.
pub fn workgroup_bytes(params: &EncodeParams) -> u32 {
    4 * (1 << params.hash_log) + 4 * params.block + 4
}

/// How many chunks one dispatch may encode so that the input, match scratch
/// and output slots each fit in `binding_limit` bytes.
pub fn chunks_per_batch(chunk_size: u32, binding_limit: u64) -> u64 {
    // Match scratch (one u32 per input byte) is the largest buffer.
    (binding_limit / (4 * u64::from(chunk_size))).max(1)
}

pub struct Lz4GpuEncoder {
    params: EncodeParams,
    pipeline: wgpu::ComputePipeline,
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

impl Lz4GpuEncoder {
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
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("lz4 encode"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("lz4 encode"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("WG_SIZE", f64::from(params.block)),
                        ("HASH_LOG", f64::from(params.hash_log)),
                    ],
                    ..Default::default()
                },
                cache: None,
            });
        Ok(Lz4GpuEncoder { params, pipeline })
    }

    pub fn params(&self) -> EncodeParams {
        self.params
    }

    /// Encodes every `chunk_size` chunk of `input` as an LZ4 block. For tests
    /// and debugging.
    pub fn encode_blocks(
        &self,
        ctx: &Context,
        input: &[u8],
        chunk_size: u32,
    ) -> Result<Vec<EncodedBlock>, GpuEncodeError> {
        let encoded = self.encode(ctx, input, chunk_size)?;
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

    /// Compresses `input` into a `.gpcz` LZ4 container.
    pub fn compress(
        &self,
        ctx: &Context,
        input: &[u8],
        options: &GpuCompressOptions,
    ) -> Result<Vec<u8>, GpuEncodeError> {
        let chunk_size = options.chunk_size;
        let chunk_count = format::chunk_count_for(input.len() as u64, chunk_size.max(1));
        let header = format::Header {
            codec: format::Codec::Lz4,
            chunk_size,
            chunk_count: u32::try_from(chunk_count)
                .map_err(|_| GpuEncodeError::TooManyChunks(chunk_count))?,
            total_size: input.len() as u64,
            checksums: options.checksums,
            level: options.level,
        };
        check_chunk_size(chunk_size)?;
        let encoded = self.encode(ctx, input, chunk_size)?;
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
        chunk_size: u32,
    ) -> Result<Encoded, GpuEncodeError> {
        check_chunk_size(chunk_size)?;
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
            self.encode_batch(ctx, batch, chunk_size, &mut encoded)?;
        }
        Ok(encoded)
    }

    fn encode_batch(
        &self,
        ctx: &Context,
        batch: &[u8],
        chunk_size: u32,
        encoded: &mut Encoded,
    ) -> Result<(), GpuEncodeError> {
        let prepared = self.prepare_batch(ctx, batch, chunk_size);
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
        chunk_size: u32,
    ) -> Result<PreparedEncode, GpuEncodeError> {
        check_chunk_size(chunk_size)?;
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
        Ok(self.prepare_batch(ctx, input, chunk_size))
    }

    fn prepare_batch(&self, ctx: &Context, batch: &[u8], chunk_size: u32) -> PreparedEncode {
        let device = &ctx.device;
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
        let params = [chunk_size, batch.len() as u32, slot, self.params.probe_len];
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("encode params"),
            contents: bytemuck::cast_slice(&params),
            usage: BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz4 encode"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                (0, &input_buf),
                (1, &scratch_buf),
                (2, &output_buf),
                (3, &sizes_buf),
                (4, &params_buf),
            ]
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }),
        });
        PreparedEncode {
            chunks: chunks as u32,
            output_buf,
            sizes_buf,
            bind_group,
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
                label: Some("lz4 encode"),
                timestamp_writes: timer.map(crate::GpuTimer::pass_writes),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &p.bind_group, &[]);
            let (x, y) = dispatch_grid(
                p.chunks,
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

/// GPU buffers for one encode batch, ready to [`dispatch`](Lz4GpuEncoder::dispatch).
pub struct PreparedEncode {
    chunks: u32,
    output_buf: wgpu::Buffer,
    sizes_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
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
    fn workgroup_memory_is_table_plus_scan() {
        let p = EncodeParams::default();
        assert_eq!(workgroup_bytes(&p), 4 * 4096 + 4 * 64 + 4);
        let small = EncodeParams {
            hash_log: 10,
            block: 128,
            ..p
        };
        assert_eq!(workgroup_bytes(&small), 4 * 1024 + 4 * 128 + 4);
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
