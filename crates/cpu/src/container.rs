//! Chunked `.gpcz` compression and decompression on the CPU, parallel over
//! chunks with rayon. These are the fair multi-threaded CPU baselines.

use std::io::{Read, Seek, SeekFrom};

pub use format::{checksum, read_index};
use format::{ChunkEntry, Codec, Filter, FormatError, Header, Index};
use rayon::prelude::*;

use crate::lz4::decode::DecodeError;
use crate::lz4::encode::Params;

#[derive(Debug, thiserror::Error)]
pub enum CpuError {
    #[error(transparent)]
    Format(#[from] FormatError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("chunk {chunk}: {source}")]
    Decode { chunk: usize, source: DecodeError },
    #[error("chunk {chunk}: lz4_flex: {message}")]
    Lz4Flex { chunk: usize, message: String },
    #[error("chunk {chunk}: checksum mismatch")]
    Checksum { chunk: usize },
    #[error("chunk {chunk}: filter not supported by this decoder")]
    UnsupportedFilter { chunk: usize },
    #[error("input too large: {0} chunks")]
    TooManyChunks(u64),
    #[error("invalid options: {0}")]
    Options(&'static str),
    #[error("chunk {chunk}: {source}")]
    Glz {
        chunk: usize,
        source: crate::glz::GlzError,
    },
}

impl From<format::ReadError> for CpuError {
    fn from(e: format::ReadError) -> Self {
        match e {
            format::ReadError::Io(e) => CpuError::Io(e),
            format::ReadError::Format(e) => CpuError::Format(e),
        }
    }
}

/// Which LZ4 block encoder compresses each chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoder {
    /// `lz4_flex`: the CPU compression baseline.
    Lz4Flex,
    /// The hand-written greedy encoder, the GPU encoder's CPU twin.
    Greedy(Params),
    /// The GLZ encoder (codec GLZ only).
    Glz(crate::glz::GlzParams),
}

/// Which LZ4 block decoder decodes each chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decoder {
    HandWritten,
    Lz4Flex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressOptions {
    pub codec: Codec,
    pub chunk_size: u32,
    pub encoder: Encoder,
    pub checksums: bool,
    pub level: u8,
}

impl Default for CompressOptions {
    fn default() -> Self {
        CompressOptions {
            codec: Codec::Lz4,
            chunk_size: format::DEFAULT_CHUNK_SIZE,
            encoder: Encoder::Lz4Flex,
            checksums: false,
            level: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecompressOptions {
    pub decoder: Decoder,
    /// Check per-chunk checksums when the file has them.
    pub verify: bool,
}

impl Default for DecompressOptions {
    fn default() -> Self {
        DecompressOptions {
            decoder: Decoder::HandWritten,
            verify: true,
        }
    }
}

pub fn compress(input: &[u8], options: &CompressOptions) -> Result<Vec<u8>, CpuError> {
    let chunk_size = options.chunk_size;
    let chunk_count = format::chunk_count_for(input.len() as u64, chunk_size.max(1));
    let header = Header {
        codec: options.codec,
        chunk_size,
        chunk_count: u32::try_from(chunk_count)
            .map_err(|_| CpuError::TooManyChunks(chunk_count))?,
        total_size: input.len() as u64,
        checksums: options.checksums,
        level: options.level,
    };
    // Validate the header (chunk size) before doing any work.
    Index {
        header,
        chunks: Vec::new(),
    }
    .validate(0)
    .or_else(|e| match e {
        FormatError::ChunkCountMismatch { .. } => Ok(()),
        e => Err(e),
    })?;

    match (options.codec, options.encoder) {
        (Codec::Glz, Encoder::Glz(_)) | (Codec::Stored, _) => {}
        (Codec::Lz4, Encoder::Lz4Flex | Encoder::Greedy(_)) => {}
        (Codec::Glz, _) => return Err(CpuError::Options("the GLZ codec needs Encoder::Glz")),
        (Codec::Lz4, Encoder::Glz(_)) => {
            return Err(CpuError::Options("Encoder::Glz needs the GLZ codec"))
        }
    }

    // Compress every chunk independently, in parallel.
    let payloads: Vec<(Vec<u8>, bool, u32)> = input
        .par_chunks(chunk_size as usize)
        .map(|chunk| {
            let sum = if options.checksums {
                checksum(chunk)
            } else {
                0
            };
            let compressed = match (options.codec, options.encoder) {
                (Codec::Stored, _) => None,
                (Codec::Lz4, Encoder::Lz4Flex) => Some(lz4_flex::block::compress(chunk)),
                (Codec::Lz4, Encoder::Greedy(params)) => {
                    Some(crate::lz4::encode::encode_block(chunk, &params))
                }
                (Codec::Glz, Encoder::Glz(params)) => {
                    Some(crate::glz::encode_block(chunk, &params))
                }
                // Rejected by the check above.
                (Codec::Lz4, Encoder::Glz(_)) | (Codec::Glz, _) => unreachable!(),
            };
            match compressed {
                Some(c) if c.len() < chunk.len() => (c, false, sum),
                _ => (chunk.to_vec(), true, sum),
            }
        })
        .collect();

    let payloads: Vec<format::ChunkPayload> = payloads
        .iter()
        .map(|(bytes, stored, checksum)| format::ChunkPayload {
            bytes,
            stored: *stored,
            checksum: *checksum,
        })
        .collect();
    Ok(format::assemble(header, &payloads))
}

pub fn decompress(file: &[u8], options: &DecompressOptions) -> Result<Vec<u8>, CpuError> {
    let index = Index::parse(file)?;
    let data = &file[index.data_offset() as usize..];
    index.validate(data.len() as u64)?;
    let mut out = vec![0u8; index.header.total_size as usize];
    out.par_chunks_mut(index.header.chunk_size as usize)
        .zip(&index.chunks)
        .enumerate()
        .try_for_each(|(i, (dst, entry))| {
            let start = entry.comp_offset as usize;
            let payload = &data[start..start + entry.comp_size as usize];
            decode_chunk(
                index.header.codec,
                i,
                entry,
                index.header.checksums,
                payload,
                dst,
                options,
            )
        })?;
    Ok(out)
}

/// Decodes one chunk's payload into `dst` (exactly `uncomp_size` bytes).
fn decode_chunk(
    codec: Codec,
    chunk: usize,
    entry: &ChunkEntry,
    has_checksums: bool,
    payload: &[u8],
    dst: &mut [u8],
    options: &DecompressOptions,
) -> Result<(), CpuError> {
    if entry.filter != Filter::None {
        return Err(CpuError::UnsupportedFilter { chunk });
    }
    if entry.stored {
        dst.copy_from_slice(payload);
    } else if codec == Codec::Glz {
        if options.decoder == Decoder::Lz4Flex {
            return Err(CpuError::Options("lz4_flex can't decode GLZ files"));
        }
        crate::glz::decode_block(payload, dst).map_err(|source| CpuError::Glz { chunk, source })?;
    } else {
        match options.decoder {
            Decoder::HandWritten => crate::lz4::decode::decode_block(payload, dst)
                .map_err(|source| CpuError::Decode { chunk, source })?,
            Decoder::Lz4Flex => {
                let n = lz4_flex::block::decompress_into(payload, dst).map_err(|e| {
                    CpuError::Lz4Flex {
                        chunk,
                        message: e.to_string(),
                    }
                })?;
                if n != dst.len() {
                    return Err(CpuError::Lz4Flex {
                        chunk,
                        message: format!("decoded {n} bytes, expected {}", dst.len()),
                    });
                }
            }
        }
    }
    if options.verify && has_checksums && checksum(dst) != entry.checksum {
        return Err(CpuError::Checksum { chunk });
    }
    Ok(())
}

/// Decompresses `[offset, offset + len)` of the original, reading only the
/// index and the chunks that overlap the range.
pub fn decompress_range<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    len: u64,
    options: &DecompressOptions,
) -> Result<Vec<u8>, CpuError> {
    let index = read_index(reader)?;
    let chunks = index.chunks_for_range(offset, len)?;
    if chunks.is_empty() {
        return Ok(Vec::new());
    }
    let chunk_size = u64::from(index.header.chunk_size);
    let first_start = chunks.start as u64 * chunk_size;

    // Read the needed payloads (contiguous in the file), then decode in parallel.
    let entries = &index.chunks[chunks.clone()];
    let payload_start = entries[0].comp_offset;
    let last = entries[entries.len() - 1];
    let payload_end = last.comp_offset + u64::from(last.comp_size);
    reader.seek(SeekFrom::Start(index.data_offset() + payload_start))?;
    let mut payloads = vec![0u8; (payload_end - payload_start) as usize];
    reader.read_exact(&mut payloads)?;

    let span: u64 = entries.iter().map(|e| u64::from(e.uncomp_size)).sum();
    let mut out = vec![0u8; span as usize];
    out.par_chunks_mut(chunk_size as usize)
        .zip(entries)
        .enumerate()
        .try_for_each(|(k, (dst, entry))| {
            let start = (entry.comp_offset - payload_start) as usize;
            let payload = &payloads[start..start + entry.comp_size as usize];
            decode_chunk(
                index.header.codec,
                chunks.start + k,
                entry,
                index.header.checksums,
                payload,
                dst,
                options,
            )
        })?;
    let skip = (offset - first_start) as usize;
    out.drain(..skip);
    out.truncate(len as usize);
    Ok(out)
}
