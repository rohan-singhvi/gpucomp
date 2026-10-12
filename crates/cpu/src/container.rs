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

/// Whether the encoder tries filters on each chunk (plan §4a).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterMode {
    /// Every chunk is compressed unfiltered.
    #[default]
    None,
    /// Candidates compete on each chunk's leading sample
    /// ([`crate::filter::sample_len`]); the winner encodes the whole chunk.
    Auto,
    /// Every chunk is compressed under each of the level's candidate filters
    /// ([`crate::filter::candidates`]) and the smallest payload wins.
    Exhaustive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressOptions {
    pub codec: Codec,
    pub chunk_size: u32,
    pub encoder: Encoder,
    pub checksums: bool,
    /// Compression level; with [`FilterMode::Auto`] it picks the filter candidates.
    pub level: u8,
    pub filters: FilterMode,
}

impl Default for CompressOptions {
    fn default() -> Self {
        CompressOptions {
            codec: Codec::Lz4,
            chunk_size: format::DEFAULT_CHUNK_SIZE,
            encoder: Encoder::Lz4Flex,
            checksums: false,
            level: 1,
            filters: FilterMode::None,
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
    let header = header_for(input.len(), options)?;
    let table_len = Index {
        header,
        chunks: Vec::new(),
    }
    .data_offset() as usize;
    let mut out = vec![0u8; table_len];
    let chunks = compress_into(input, options, &mut out)?;
    out[..table_len].copy_from_slice(&Index { header, chunks }.to_bytes());
    Ok(out)
}

/// The header of `input`'s file, checked (chunk size, chunk count).
fn header_for(len: usize, options: &CompressOptions) -> Result<Header, CpuError> {
    let chunk_size = options.chunk_size;
    let chunk_count = format::chunk_count_for(len as u64, chunk_size.max(1));
    let header = Header {
        codec: options.codec,
        chunk_size,
        chunk_count: u32::try_from(chunk_count)
            .map_err(|_| CpuError::TooManyChunks(chunk_count))?,
        total_size: len as u64,
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
    Ok(header)
}

/// Appends the data section of `input`'s file (its chunks' payloads, each
/// padded to 4 bytes, back to back) to `out`, and returns the chunk entries,
/// with offsets from where the data starts. [`compress`] puts the header and
/// table in front; a caller joining several inputs' data can write its own.
pub fn compress_into(
    input: &[u8],
    options: &CompressOptions,
    out: &mut Vec<u8>,
) -> Result<Vec<ChunkEntry>, CpuError> {
    let chunk_size = header_for(input.len(), options)?.chunk_size;
    match (options.codec, options.encoder) {
        (Codec::Glz | Codec::GlzE, Encoder::Glz(_)) | (Codec::Stored, _) => {}
        (Codec::Lz4, Encoder::Lz4Flex | Encoder::Greedy(_)) => {}
        (Codec::Glz | Codec::GlzE, _) => {
            return Err(CpuError::Options("the GLZ codecs need Encoder::Glz"))
        }
        (Codec::Lz4, Encoder::Glz(_)) => {
            return Err(CpuError::Options("Encoder::Glz needs the GLZ codec"))
        }
    }

    // The stored codec never filters: a stored chunk is stored unfiltered.
    let candidates: &[Filter] = match (options.filters, options.codec) {
        (FilterMode::Auto | FilterMode::Exhaustive, Codec::Lz4 | Codec::Glz | Codec::GlzE) => {
            crate::filter::candidates(options.level)
        }
        _ => &[Filter::None],
    };

    // Compress every chunk independently, in parallel.
    let payloads: Vec<(Vec<u8>, bool, u32, Filter)> = input
        .par_chunks(chunk_size as usize)
        .map(|chunk| {
            let sum = if options.checksums {
                checksum(chunk)
            } else {
                0
            };
            let encode = |block: &[u8]| encode_block(options, block);
            let best = match options.filters {
                // Candidates compete on the leading sample; the winner then
                // encodes the whole chunk.
                FilterMode::Auto if candidates.len() > 1 => {
                    let sample = crate::filter::sample_len(chunk_size) as usize;
                    select(&chunk[..sample.min(chunk.len())], candidates, encode)
                        .and_then(|(_, filter)| select(chunk, &[filter], encode))
                }
                _ => select(chunk, candidates, encode),
            };
            match best {
                Some((c, filter)) if c.len() < chunk.len() => (c, false, sum, filter),
                _ => (chunk.to_vec(), true, sum, Filter::None),
            }
        })
        .collect();

    let start = out.len();
    out.reserve(payloads.iter().map(|p| p.0.len() + 3).sum());
    let mut chunks = Vec::with_capacity(payloads.len());
    for ((bytes, stored, checksum, filter), chunk) in
        payloads.iter().zip(input.chunks(chunk_size as usize))
    {
        chunks.push(ChunkEntry {
            comp_offset: (out.len() - start) as u64,
            comp_size: bytes.len() as u32,
            stored: *stored,
            uncomp_size: chunk.len() as u32,
            checksum: *checksum,
            filter: *filter,
        });
        out.extend_from_slice(bytes);
        out.resize(start + format::pad4((out.len() - start) as u64) as usize, 0);
    }
    Ok(chunks)
}

/// Encodes one (possibly filtered) chunk with the configured block encoder;
/// `None` for the stored codec.
fn encode_block(options: &CompressOptions, block: &[u8]) -> Option<Vec<u8>> {
    match (options.codec, options.encoder) {
        (Codec::Stored, _) => None,
        (Codec::Lz4, Encoder::Lz4Flex) => Some(lz4_flex::block::compress(block)),
        (Codec::Lz4, Encoder::Greedy(params)) => {
            Some(crate::lz4::encode::encode_block(block, &params))
        }
        (Codec::Glz, Encoder::Glz(params)) => Some(crate::glz::encode_block(block, &params)),
        (Codec::GlzE, Encoder::Glz(params)) => Some(crate::glze::encode_block(block, &params)),
        // Rejected by `compress` before any chunk is encoded.
        (Codec::Lz4, Encoder::Glz(_)) | (Codec::Glz | Codec::GlzE, _) => unreachable!(),
    }
}

/// Encodes `chunk` under every candidate filter and returns the smallest
/// block with its filter. Candidates are in tie-break order, and a later
/// candidate wins only if it is strictly smaller. `None` if `encode` returns
/// `None` (stored codec).
fn select(
    chunk: &[u8],
    candidates: &[Filter],
    encode: impl Fn(&[u8]) -> Option<Vec<u8>>,
) -> Option<(Vec<u8>, Filter)> {
    let mut best: Option<(Vec<u8>, Filter)> = None;
    let mut scratch = Vec::new();
    for &filter in candidates {
        let block = if filter == Filter::None {
            encode(chunk)?
        } else {
            scratch.resize(chunk.len(), 0);
            crate::filter::forward_into(filter, chunk, &mut scratch);
            encode(&scratch)?
        };
        if best.as_ref().is_none_or(|(b, _)| block.len() < b.len()) {
            best = Some((block, filter));
        }
    }
    best
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

/// Decodes chunks `chunks` of `file` (whose index is `index`, already
/// validated) into `dst`, which must hold exactly their uncompressed bytes.
/// For callers that split a file's chunks between decoders.
pub fn decompress_chunks_into(
    file: &[u8],
    index: &Index,
    chunks: std::ops::Range<usize>,
    dst: &mut [u8],
    options: &DecompressOptions,
) -> Result<(), CpuError> {
    let entries = index
        .chunks
        .get(chunks.clone())
        .ok_or(CpuError::Options("chunk range out of bounds"))?;
    let expected: u64 = entries.iter().map(|e| u64::from(e.uncomp_size)).sum();
    if dst.len() as u64 != expected {
        return Err(CpuError::Options(
            "destination isn't the chunks' uncompressed size",
        ));
    }
    let data = &file[index.data_offset() as usize..];
    dst.par_chunks_mut(index.header.chunk_size as usize)
        .zip(entries)
        .enumerate()
        .try_for_each(|(i, (dst, entry))| {
            let start = entry.comp_offset as usize;
            let payload = &data[start..start + entry.comp_size as usize];
            decode_chunk(
                index.header.codec,
                chunks.start + i,
                entry,
                index.header.checksums,
                payload,
                dst,
                options,
            )
        })
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
    if entry.stored {
        dst.copy_from_slice(payload);
    } else if matches!(codec, Codec::Glz | Codec::GlzE) {
        if options.decoder == Decoder::Lz4Flex {
            return Err(CpuError::Options("lz4_flex can't decode GLZ files"));
        }
        let decode = if codec == Codec::Glz {
            crate::glz::decode_block
        } else {
            crate::glze::decode_block
        };
        decode(payload, dst).map_err(|source| CpuError::Glz { chunk, source })?;
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
    // Checksums cover the original (unfiltered) bytes.
    crate::filter::inverse_in_place(entry.filter, dst);
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
