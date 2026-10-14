//! Chunked `.gpcz` compression and decompression on the CPU, parallel over
//! chunks with rayon. These are the fair multi-threaded CPU baselines.

use std::io::{Read, Seek, SeekFrom, Write};

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
    let header = header_for(input.len() as u64, options)?;
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
fn header_for(len: u64, options: &CompressOptions) -> Result<Header, CpuError> {
    let chunk_size = options.chunk_size;
    let chunk_count = format::chunk_count_for(len, chunk_size.max(1));
    let header = Header {
        codec: options.codec,
        chunk_size,
        chunk_count: u32::try_from(chunk_count)
            .map_err(|_| CpuError::TooManyChunks(chunk_count))?,
        total_size: len,
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

/// Rejects an encoder that can't write the codec.
fn check_encoder(options: &CompressOptions) -> Result<(), CpuError> {
    match (options.codec, options.encoder) {
        (Codec::Glz | Codec::GlzE, Encoder::Glz(_)) | (Codec::Stored, _) => Ok(()),
        (Codec::Lz4, Encoder::Lz4Flex | Encoder::Greedy(_)) => Ok(()),
        (Codec::Glz | Codec::GlzE, _) => Err(CpuError::Options("the GLZ codecs need Encoder::Glz")),
        (Codec::Lz4, Encoder::Glz(_)) => Err(CpuError::Options("Encoder::Glz needs the GLZ codec")),
    }
}

/// Appends the data section of `input`'s file (its chunks' payloads, each
/// padded to 4 bytes, back to back) to `out`, and returns the chunk entries,
/// with offsets from where the data starts. [`compress`] puts the header and
/// table in front; a caller joining several inputs' data can write its own.
/// On an error, `out` is left as it was.
///
/// An input of several groups goes through the streaming pipeline: small
/// groups, several at once, keep the working set in cache, ~10% faster than
/// encoding every chunk first and copying them all into `out` at the end
/// (DECISIONS "Streamed fixed split").
pub fn compress_into(
    input: &[u8],
    options: &CompressOptions,
    out: &mut Vec<u8>,
) -> Result<Vec<ChunkEntry>, CpuError> {
    let group = stream_group_bytes(options.chunk_size);
    let start = out.len();
    let result = if input.len() <= PIPELINE_MIN_GROUPS * group {
        encode_group(input, options, out)
    } else {
        let mut reader = input;
        compress_stream_data(&mut reader, input.len() as u64, out, options, group, None)
            .map(|(chunks, _)| chunks)
    };
    if result.is_err() {
        out.truncate(start);
    }
    result
}

/// Inputs of at most this many streaming groups are encoded directly, without
/// starting the pipeline's threads.
const PIPELINE_MIN_GROUPS: usize = 4;

/// Encodes every chunk of `input` in parallel, then appends their payloads to
/// `out`: [`compress_into`] for one group.
fn encode_group(
    input: &[u8],
    options: &CompressOptions,
    out: &mut Vec<u8>,
) -> Result<Vec<ChunkEntry>, CpuError> {
    let chunk_size = header_for(input.len() as u64, options)?.chunk_size;
    check_encoder(options)?;

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

/// Input (or, decoding, output) bytes per streaming group, at least: small
/// groups, many in flight, keep the working set in cache (DECISIONS
/// "Streaming CLI paths").
const STREAM_GROUP_BYTES: usize = 256 << 10;

/// The default group size for streaming `chunk_size` chunks: at least
/// [`STREAM_GROUP_BYTES`], and at least one chunk.
pub fn stream_group_bytes(chunk_size: u32) -> usize {
    STREAM_GROUP_BYTES.max(chunk_size as usize)
}

/// Streaming groups in flight: four per core, so every core has work while
/// one group is read and another written.
fn stream_depth() -> usize {
    (4 * rayon::current_num_threads()).max(8)
}

/// Compresses `len` bytes from `reader` into a `.gpcz` file written to
/// `writer` from its current position, and returns the bytes written; the
/// writer is left at the file's end. The file is the one [`compress`]
/// writes. Reading, compressing (on every core) and writing overlap, and
/// only a few groups of chunks are in memory at a time.
pub fn compress_stream<R: Read + Send, W: Write + Seek>(
    reader: &mut R,
    len: u64,
    writer: &mut W,
    options: &CompressOptions,
) -> Result<u64, CpuError> {
    let group = stream_group_bytes(options.chunk_size);
    compress_stream_grouped(reader, len, writer, options, group)
}

/// [`compress_stream`], reading `group_bytes` of input at a time (rounded
/// down to whole chunks, at least one).
pub fn compress_stream_grouped<R: Read + Send, W: Write + Seek>(
    reader: &mut R,
    len: u64,
    writer: &mut W,
    options: &CompressOptions,
    group_bytes: usize,
) -> Result<u64, CpuError> {
    let header = header_for(len, options)?;
    check_encoder(options)?;
    let table_len = Index {
        header,
        chunks: Vec::new(),
    }
    .data_offset();
    let start = writer.stream_position()?;
    // Reserved for the header and table, written last.
    writer.write_all(&vec![0; table_len as usize])?;
    let (chunks, data_len) = compress_stream_data(reader, len, writer, options, group_bytes, None)?;
    writer.seek(SeekFrom::Start(start))?;
    writer.write_all(&Index { header, chunks }.to_bytes())?;
    writer.seek(SeekFrom::Start(start + table_len + data_len))?;
    Ok(table_len + data_len)
}

/// Streams the data section of `len` bytes from `reader` to `writer`: each
/// chunk's payload, padded, back to back, as [`compress_into`] appends them.
/// Returns the chunk entries, with offsets from the data's start, and the
/// bytes written. The work runs on `pool` (rayon's global pool if `None`).
pub fn compress_stream_data<R: Read + Send, W: Write>(
    reader: &mut R,
    len: u64,
    writer: &mut W,
    options: &CompressOptions,
    group_bytes: usize,
    pool: Option<&rayon::ThreadPool>,
) -> Result<(Vec<ChunkEntry>, u64), CpuError> {
    let header = header_for(len, options)?;
    check_encoder(options)?;
    let chunk = options.chunk_size as usize;
    let group = (group_bytes / chunk).max(1) * chunk;
    let mut remaining = len;
    let mut chunks = Vec::with_capacity(header.chunk_count as usize);
    let mut data_len = 0u64;
    crate::pipeline::ordered(
        pool,
        stream_depth(),
        |input| {
            if remaining == 0 {
                return Ok(None);
            }
            let n = (group as u64).min(remaining) as usize;
            input.resize(n, 0);
            reader.read_exact(input)?;
            remaining -= n as u64;
            Ok(Some(()))
        },
        |(), input, out| encode_group(input, options, out),
        |data, entries| {
            writer.write_all(data)?;
            chunks.extend(entries.into_iter().map(|entry| ChunkEntry {
                comp_offset: data_len + entry.comp_offset,
                ..entry
            }));
            data_len += data.len() as u64;
            Ok(())
        },
    )?;
    Ok((chunks, data_len))
}

/// Decompresses the `.gpcz` file in `reader` (from its start) into `writer`,
/// and returns the bytes written. Reading, decoding (on every core) and
/// writing overlap, and only a few groups of chunks are in memory at a time.
/// On an error, the chunks before the failing one may have been written.
pub fn decompress_stream<R: Read + Seek + Send, W: Write>(
    reader: &mut R,
    writer: &mut W,
    options: &DecompressOptions,
) -> Result<u64, CpuError> {
    let index = read_index(reader)?;
    let group = stream_group_bytes(index.header.chunk_size);
    decompress_index_stream(reader, &index, writer, options, group)
}

/// [`decompress_stream`], decoding `group_bytes` of output at a time
/// (rounded down to whole chunks, at least one).
pub fn decompress_stream_grouped<R: Read + Seek + Send, W: Write>(
    reader: &mut R,
    writer: &mut W,
    options: &DecompressOptions,
    group_bytes: usize,
) -> Result<u64, CpuError> {
    let index = read_index(reader)?;
    decompress_index_stream(reader, &index, writer, options, group_bytes)
}

/// Streams the chunks of `index` (read and validated from `reader`).
fn decompress_index_stream<R: Read + Seek + Send, W: Write>(
    reader: &mut R,
    index: &Index,
    writer: &mut W,
    options: &DecompressOptions,
    group_bytes: usize,
) -> Result<u64, CpuError> {
    let header = index.header;
    let data_offset = index.data_offset();
    let chunk = header.chunk_size as usize;
    let per_group = (group_bytes / chunk).max(1);
    let mut next = 0;
    // Where the reader is, once known: payloads are usually back to back.
    let mut at = None;
    crate::pipeline::ordered(
        None,
        stream_depth(),
        |payloads| {
            if next == index.chunks.len() {
                return Ok(None);
            }
            let chunks = next..(next + per_group).min(index.chunks.len());
            next = chunks.end;
            let entries = &index.chunks[chunks.clone()];
            let lo = entries.iter().map(|e| e.comp_offset).min().unwrap_or(0);
            let hi = entries
                .iter()
                .map(|e| e.comp_offset + u64::from(e.comp_size))
                .max()
                .unwrap_or(0);
            if at != Some(data_offset + lo) {
                reader.seek(SeekFrom::Start(data_offset + lo))?;
            }
            payloads.resize((hi - lo) as usize, 0);
            reader.read_exact(payloads)?;
            at = Some(data_offset + hi);
            Ok(Some((chunks, lo)))
        },
        |(chunks, lo), payloads, out| {
            let entries = &index.chunks[chunks.clone()];
            out.resize(entries.iter().map(|e| e.uncomp_size as usize).sum(), 0);
            out.par_chunks_mut(chunk)
                .zip(entries)
                .enumerate()
                .try_for_each(|(i, (dst, entry))| {
                    let start = (entry.comp_offset - lo) as usize;
                    let payload = &payloads[start..start + entry.comp_size as usize];
                    decode_chunk(
                        header.codec,
                        chunks.start + i,
                        entry,
                        header.checksums,
                        payload,
                        dst,
                        options,
                    )
                })
        },
        |data, ()| Ok(writer.write_all(data)?),
    )?;
    Ok(header.total_size)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_inputs_stream_into_the_same_data_section() {
        // Text with repeats and a stretch of noise, 12 groups of 256 KiB.
        let mut s = 7u64;
        let input: Vec<u8> = (0..3 << 20)
            .map(|i: usize| {
                if i % 100_000 < 70_000 {
                    b"the quick brown fox jumps over the lazy dog "[i % 44]
                } else {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    s as u8
                }
            })
            .collect();
        for chunk_size in [4096, 65_536] {
            let options = CompressOptions {
                chunk_size,
                checksums: true,
                ..CompressOptions::default()
            };
            assert!(input.len() > PIPELINE_MIN_GROUPS * stream_group_bytes(chunk_size));
            let (mut direct, mut streamed) = (b"prefix".to_vec(), b"prefix".to_vec());
            let expected = encode_group(&input, &options, &mut direct).unwrap();
            let entries = compress_into(&input, &options, &mut streamed).unwrap();
            assert_eq!(entries, expected, "{chunk_size}");
            assert!(streamed == direct, "{chunk_size}");
        }
    }
}
