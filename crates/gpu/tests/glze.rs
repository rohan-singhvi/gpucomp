//! GPU GLZ-E decompression (M9e): entropy-decode each chunk back into its GLZ
//! block, then GLZ-decode it. Output must be exact, and malformed blocks must
//! report the error the CPU decoder (cpu::glze::decode_block) reports.

mod common;

use std::io::Cursor;

use common::{context, fixtures, random, text, CHUNK};
use cpu::container::{compress, CompressOptions, Encoder, FilterMode};
use cpu::glz::{GlzError, GlzParams};
use cpu::huffman::StreamError;
use cpu::lz4::encode::Params;
use format::{ChunkEntry, Codec, Filter, Header, Index};
use gpu::decode::{ChunkStatus, GpuDecodeError, GpuDecoder};

fn glze(level: u8, filters: FilterMode, chunk_size: u32) -> CompressOptions {
    CompressOptions {
        codec: Codec::GlzE,
        chunk_size,
        encoder: Encoder::Glz(GlzParams {
            lz: Params::for_level(level),
            independent_groups: None,
        }),
        checksums: true,
        level,
        filters,
    }
}

#[test]
fn every_glze_file_decodes_exactly_on_the_gpu() {
    let Some(ctx) = context() else { return };
    let decoder = GpuDecoder::new(&ctx);
    for level in [1, 3] {
        for filters in [FilterMode::None, FilterMode::Auto] {
            for (name, input) in fixtures() {
                let file = compress(&input, &glze(level, filters, CHUNK)).unwrap();
                let out = decoder
                    .decompress(&ctx, &file, true)
                    .unwrap_or_else(|e| panic!("{name} l{level} {filters:?}: {e}"));
                assert!(out == input, "{name} l{level} {filters:?}: output differs");
            }
        }
    }
    for (name, input, chunk) in [
        ("text, 64 KiB chunks", text(1 << 20), 1 << 16),
        ("zeros, 1 MiB chunks", vec![0u8; 3 << 20], 1 << 20),
        (
            "mixed, 64 KiB chunks",
            [text(300_000), random(100_000, 3), vec![5; 200_000]].concat(),
            1 << 16,
        ),
    ] {
        let file = compress(&input, &glze(1, FilterMode::None, chunk)).unwrap();
        let out = decoder.decompress(&ctx, &file, true).unwrap();
        assert!(out == input, "{name}");
    }
}

#[test]
fn glze_range_and_stream_reads_on_the_gpu() {
    let Some(ctx) = context() else { return };
    let decoder = GpuDecoder::new(&ctx);
    let input = text(10 * CHUNK as usize + 99);
    let file = compress(&input, &glze(1, FilterMode::None, CHUNK)).unwrap();
    let c = u64::from(CHUNK);
    for (offset, len) in [(0, 10), (c - 5, 10), (3 * c + 7, 2 * c)] {
        let got = decoder
            .decompress_range(&ctx, &mut Cursor::new(&file), offset, len, true)
            .unwrap();
        assert!(got == input[offset as usize..(offset + len) as usize]);
    }
    let mut out = Vec::new();
    decoder
        .decompress_stream(&ctx, &mut Cursor::new(&file), &mut out, true)
        .unwrap();
    assert!(out == input);
}

/// A one-chunk GLZ-E file whose payload is `payload`, decoding to `n` bytes.
fn single_chunk_file(payload: &[u8], n: u32) -> Vec<u8> {
    let index = Index {
        header: Header {
            codec: Codec::GlzE,
            chunk_size: 1 << 16,
            chunk_count: 1,
            total_size: u64::from(n),
            checksums: false,
            level: 1,
        },
        chunks: vec![ChunkEntry {
            comp_offset: 0,
            comp_size: payload.len() as u32,
            stored: false,
            uncomp_size: n,
            checksum: 0,
            filter: Filter::None,
        }],
    };
    let mut file = index.to_bytes();
    file.extend_from_slice(payload);
    file.resize(file.len().next_multiple_of(4), 0);
    file
}

fn expected_status(e: GlzError) -> ChunkStatus {
    match e {
        GlzError::Truncated => ChunkStatus::Truncated,
        GlzError::BadSequence => ChunkStatus::BadSequence,
        GlzError::ZeroOffset => ChunkStatus::ZeroOffset,
        GlzError::OffsetBeforeStart => ChunkStatus::OffsetBeforeStart,
        GlzError::OutputOverflow => ChunkStatus::OutputOverflow,
        GlzError::SizeMismatch => ChunkStatus::SizeMismatch,
        GlzError::Stream(StreamError::Truncated) => ChunkStatus::StreamTruncated,
        GlzError::Stream(StreamError::BadMode) => ChunkStatus::StreamBadMode,
        GlzError::Stream(StreamError::BadTable) => ChunkStatus::StreamBadTable,
        GlzError::Stream(StreamError::LaneOverrun) => ChunkStatus::StreamLaneOverrun,
    }
}

/// Start of each stream (tokens, off_lo, off_hi, literals) and of the
/// extension words, in a valid GLZ-E block.
fn layout(block: &[u8]) -> ([usize; 4], usize) {
    let word = |at: usize| u32::from_le_bytes(block[at..at + 4].try_into().unwrap());
    let count = (word(0) & !cpu::glz::WIDE_BIT) as usize;
    let wide = word(0) & cpu::glz::WIDE_BIT != 0;
    let ext = word(4) as usize;
    let walk = |at, n| cpu::huffman::decode_stream(block, at, n, &mut Vec::new()).unwrap();
    let tokens = 12;
    let lo = walk(tokens, count);
    let hi = walk(lo, count);
    let ext_at = walk(hi, count);
    let literals = ext_at + (ext * if wide { 4 } else { 2 }).next_multiple_of(4);
    ([tokens, lo, hi, literals], ext_at)
}

#[test]
fn malformed_glze_blocks_report_the_same_error_as_the_cpu_decoder() {
    let Some(ctx) = context() else { return };
    let decoder = GpuDecoder::new(&ctx);
    let input: Vec<u8> = (0..20_000u32)
        .flat_map(|i| format!("item {} of {}, ", (i * 7919) % 2003, i % 89).into_bytes())
        .take(1 << 16)
        .collect();
    let n = input.len() as u32;
    let block = cpu::glze::encode_block(&input, &GlzParams::default());
    let ([tokens, _, _, literals], ext_at) = layout(&block);
    assert_eq!(block[literals], 2, "the literal stream is Huffman-coded");
    let set = |at: usize, bytes: &[u8]| {
        let mut b = block.clone();
        b[at..at + bytes.len()].copy_from_slice(bytes);
        b
    };
    let lane_sizes = literals + 4 + 128;
    let first_lane_words = u16::from_le_bytes([block[lane_sizes], block[lane_sizes + 1]]);
    // A GLZ block with a zero offset, transcoded: a sequence error.
    let mut glz = cpu::glz::encode_block(&input, &GlzParams::default());
    let count = (u32::from_le_bytes(glz[..4].try_into().unwrap()) & !cpu::glz::WIDE_BIT) as usize;
    let offsets_at = 8 + count.next_multiple_of(4);
    glz[offsets_at..offsets_at + 2].copy_from_slice(&[0, 0]);
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", vec![]),
        ("header cut", block[..8].to_vec()),
        ("no sequences", set(0, &0u32.to_le_bytes())),
        ("ext over 2x count", set(4, &u32::MAX.to_le_bytes())),
        ("too many literals", set(8, &(n + 1).to_le_bytes())),
        ("tokens: unknown mode", set(tokens, &[9])),
        ("tokens: reserved bits", set(tokens + 2, &[1])),
        ("tokens: cut in the mode word", block[..tokens + 2].to_vec()),
        (
            "literals: cut in the table",
            block[..literals + 60].to_vec(),
        ),
        ("literals: length 12", set(literals + 4, &[0xCC])),
        (
            "literals: incomplete code",
            set(literals + 4 + input[0] as usize / 2, &[0]),
        ),
        (
            "literals: lane 0 overruns",
            set(lane_sizes, &(first_lane_words - 1).to_le_bytes()),
        ),
        (
            "literals: cut in the lanes",
            block[..block.len() - 8].to_vec(),
        ),
        ("cut before the extension words", block[..ext_at].to_vec()),
        ("trailing word", [block.clone(), vec![0; 4]].concat()),
        ("sequence error", cpu::glze::transcode(&glz)),
        ("valid block, wrong size", block.clone()),
    ];
    for (name, payload) in cases {
        let size = if name.contains("wrong size") {
            n - 1
        } else {
            n
        };
        let mut reference = vec![0; size as usize];
        let cpu_error = cpu::glze::decode_block(&payload, &mut reference).unwrap_err();
        match decoder.decompress(&ctx, &single_chunk_file(&payload, size), false) {
            Err(GpuDecodeError::Chunk { chunk: 0, status }) => {
                assert_eq!(status, expected_status(cpu_error), "{name}");
            }
            other => panic!("{name}: expected {cpu_error:?}, got {other:?}"),
        }
    }
}

// ---- Step 3: the GPU GLZ-E encoder equals the CPU twin ----

use gpu::encode::{EncodeParams, EncodedBlock, GpuCompressOptions, GpuEncoder};

fn varied(n: usize) -> Vec<u8> {
    (0..n as u32)
        .flat_map(|i| format!("row {} col {}; ", (i * 7919) % 1013, i % 37).into_bytes())
        .take(n)
        .collect()
}

#[test]
fn gpu_glze_blocks_equal_the_cpu_twin() {
    let Some(ctx) = context() else { return };
    let mut inputs = fixtures();
    inputs.push(("varied", varied(300_000)));
    inputs.push((
        "mixed",
        [
            varied(70_000),
            random(20_000, 9),
            vec![3; 50_000],
            text(60_000),
        ]
        .concat(),
    ));
    for level in [1, 3] {
        let encoder = GpuEncoder::new(&ctx, EncodeParams::for_level(level)).unwrap();
        let twin = GlzParams {
            lz: Params::for_level(level),
            independent_groups: None,
        };
        for chunk in [CHUNK, 1 << 16] {
            let options = GpuCompressOptions {
                codec: Codec::GlzE,
                chunk_size: chunk,
                level,
                ..GpuCompressOptions::default()
            };
            for (name, input) in &inputs {
                let blocks = encoder.encode_blocks(&ctx, input, &options).unwrap();
                for (i, (block, c)) in blocks.iter().zip(input.chunks(chunk as usize)).enumerate() {
                    let want = cpu::glze::encode_block(c, &twin);
                    let ok = match block {
                        EncodedBlock::Compressed(b) => *b == want,
                        EncodedBlock::Incompressible { size } => {
                            *size as usize == want.len() && want.len() >= c.len()
                        }
                    };
                    assert!(ok, "l{level} chunk {chunk} {name}: block {i} differs");
                }
            }
        }
    }
}

#[test]
fn gpu_glze_files_equal_cpu_files() {
    let Some(ctx) = context() else { return };
    let input = [varied(150_000), random(30_000, 4), vec![0; 40_000]].concat();
    let encoder = GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
    for (filters, gpu_filters) in [
        (FilterMode::None, gpu::encode::FilterMode::None),
        (FilterMode::Auto, gpu::encode::FilterMode::Auto),
        (FilterMode::Exhaustive, gpu::encode::FilterMode::Exhaustive),
    ] {
        for chunk in [CHUNK, 1 << 16] {
            let cpu_file = compress(&input, &glze(1, filters, chunk)).unwrap();
            let gpu_file = encoder
                .compress(
                    &ctx,
                    &input,
                    &GpuCompressOptions {
                        codec: Codec::GlzE,
                        chunk_size: chunk,
                        checksums: true,
                        filters: gpu_filters,
                        ..GpuCompressOptions::default()
                    },
                )
                .unwrap();
            assert!(gpu_file == cpu_file, "{filters:?} chunk {chunk}");
        }
    }
}
