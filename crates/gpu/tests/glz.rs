//! GPU GLZ decompression (M6): byte-identical output, and the same error
//! status as the CPU reference decoder for malformed blocks.

mod common;

use std::io::Cursor;

use common::{context, fixtures, random, text, CHUNK};
use cpu::container::{compress, CompressOptions, Encoder};
use cpu::glz::{GlzError, GlzParams};
use format::{ChunkEntry, Codec, Filter, Header, Index};
use gpu::decode::{ChunkStatus, GpuDecodeError, GpuDecoder};
use proptest::prelude::*;

fn glz(groups: Option<u32>, chunk_size: u32) -> CompressOptions {
    CompressOptions {
        codec: Codec::Glz,
        chunk_size,
        encoder: Encoder::Glz(GlzParams {
            independent_groups: groups,
            ..Default::default()
        }),
        checksums: true,
        ..CompressOptions::default()
    }
}

#[test]
fn every_glz_file_decodes_exactly_on_the_gpu() {
    let Some(ctx) = context() else { return };
    let decoder = GpuDecoder::new(&ctx);
    let extra = [
        ("long text", text(1 << 20)),
        (
            "zeros, 1 MiB chunks (wide extension values)",
            vec![0u8; 3 << 20],
        ),
    ];
    for groups in [None, Some(4), Some(32)] {
        for (name, input) in fixtures()
            .into_iter()
            .chain(extra.iter().map(|(n, d)| (*n, d.clone())))
        {
            let chunk = if name.contains("1 MiB") {
                1 << 20
            } else {
                CHUNK
            };
            let file = compress(&input, &glz(groups, chunk)).unwrap();
            let out = decoder
                .decompress(&ctx, &file, true)
                .unwrap_or_else(|e| panic!("{name} {groups:?}: {e}"));
            assert!(out == input, "{name} {groups:?}: output differs");
        }
    }
}

#[test]
fn glz_range_reads_on_the_gpu() {
    let Some(ctx) = context() else { return };
    let decoder = GpuDecoder::new(&ctx);
    let input = text(10 * CHUNK as usize + 99);
    let file = compress(&input, &glz(None, CHUNK)).unwrap();
    let c = u64::from(CHUNK);
    for (offset, len) in [
        (0, 10),
        (c - 5, 10),
        (3 * c + 7, 2 * c),
        (input.len() as u64 - 99, 99),
    ] {
        let got = decoder
            .decompress_range(&ctx, &mut Cursor::new(&file), offset, len, true)
            .unwrap();
        assert!(
            got == input[offset as usize..(offset + len) as usize],
            "{offset}+{len}"
        );
    }
}

/// A one-chunk GLZ file whose payload is `payload`, decoding to `n` bytes.
fn single_chunk_file(payload: &[u8], n: u32) -> Vec<u8> {
    let index = Index {
        header: Header {
            codec: Codec::Glz,
            chunk_size: CHUNK,
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
    file.resize(file.len().div_ceil(4) * 4, 0);
    file
}

fn raw_block(tokens: &[u8], offsets: &[u16], ext: &[u16], literals: &[u8]) -> Vec<u8> {
    let mut out = (tokens.len() as u32).to_le_bytes().to_vec();
    out.extend_from_slice(&(ext.len() as u32).to_le_bytes());
    out.extend_from_slice(tokens);
    out.resize(out.len().div_ceil(4) * 4, 0);
    for array in [offsets, ext] {
        for v in array {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.resize(out.len().div_ceil(4) * 4, 0);
    }
    out.extend_from_slice(literals);
    out
}

fn expected_status(e: GlzError) -> ChunkStatus {
    match e {
        GlzError::Truncated => ChunkStatus::Truncated,
        GlzError::BadSequence => ChunkStatus::BadSequence,
        GlzError::ZeroOffset => ChunkStatus::ZeroOffset,
        GlzError::OffsetBeforeStart => ChunkStatus::OffsetBeforeStart,
        GlzError::OutputOverflow => ChunkStatus::OutputOverflow,
        GlzError::SizeMismatch => ChunkStatus::SizeMismatch,
        GlzError::Stream(_) => unreachable!("GLZ blocks have no entropy-coded streams"),
    }
}

#[test]
fn malformed_glz_blocks_report_the_same_error_as_the_cpu_decoder() {
    let Some(ctx) = context() else { return };
    let decoder = GpuDecoder::new(&ctx);
    let mut ext_mismatch = raw_block(&[0xF0], &[0], &[0], &[0; 15]);
    ext_mismatch[4] = 2;
    let mut cut = raw_block(&[0x10, 0x00], &[1, 0], &[], b"a");
    cut.truncate(14);
    // Errors in two sequences (zero offset, then offset before start): the
    // earlier one must win.
    let two_errors = raw_block(&[0x10, 0x10, 0x00], &[0, 9, 0], &[], b"ab");
    let cases: Vec<(Vec<u8>, u32)> = vec![
        (vec![], 4),
        (vec![1, 0, 0], 4),
        (raw_block(&[], &[], &[], &[]), 4),
        (cut, 5),
        (ext_mismatch, 15),
        (raw_block(&[0x30], &[0], &[], b"ab"), 3),
        (raw_block(&[0x30], &[0], &[], b"abc"), 2),
        (raw_block(&[0x11], &[1], &[], b"a"), 6),
        (raw_block(&[0x10], &[1], &[], b"a"), 1),
        (raw_block(&[0x10, 0x00], &[0, 0], &[], b"a"), 5),
        (raw_block(&[0x10, 0x00], &[2, 0], &[], b"a"), 5),
        (raw_block(&[0x10, 0x00], &[1, 0], &[], b"a"), 4),
        (raw_block(&[0x20], &[0], &[], b"ab"), 3),
        (raw_block(&[0x20], &[0], &[], b"abc"), 2),
        (raw_block(&[0x00, 0x00, 0x40], &[1, 1, 0], &[], b"abcd"), 4),
        (two_errors, 10),
    ];
    for (payload, n) in cases {
        let mut reference = vec![0; n as usize];
        let cpu_error = cpu::glz::decode_block(&payload, &mut reference).unwrap_err();
        match decoder.decompress(&ctx, &single_chunk_file(&payload, n), false) {
            Err(GpuDecodeError::Chunk { chunk: 0, status }) => {
                assert_eq!(status, expected_status(cpu_error), "{payload:?} -> {n}");
            }
            other => panic!("{payload:?} -> {n}: expected {cpu_error:?}, got {other:?}"),
        }
    }
}

#[test]
fn dependent_chains_resolve_in_rounds() {
    let Some(ctx) = context() else { return };
    // Each match copies the previous match's output: worst case for rounds.
    let mut input = b"wxyz".to_vec();
    for _ in 0..300 {
        let n = input.len();
        input.extend_from_within(n - 4..n);
        input.push(b'.');
    }
    input.extend(random(64, 1));
    let file = compress(&input, &glz(None, CHUNK)).unwrap();
    assert_eq!(
        GpuDecoder::new(&ctx).decompress(&ctx, &file, true).unwrap(),
        input
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn arbitrary_glz_inputs_round_trip_on_the_gpu(
        input in prop::collection::vec(prop_oneof![3 => 0u8..4, 1 => any::<u8>()], 0..60_000),
        chunk_shift in 12u32..=16,
        groups in prop_oneof![Just(None), (1u32..=64).prop_map(Some)],
    ) {
        let Some(ctx) = context() else { return Ok(()) };
        let file = compress(&input, &glz(groups, 1 << chunk_shift)).unwrap();
        prop_assert!(GpuDecoder::new(&ctx).decompress(&ctx, &file, true).unwrap() == input);
    }

    #[test]
    fn corrupted_glz_files_fail_cleanly_on_the_gpu(
        flips in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..8),
    ) {
        let Some(ctx) = context() else { return Ok(()) };
        let input = text(6 * CHUNK as usize);
        let mut file = compress(&input, &glz(None, CHUNK)).unwrap();
        for (at, byte) in flips {
            let i = at.index(file.len());
            file[i] ^= byte | 1;
        }
        if let Ok(out) = GpuDecoder::new(&ctx).decompress(&ctx, &file, true) {
            prop_assert!(out == input);
        }
    }
}
