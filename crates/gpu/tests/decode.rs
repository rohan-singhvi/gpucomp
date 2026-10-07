//! GPU LZ4 decompression must be byte-identical to the original (M2).

mod common;

use std::io::Cursor;

use common::{context, fixtures, random, text, CHUNK};
use cpu::container::{compress, CompressOptions, Encoder};
use cpu::lz4::decode::DecodeError;
use format::{ChunkEntry, Codec, Filter, Header, Index};
use gpu::decode::{ChunkStatus, GpuDecodeError, Lz4GpuDecoder};
use proptest::prelude::*;

const ENCODERS: [Encoder; 2] = [
    Encoder::Lz4Flex,
    Encoder::Greedy(cpu::lz4::encode::Params {
        block: 64,
        hash_log: 12,
        probe_len: 16,
    }),
];

fn opts(encoder: Encoder) -> CompressOptions {
    CompressOptions {
        chunk_size: CHUNK,
        encoder,
        checksums: true,
        ..CompressOptions::default()
    }
}

#[test]
fn every_cpu_encoding_decodes_exactly_on_the_gpu() {
    let Some(ctx) = context() else { return };
    let decoder = Lz4GpuDecoder::new(&ctx);
    for (name, input) in fixtures() {
        for encoder in ENCODERS {
            let file = compress(&input, &opts(encoder)).unwrap();
            let out = decoder
                .decompress(&ctx, &file, true)
                .unwrap_or_else(|e| panic!("{name} {encoder:?}: {e}"));
            assert!(out == input, "{name} {encoder:?}: output differs");
        }
    }
}

#[test]
fn stored_codec_files_decode_on_the_gpu() {
    let Some(ctx) = context() else { return };
    let input = text(3 * CHUNK as usize + 5);
    let options = CompressOptions {
        codec: Codec::Stored,
        ..opts(Encoder::Lz4Flex)
    };
    let file = compress(&input, &options).unwrap();
    assert_eq!(
        Lz4GpuDecoder::new(&ctx)
            .decompress(&ctx, &file, true)
            .unwrap(),
        input
    );
}

#[test]
fn gpu_range_reads_match_slices_of_the_original() {
    let Some(ctx) = context() else { return };
    let decoder = Lz4GpuDecoder::new(&ctx);
    let input = text(10 * CHUNK as usize + 1234);
    let file = compress(&input, &opts(ENCODERS[1])).unwrap();
    let (c, n) = (u64::from(CHUNK), input.len() as u64);
    for (offset, len) in [
        (0, 10),
        (c + 5, 100),
        (c - 50, 100),
        (c - 1, 2 * c + 2),
        (n - 1234, 1234),
        (n - 10, 10),
        (0, n),
        (3 * c, 0),
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

#[test]
fn checksum_mismatch_is_reported() {
    let Some(ctx) = context() else { return };
    let mut file = compress(&random(2 * CHUNK as usize, 3), &opts(Encoder::Lz4Flex)).unwrap();
    let data = Index::parse(&file).unwrap().data_offset() as usize;
    file[data + 4097] ^= 0xFF; // inside stored chunk 1
    assert!(matches!(
        Lz4GpuDecoder::new(&ctx).decompress(&ctx, &file, true),
        Err(GpuDecodeError::Checksum { chunk: 1 })
    ));
}

#[test]
fn filtered_chunks_are_rejected() {
    let Some(ctx) = context() else { return };
    let file = compress(&text(10_000), &opts(Encoder::Lz4Flex)).unwrap();
    let mut index = Index::parse(&file).unwrap();
    index.chunks[2].filter = Filter::Delta { width: 2 };
    let rest = &file[index.data_offset() as usize..];
    let patched = [index.to_bytes(), rest.to_vec()].concat();
    assert!(matches!(
        Lz4GpuDecoder::new(&ctx).decompress(&ctx, &patched, true),
        Err(GpuDecodeError::UnsupportedFilter { chunk: 2 })
    ));
}

/// A one-chunk LZ4 file whose payload is `payload`, decoding to `n` bytes.
fn single_chunk_file(payload: &[u8], n: u32) -> Vec<u8> {
    let index = Index {
        header: Header {
            codec: Codec::Lz4,
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

fn expected_status(e: DecodeError) -> ChunkStatus {
    match e {
        DecodeError::Truncated => ChunkStatus::Truncated,
        DecodeError::ZeroOffset => ChunkStatus::ZeroOffset,
        DecodeError::OffsetBeforeStart => ChunkStatus::OffsetBeforeStart,
        DecodeError::OutputOverflow => ChunkStatus::OutputOverflow,
        DecodeError::SizeMismatch { .. } => ChunkStatus::SizeMismatch,
    }
}

#[test]
fn malformed_payloads_report_the_same_error_as_the_cpu_decoder() {
    let Some(ctx) = context() else { return };
    let decoder = Lz4GpuDecoder::new(&ctx);
    let cases: [(&[u8], u32); 9] = [
        (&[], 4),                       // truncated: no token
        (b"\x50hel", 5),                // truncated literals
        (&[0x14, b'a', 1], 9),          // truncated offset
        (&[0xF0, 255], 300),            // truncated length continuation
        (&[0x14, b'a', 0, 0, 0x00], 9), // zero offset
        (&[0x14, b'a', 2, 0, 0x00], 9), // offset before start
        (b"\x50hello", 4),              // literals overflow
        (&[0x15, b'a', 1, 0, 0x00], 5), // match overflow
        (b"\x50hello", 6),              // output too short
    ];
    for (payload, n) in cases {
        let mut reference = vec![0; n as usize];
        let cpu_error = cpu::lz4::decode::decode_block(payload, &mut reference).unwrap_err();
        let file = single_chunk_file(payload, n);
        match decoder.decompress(&ctx, &file, false) {
            Err(GpuDecodeError::Chunk { chunk: 0, status }) => {
                assert_eq!(status, expected_status(cpu_error), "{payload:?} -> {n}");
            }
            other => panic!("{payload:?} -> {n}: expected a chunk error, got {other:?}"),
        }
    }
}

#[test]
fn hand_made_overlapping_matches_decode_on_the_gpu() {
    let Some(ctx) = context() else { return };
    let decoder = Lz4GpuDecoder::new(&ctx);
    for (payload, expected) in [
        (&[0x15u8, b'a', 1, 0, 0x00][..], b"aaaaaaaaaa".to_vec()),
        (&[0x22, b'a', b'b', 2, 0, 0x00], b"abababab".to_vec()),
        (&[0x1F, b'z', 1, 0, 255, 1, 0x00], vec![b'z'; 276]),
    ] {
        let file = single_chunk_file(payload, expected.len() as u32);
        assert_eq!(decoder.decompress(&ctx, &file, false).unwrap(), expected);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn arbitrary_inputs_round_trip_on_the_gpu(
        input in prop::collection::vec(prop_oneof![3 => 0u8..4, 1 => any::<u8>()], 0..60_000),
        chunk_shift in 12u32..=16,
        greedy: bool,
    ) {
        let Some(ctx) = context() else { return Ok(()) };
        let options = CompressOptions {
            chunk_size: 1 << chunk_shift,
            ..opts(ENCODERS[usize::from(greedy)])
        };
        let file = compress(&input, &options).unwrap();
        let out = Lz4GpuDecoder::new(&ctx).decompress(&ctx, &file, true).unwrap();
        prop_assert!(out == input);
    }

    #[test]
    fn corrupted_files_fail_cleanly_on_the_gpu(
        flips in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..8),
    ) {
        let Some(ctx) = context() else { return Ok(()) };
        let input = text(6 * CHUNK as usize);
        let mut file = compress(&input, &opts(ENCODERS[1])).unwrap();
        for (at, byte) in flips {
            let i = at.index(file.len());
            file[i] ^= byte | 1;
        }
        if let Ok(out) = Lz4GpuDecoder::new(&ctx).decompress(&ctx, &file, true) {
            prop_assert!(out == input);
        }
    }
}
