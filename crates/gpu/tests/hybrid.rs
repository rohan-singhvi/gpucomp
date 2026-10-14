//! CPU+GPU compression: every chunk encodes to the same bytes on the GPU and
//! with its CPU twin, so a file compressed by both equals the GPU encoder's,
//! whatever the split.

mod common;

use common::{context, fixtures, random, text, CHUNK};
use format::Codec;
use gpu::decode::GpuDecoder;
use gpu::encode::{EncodeParams, FilterMode, GpuCompressOptions, GpuEncoder};
use gpu::hybrid::{
    compress, compress_parts, compress_parts_with, compress_static, compress_static_stream,
    decompress, decompress_parts, twin_options, Device, SplitInput,
};
use gpu::{Context, ContextOptions};
use std::io::{Cursor, Seek, SeekFrom};

/// 18 chunks: text, random bytes, zeros, and a short last chunk.
fn input() -> Vec<u8> {
    [
        text(9 * 4096),
        random(3 * 4096, 1),
        vec![0; 5 * 4096],
        text(77),
    ]
    .concat()
}

const SPLITS: [&[(Device, u32)]; 4] = [
    &[(Device::Gpu, 18)],
    &[(Device::Cpu, 18)],
    &[
        (Device::Gpu, 5),
        (Device::Cpu, 4),
        (Device::Gpu, 1),
        (Device::Cpu, 8),
    ],
    &[(Device::Cpu, 1), (Device::Gpu, 17)],
];

#[test]
fn hybrid_files_equal_the_gpu_encoders_for_any_split() {
    let Some(ctx) = context() else { return };
    let input = input();
    for level in 1..=3 {
        let encoder = GpuEncoder::new(&ctx, EncodeParams::for_level(level)).unwrap();
        for codec in [Codec::Lz4, Codec::Glz, Codec::GlzE] {
            for (filters, checksums) in [
                (FilterMode::None, false),
                (FilterMode::Auto, true),
                (FilterMode::Exhaustive, false),
            ] {
                let options = GpuCompressOptions {
                    codec,
                    chunk_size: CHUNK,
                    checksums,
                    level,
                    filters,
                    independent_groups: None,
                };
                let expected = encoder.compress(&ctx, &input, &options).unwrap();
                for split in SPLITS {
                    let file = compress_parts(&ctx, &encoder, &input, &options, split).unwrap();
                    assert!(file == expected, "{options:?}, split {split:?}");
                }
                let file = compress(&ctx, &encoder, &input, &options).unwrap();
                assert!(file == expected, "{options:?}, scheduled");
            }
        }
    }
}

#[test]
fn hybrid_handles_groups_fixtures_and_empty_input() {
    let Some(ctx) = context() else { return };
    let encoder = GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
    let groups = GpuCompressOptions {
        codec: Codec::Glz,
        chunk_size: CHUNK,
        independent_groups: Some(4),
        ..GpuCompressOptions::default()
    };
    let input = input();
    let expected = encoder.compress(&ctx, &input, &groups).unwrap();
    assert!(compress_parts(&ctx, &encoder, &input, &groups, SPLITS[2]).unwrap() == expected);
    let options = GpuCompressOptions {
        chunk_size: CHUNK,
        ..GpuCompressOptions::default()
    };
    for (name, input) in fixtures() {
        assert!(
            compress(&ctx, &encoder, &input, &options).unwrap()
                == encoder.compress(&ctx, &input, &options).unwrap(),
            "{name}"
        );
    }
}

#[test]
fn a_split_must_cover_the_input_exactly() {
    let Some(ctx) = context() else { return };
    let encoder = GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
    let options = GpuCompressOptions {
        chunk_size: CHUNK,
        ..GpuCompressOptions::default()
    };
    let input = input();
    for split in [
        &[(Device::Gpu, 17)][..],
        &[(Device::Cpu, 10), (Device::Gpu, 9)],
    ] {
        assert!(compress_parts(&ctx, &encoder, &input, &options, split).is_err());
    }
}

#[test]
fn hybrid_decompression_restores_the_input_for_any_split() {
    let Some(ctx) = context() else { return };
    let encoder = GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
    let decoder = GpuDecoder::new(&ctx);
    let input = input();
    for codec in [Codec::Lz4, Codec::Glz, Codec::GlzE] {
        for (filters, checksums) in [(FilterMode::None, false), (FilterMode::Auto, true)] {
            let options = GpuCompressOptions {
                codec,
                chunk_size: CHUNK,
                checksums,
                filters,
                ..GpuCompressOptions::default()
            };
            let file = encoder.compress(&ctx, &input, &options).unwrap();
            for split in SPLITS {
                let out = decompress_parts(&ctx, &decoder, &file, checksums, split).unwrap();
                assert!(out == input, "{options:?}, split {split:?}");
            }
            assert!(decompress(&ctx, &decoder, &file, checksums).unwrap() == input);
        }
    }
    let empty = encoder
        .compress(&ctx, &[], &GpuCompressOptions::default())
        .unwrap();
    assert!(decompress(&ctx, &decoder, &empty, false)
        .unwrap()
        .is_empty());
}

#[test]
fn hybrid_decompression_reports_bad_splits_and_corrupt_chunks() {
    let Some(ctx) = context() else { return };
    let encoder = GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
    let decoder = GpuDecoder::new(&ctx);
    let options = GpuCompressOptions {
        chunk_size: CHUNK,
        checksums: true,
        ..GpuCompressOptions::default()
    };
    let mut file = encoder.compress(&ctx, &input(), &options).unwrap();
    assert!(decompress_parts(&ctx, &decoder, &file, true, &[(Device::Gpu, 17)]).is_err());
    // Corrupt the last chunk's payload: whichever device decodes it fails.
    let n = file.len();
    file[n - 8] ^= 0x55;
    for split in [&[(Device::Gpu, 18)][..], &[(Device::Cpu, 18)]] {
        assert!(
            decompress_parts(&ctx, &decoder, &file, true, split).is_err(),
            "{split:?}"
        );
    }
    assert!(decompress(&ctx, &decoder, &file, true).is_err());
}

#[test]
fn a_fixed_share_splits_the_chunks_deterministically() {
    let Some(ctx) = context() else { return };
    let encoder = GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
    let options = GpuCompressOptions {
        chunk_size: CHUNK,
        checksums: true,
        ..GpuCompressOptions::default()
    };
    let input = input(); // 18 chunks
    let flex = cpu::container::CompressOptions {
        chunk_size: CHUNK,
        checksums: true,
        ..Default::default()
    };
    // Share 1: the GPU's file; share 0: the CPU encoder's.
    let gpu_file = encoder.compress(&ctx, &input, &options).unwrap();
    let cpu_file = cpu::container::compress(&input, &flex).unwrap();
    assert!(compress_static(&ctx, &encoder, &input, &options, 1.0, &flex).unwrap() == gpu_file);
    assert!(compress_static(&ctx, &encoder, &input, &options, 0.0, &flex).unwrap() == cpu_file);
    // Share 0.5: lz4_flex takes the first 9 chunks, the GPU the last 9.
    let file = compress_static(&ctx, &encoder, &input, &options, 0.5, &flex).unwrap();
    let expected = compress_parts_with(
        &ctx,
        &encoder,
        &input,
        &options,
        &[(Device::Cpu, 9), (Device::Gpu, 9)],
        &flex,
    )
    .unwrap();
    assert!(file == expected);
    assert!(file != gpu_file && file != cpu_file);
    assert!(compress_static(&ctx, &encoder, &input, &options, 0.5, &flex).unwrap() == file);
    let decoder = GpuDecoder::new(&ctx);
    assert!(decompress(&ctx, &decoder, &file, true).unwrap() == input);
    assert!(compress_static(&ctx, &encoder, &input, &options, 1.5, &flex).is_err());
}

/// A new context and level-1 encoder, as a one-shot command creates them.
fn new_gpu() -> Result<(Context, GpuEncoder), gpu::encode::GpuEncodeError> {
    let ctx = Context::new(&ContextOptions::default())?;
    let encoder = GpuEncoder::new(&ctx, EncodeParams::default())?;
    Ok((ctx, encoder))
}

#[test]
fn a_streamed_fixed_split_writes_the_in_memory_file() {
    let Some(ctx) = context() else { return };
    let encoder = GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
    let lz4 = GpuCompressOptions {
        chunk_size: CHUNK,
        checksums: true,
        ..GpuCompressOptions::default()
    };
    let flex = cpu::container::CompressOptions {
        encoder: cpu::container::Encoder::Lz4Flex,
        ..Default::default()
    };
    let glze = GpuCompressOptions {
        codec: Codec::GlzE,
        ..lz4
    };
    let twin = twin_options(&glze, encoder.params());
    for (options, cpu) in [(lz4, flex), (glze, twin)] {
        for input in [input(), text(77), Vec::new()] {
            for share in [0.0, 0.5, 1.0] {
                let expected =
                    compress_static(&ctx, &encoder, &input, &options, share, &cpu).unwrap();
                // The file starts where the writer is.
                let mut out = Cursor::new(b"prefix".to_vec());
                out.seek(SeekFrom::End(0)).unwrap();
                let split = SplitInput {
                    head: &mut Cursor::new(&input),
                    tail: Cursor::new(&input),
                    len: input.len() as u64,
                };
                let n = compress_static_stream(split, &mut out, &options, share, &cpu, new_gpu)
                    .unwrap();
                let what = format!("{:?} {} bytes, share {share}", options.codec, input.len());
                assert_eq!(n, expected.len() as u64, "{what}");
                assert_eq!(out.stream_position().unwrap(), 6 + n, "{what}");
                assert!(out.into_inner()[6..] == expected[..], "{what}");
            }
        }
    }
}

#[test]
fn a_streamed_split_starts_the_gpu_only_for_a_share() {
    let input = input();
    let options = GpuCompressOptions {
        chunk_size: CHUNK,
        ..GpuCompressOptions::default()
    };
    let flex = cpu::container::CompressOptions {
        encoder: cpu::container::Encoder::Lz4Flex,
        ..Default::default()
    };
    let no_gpu = || Err(gpu::encode::GpuEncodeError::Unsupported("no GPU".into()));
    let run = |share, gpu: fn() -> Result<(Context, GpuEncoder), gpu::encode::GpuEncodeError>| {
        let split = SplitInput {
            head: &mut Cursor::new(&input),
            tail: Cursor::new(&input),
            len: input.len() as u64,
        };
        compress_static_stream(
            split,
            &mut Cursor::new(Vec::new()),
            &options,
            share,
            &flex,
            gpu,
        )
    };
    // All on the CPU: no GPU needed, and the file is the CPU's.
    assert!(run(0.0, no_gpu).is_ok());
    // A GPU that can't start fails the whole file.
    assert!(run(0.5, no_gpu).is_err());
    assert!(matches!(
        run(1.5, no_gpu),
        Err(gpu::encode::GpuEncodeError::BadShare(_))
    ));
}
