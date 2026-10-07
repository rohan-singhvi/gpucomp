//! GPU LZ4 compression (M3): valid, deterministic, and byte-identical to the
//! CPU twin encoder that runs the same algorithm.

mod common;

use common::{context, fixtures, random, text, CHUNK};
use cpu::container::{
    compress as cpu_compress, decompress as cpu_decompress, CompressOptions, Decoder,
    DecompressOptions, Encoder,
};
use cpu::lz4::encode::{encode_block, Params};
use format::Codec;
use gpu::decode::GpuDecoder;
use gpu::encode::{EncodeParams, EncodedBlock, GpuCompressOptions, GpuEncodeError, GpuEncoder};
use proptest::prelude::*;

fn twin(p: EncodeParams) -> Params {
    Params {
        block: p.block as usize,
        hash_log: p.hash_log,
        probe_len: p.probe_len as usize,
    }
}

/// What the GPU must report for a chunk the twin encodes as `twin_block`:
/// the block itself, or, if it doesn't shrink the chunk, just its size.
fn expected(twin_block: Vec<u8>, chunk_len: usize) -> EncodedBlock {
    if twin_block.len() >= chunk_len {
        EncodedBlock::Incompressible {
            size: twin_block.len() as u32,
        }
    } else {
        EncodedBlock::Compressed(twin_block)
    }
}

fn gpu_opts() -> GpuCompressOptions {
    GpuCompressOptions {
        chunk_size: CHUNK,
        checksums: true,
        ..GpuCompressOptions::default()
    }
}

fn glz_opts(groups: Option<u32>) -> GpuCompressOptions {
    GpuCompressOptions {
        codec: Codec::Glz,
        independent_groups: groups,
        ..gpu_opts()
    }
}

fn cpu_glz(groups: Option<u32>) -> cpu::glz::GlzParams {
    cpu::glz::GlzParams {
        lz: twin(EncodeParams::default()),
        independent_groups: groups,
    }
}

const GROUPS: [Option<u32>; 4] = [None, Some(1), Some(4), Some(64)];

#[test]
fn gpu_blocks_are_byte_identical_to_the_cpu_twin() {
    let Some(ctx) = context() else { return };
    let params = EncodeParams::default();
    let encoder = GpuEncoder::new(&ctx, params).unwrap();
    for (name, input) in fixtures() {
        let blocks = encoder.encode_blocks(&ctx, &input, &gpu_opts()).unwrap();
        let chunks: Vec<&[u8]> = input.chunks(CHUNK as usize).collect();
        if name == "random" {
            assert!(blocks
                .iter()
                .all(|b| matches!(b, EncodedBlock::Incompressible { .. })));
        }
        assert_eq!(blocks.len(), chunks.len(), "{name}");
        for (i, (block, chunk)) in blocks.iter().zip(chunks).enumerate() {
            assert!(
                *block == expected(encode_block(chunk, &twin(params)), chunk.len()),
                "{name}: chunk {i} differs from the CPU twin"
            );
        }
    }
}

#[test]
fn other_parameters_also_match_the_twin() {
    let Some(ctx) = context() else { return };
    let input = [text(20_000), random(5_000, 8), vec![9; 9_000]].concat();
    for params in [
        EncodeParams {
            block: 32,
            hash_log: 10,
            probe_len: 8,
        },
        EncodeParams {
            block: 128,
            hash_log: 11,
            probe_len: 32,
        },
        EncodeParams {
            block: 256,
            hash_log: 12,
            probe_len: 4,
        },
    ] {
        let encoder = GpuEncoder::new(&ctx, params).unwrap();
        let blocks = encoder
            .encode_blocks(
                &ctx,
                &input,
                &GpuCompressOptions {
                    chunk_size: 16_384,
                    ..gpu_opts()
                },
            )
            .unwrap();
        assert_eq!(blocks.len(), input.len().div_ceil(16_384), "{params:?}");
        for (i, (block, chunk)) in blocks.iter().zip(input.chunks(16_384)).enumerate() {
            assert!(
                *block == expected(encode_block(chunk, &twin(params)), chunk.len()),
                "{params:?} chunk {i}"
            );
        }
    }
}

#[test]
fn gpu_container_is_byte_identical_to_the_cpu_greedy_container() {
    let Some(ctx) = context() else { return };
    let encoder = GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
    for (name, input) in fixtures() {
        let gpu_file = encoder.compress(&ctx, &input, &gpu_opts()).unwrap();
        let cpu_file = cpu_compress(
            &input,
            &CompressOptions {
                chunk_size: CHUNK,
                encoder: Encoder::Greedy(twin(EncodeParams::default())),
                checksums: true,
                ..CompressOptions::default()
            },
        )
        .unwrap();
        assert!(
            gpu_file == cpu_file,
            "{name}: GPU and CPU containers differ"
        );
    }
}

#[test]
fn gpu_compressed_files_decode_with_every_decoder() {
    let Some(ctx) = context() else { return };
    let encoder = GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
    let gpu_decoder = GpuDecoder::new(&ctx);
    for (name, input) in fixtures() {
        let file = encoder.compress(&ctx, &input, &gpu_opts()).unwrap();
        for decoder in [Decoder::HandWritten, Decoder::Lz4Flex] {
            let options = DecompressOptions {
                decoder,
                verify: true,
            };
            assert!(
                cpu_decompress(&file, &options).unwrap() == input,
                "{name} {decoder:?}"
            );
        }
        assert!(
            gpu_decoder.decompress(&ctx, &file, true).unwrap() == input,
            "{name} GPU"
        );
    }
}

#[test]
fn gpu_compression_is_deterministic() {
    let Some(ctx) = context() else { return };
    let encoder = GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
    let input = [text(200_000), random(50_000, 4)].concat();
    let first = encoder.compress(&ctx, &input, &gpu_opts()).unwrap();
    assert!(!first.is_empty());
    for _ in 0..3 {
        assert!(encoder.compress(&ctx, &input, &gpu_opts()).unwrap() == first);
    }
}

#[test]
fn gpu_glz_blocks_are_byte_identical_to_the_cpu_glz_encoder() {
    let Some(ctx) = context() else { return };
    let encoder = GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
    for groups in GROUPS {
        for (name, input) in fixtures() {
            let blocks = encoder
                .encode_blocks(&ctx, &input, &glz_opts(groups))
                .unwrap();
            let chunks: Vec<&[u8]> = input.chunks(CHUNK as usize).collect();
            assert_eq!(blocks.len(), chunks.len(), "{name}");
            for (i, (block, chunk)) in blocks.iter().zip(chunks).enumerate() {
                let reference = cpu::glz::encode_block(chunk, &cpu_glz(groups));
                assert!(
                    *block == expected(reference, chunk.len()),
                    "{name} {groups:?}: chunk {i} differs from the CPU GLZ encoder"
                );
            }
        }
    }
}

#[test]
fn gpu_glz_containers_match_cpu_and_decode_everywhere() {
    let Some(ctx) = context() else { return };
    let encoder = GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
    let decoder = GpuDecoder::new(&ctx);
    let extra = [("long text", text(1 << 20))];
    for groups in GROUPS {
        for (name, input) in fixtures()
            .into_iter()
            .chain(extra.iter().map(|(n, d)| (*n, d.clone())))
        {
            let gpu_file = encoder.compress(&ctx, &input, &glz_opts(groups)).unwrap();
            let cpu_file = cpu_compress(
                &input,
                &CompressOptions {
                    codec: Codec::Glz,
                    chunk_size: CHUNK,
                    encoder: Encoder::Glz(cpu_glz(groups)),
                    checksums: true,
                    ..CompressOptions::default()
                },
            )
            .unwrap();
            assert!(
                gpu_file == cpu_file,
                "{name} {groups:?}: GPU and CPU GLZ files differ"
            );
            let options = DecompressOptions {
                decoder: Decoder::HandWritten,
                verify: true,
            };
            assert!(
                cpu_decompress(&gpu_file, &options).unwrap() == input,
                "{name} CPU"
            );
            assert!(
                decoder.decompress(&ctx, &gpu_file, true).unwrap() == input,
                "{name} GPU"
            );
        }
    }
}

#[test]
fn independent_groups_need_glz_and_a_sane_size() {
    let Some(ctx) = context() else { return };
    let encoder = GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
    for bad in [
        GpuCompressOptions {
            independent_groups: Some(8),
            ..gpu_opts()
        }, // LZ4
        glz_opts(Some(0)),
        glz_opts(Some(1000)),
    ] {
        assert!(
            matches!(
                encoder.compress(&ctx, b"abc", &bad),
                Err(GpuEncodeError::Unsupported(_))
            ),
            "{bad:?}"
        );
    }
}

#[test]
fn oversized_hash_tables_are_rejected() {
    let Some(ctx) = context() else { return };
    let params = EncodeParams {
        hash_log: 20, // 4 MiB of workgroup memory
        ..EncodeParams::default()
    };
    assert!(matches!(
        GpuEncoder::new(&ctx, params),
        Err(GpuEncodeError::Unsupported(_))
    ));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn arbitrary_inputs_match_the_twin(
        input in prop::collection::vec(prop_oneof![3 => 0u8..4, 1 => any::<u8>()], 1..50_000),
        chunk_shift in 12u32..=15,
    ) {
        let Some(ctx) = context() else { return Ok(()) };
        let params = EncodeParams::default();
        let encoder = GpuEncoder::new(&ctx, params).unwrap();
        let blocks = encoder.encode_blocks(&ctx, &input, &GpuCompressOptions { chunk_size: 1 << chunk_shift, ..gpu_opts() }).unwrap();
        prop_assert_eq!(blocks.len(), input.len().div_ceil(1 << chunk_shift));
        for (block, chunk) in blocks.iter().zip(input.chunks(1 << chunk_shift)) {
            prop_assert!(*block == expected(encode_block(chunk, &twin(params)), chunk.len()));
        }
    }
}
