//! GPU LZ4 compression (M3): valid, deterministic, and byte-identical to the
//! CPU twin encoder that runs the same algorithm.

mod common;

use common::{context, fixtures, random, text, CHUNK};
use cpu::container::{
    compress as cpu_compress, decompress as cpu_decompress, CompressOptions, Decoder,
    DecompressOptions, Encoder,
};
use cpu::lz4::encode::{encode_block, Params};
use gpu::decode::Lz4GpuDecoder;
use gpu::encode::{EncodeParams, EncodedBlock, GpuCompressOptions, GpuEncodeError, Lz4GpuEncoder};
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
        level: 1,
    }
}

#[test]
fn gpu_blocks_are_byte_identical_to_the_cpu_twin() {
    let Some(ctx) = context() else { return };
    let params = EncodeParams::default();
    let encoder = Lz4GpuEncoder::new(&ctx, params).unwrap();
    for (name, input) in fixtures() {
        let blocks = encoder.encode_blocks(&ctx, &input, CHUNK).unwrap();
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
        let encoder = Lz4GpuEncoder::new(&ctx, params).unwrap();
        let blocks = encoder.encode_blocks(&ctx, &input, 16_384).unwrap();
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
    let encoder = Lz4GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
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
    let encoder = Lz4GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
    let gpu_decoder = Lz4GpuDecoder::new(&ctx);
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
    let encoder = Lz4GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
    let input = [text(200_000), random(50_000, 4)].concat();
    let first = encoder.compress(&ctx, &input, &gpu_opts()).unwrap();
    assert!(!first.is_empty());
    for _ in 0..3 {
        assert!(encoder.compress(&ctx, &input, &gpu_opts()).unwrap() == first);
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
        Lz4GpuEncoder::new(&ctx, params),
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
        let encoder = Lz4GpuEncoder::new(&ctx, params).unwrap();
        let blocks = encoder.encode_blocks(&ctx, &input, 1 << chunk_shift).unwrap();
        prop_assert_eq!(blocks.len(), input.len().div_ceil(1 << chunk_shift));
        for (block, chunk) in blocks.iter().zip(input.chunks(1 << chunk_shift)) {
            prop_assert!(*block == expected(encode_block(chunk, &twin(params)), chunk.len()));
        }
    }
}
