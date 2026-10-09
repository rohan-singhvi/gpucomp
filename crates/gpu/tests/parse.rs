//! The GPU parse splits each chunk into segments walked in parallel and
//! stitched where the walks merge (see encode_parse_seg.wgsl). Its output must
//! equal the serial greedy parse, so these inputs target the stitching:
//! matches across segment boundaries, runs covering whole segments, periodic
//! data whose shifted parses merge late, and parses ending mid-segment.

mod common;

use common::{context, random, text};
use cpu::lz4::encode::{encode_block, Params};
use format::Codec;
use gpu::encode::{EncodeParams, EncodedBlock, GpuCompressOptions, GpuEncoder};

fn twin(p: EncodeParams) -> Params {
    Params {
        block: p.block as usize,
        hash_log: p.hash_log,
        probe_len: p.probe_len as usize,
    }
}

/// Inputs built around `seg`-byte segments.
fn adversarial(seg: usize) -> Vec<(String, Vec<u8>)> {
    let mut cases = Vec::new();
    // A run that starts just before a boundary and covers several segments.
    let mut run = random(seg - 3, 1);
    run.extend(vec![0u8; 3 * seg + 5]);
    run.extend(random(seg, 2));
    cases.push(("run across segments".to_string(), run));
    // Short periods: parses shifted by one byte follow the repeat for a long
    // time before merging.
    for period in [1usize, 2, 3, 5, 7, 13] {
        let unit = random(period, period as u64);
        let mut v = random(seg / 2, 3);
        v.extend(unit.iter().cycle().take(4 * seg).copied());
        v.extend(random(seg / 2, 4));
        cases.push((format!("period {period}"), v));
    }
    // Copies of earlier data placed to start at every offset near a boundary.
    let base = text(seg);
    for shift in 0..8usize {
        let mut v = base.clone();
        v.extend(random(seg - shift - 4, 10 + shift as u64));
        v.extend_from_slice(&base[..seg / 2]);
        v.extend(random(seg, 20 + shift as u64));
        cases.push((format!("copy at boundary - {}", shift + 4), v));
    }
    // Text everywhere: many short matches crossing every boundary.
    cases.push(("text".to_string(), text(8 * seg + 11)));
    cases
}

fn check(encoder: &GpuEncoder, ctx: &gpu::Context, params: EncodeParams, chunk: u32) {
    let seg = chunk as usize / 32;
    for (name, input) in adversarial(seg) {
        let options = GpuCompressOptions {
            chunk_size: chunk,
            ..GpuCompressOptions::default()
        };
        let blocks = encoder.encode_blocks(ctx, &input, &options).unwrap();
        for (i, (block, c)) in blocks.iter().zip(input.chunks(chunk as usize)).enumerate() {
            let want = encode_block(c, &twin(params));
            let ok = match block {
                EncodedBlock::Compressed(b) => *b == want,
                EncodedBlock::Incompressible { size } => *size as usize == want.len(),
            };
            assert!(ok, "chunk {chunk}, {name}: chunk {i} differs from the twin");
        }
    }
}

#[test]
fn segment_stitching_matches_the_serial_parse() {
    let Some(ctx) = context() else { return };
    let params = EncodeParams::default();
    let encoder = GpuEncoder::new(&ctx, params).unwrap();
    for chunk in [4096, 65_536] {
        check(&encoder, &ctx, params, chunk);
    }
}

#[test]
fn long_probes_also_match_the_serial_parse() {
    let Some(ctx) = context() else { return };
    let params = EncodeParams {
        probe_len: 32_767,
        ..EncodeParams::default()
    };
    let encoder = GpuEncoder::new(&ctx, params).unwrap();
    check(&encoder, &ctx, params, 16_384);
}

#[test]
fn short_tails_match_the_serial_parse() {
    let Some(ctx) = context() else { return };
    let params = EncodeParams::default();
    let encoder = GpuEncoder::new(&ctx, params).unwrap();
    let body = text(4096);
    for tail in 0..48usize {
        let mut input = body.clone();
        input.extend(text(tail + 4096)[4096..].iter());
        let options = GpuCompressOptions {
            chunk_size: 4096,
            ..GpuCompressOptions::default()
        };
        let file = encoder.compress(&ctx, &input, &options).unwrap();
        let cpu = cpu::container::compress(
            &input,
            &cpu::container::CompressOptions {
                chunk_size: 4096,
                encoder: cpu::container::Encoder::Greedy(twin(params)),
                ..cpu::container::CompressOptions::default()
            },
        )
        .unwrap();
        assert!(file == cpu, "tail {tail}");
    }
}

#[test]
fn glz_with_and_without_groups_matches_the_cpu() {
    let Some(ctx) = context() else { return };
    let params = EncodeParams::default();
    let encoder = GpuEncoder::new(&ctx, params).unwrap();
    for groups in [None, Some(4)] {
        let glz = cpu::glz::GlzParams {
            lz: twin(params),
            independent_groups: groups,
        };
        for (name, input) in adversarial(2048) {
            let options = GpuCompressOptions {
                codec: Codec::Glz,
                chunk_size: 65_536,
                independent_groups: groups,
                ..GpuCompressOptions::default()
            };
            let blocks = encoder.encode_blocks(&ctx, &input, &options).unwrap();
            for (i, (block, c)) in blocks.iter().zip(input.chunks(65_536)).enumerate() {
                let want = cpu::glz::encode_block(c, &glz);
                let ok = match block {
                    EncodedBlock::Compressed(b) => *b == want,
                    EncodedBlock::Incompressible { size } => *size as usize == want.len(),
                };
                assert!(ok, "{groups:?} {name}: chunk {i}");
            }
        }
    }
}

#[test]
fn probes_that_would_reach_the_visit_mark_are_rejected() {
    // Bit 31 of a match word marks positions a speculative walk visited, so
    // probe-capped lengths must stay below 1 << 15.
    let Some(ctx) = context() else { return };
    let params = EncodeParams {
        probe_len: 32_768,
        ..EncodeParams::default()
    };
    assert!(matches!(
        GpuEncoder::new(&ctx, params),
        Err(gpu::encode::GpuEncodeError::Unsupported(_))
    ));
}
