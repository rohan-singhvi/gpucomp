//! M7: automatic per-chunk filter selection in the CPU container.

use std::io::Cursor;

use cpu::container::{
    compress, decompress, decompress_range, CompressOptions, Decoder, DecompressOptions, Encoder,
    FilterMode,
};
use cpu::filter;
use format::{Codec, Filter, Index};
use proptest::prelude::*;

const CHUNK: u32 = 4096;

/// Sorted u32s with small random gaps: delta's best case.
fn sorted_u32(count: usize) -> Vec<u8> {
    let mut s = 0x1234_5678u64;
    let mut v = 1_000_000u32;
    (0..count)
        .flat_map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            v = v.wrapping_add((s % 50) as u32);
            v.to_le_bytes()
        })
        .collect()
}

/// Smooth f32 xyz coordinates: shuffle's best case.
fn f32_points(count: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|i| {
            let t = i as f32 * 0.001;
            [t.sin() * 100.0, t.cos() * 100.0, t * 0.5]
                .into_iter()
                .flat_map(f32::to_le_bytes)
        })
        .collect()
}

/// u64 timestamps with small irregular steps.
fn timestamps(count: usize) -> Vec<u8> {
    let mut t = 1_700_000_000_000_000u64;
    (0..count as u64)
        .flat_map(|i| {
            t += 1000 + (i * 7919) % 13;
            t.to_le_bytes()
        })
        .collect()
}

fn random(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s as u8
        })
        .collect()
}

fn text(n: usize) -> Vec<u8> {
    b"the quick brown fox jumps over the lazy dog; "
        .iter()
        .copied()
        .cycle()
        .take(n)
        .collect()
}

fn inputs() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("empty", vec![]),
        ("three bytes", vec![1, 2, 3]),
        ("sorted u32", sorted_u32(5000)),
        ("f32 points", f32_points(3001)),
        ("timestamps", [timestamps(2000), vec![9; 3]].concat()),
        ("random", random(3 * CHUNK as usize + 5, 3)),
        ("zeros", vec![0; 3 * CHUNK as usize]),
        ("text", text(3 * CHUNK as usize + 1)),
    ]
}

fn glz() -> Encoder {
    Encoder::Glz(Default::default())
}

fn encoders() -> [(Codec, Encoder); 3] {
    [
        (Codec::Lz4, Encoder::Lz4Flex),
        (Codec::Lz4, Encoder::Greedy(Default::default())),
        (Codec::Glz, glz()),
    ]
}

fn auto(codec: Codec, encoder: Encoder, level: u8) -> CompressOptions {
    CompressOptions {
        codec,
        chunk_size: CHUNK,
        encoder,
        checksums: true,
        level,
        filters: FilterMode::Auto,
    }
}

fn decoders(codec: Codec) -> Vec<Decoder> {
    match codec {
        Codec::Glz => vec![Decoder::HandWritten],
        _ => vec![Decoder::HandWritten, Decoder::Lz4Flex],
    }
}

/// The encoded block for one chunk (`None` for the stored codec).
fn encode(codec: Codec, encoder: Encoder, chunk: &[u8]) -> Option<Vec<u8>> {
    match (codec, encoder) {
        (Codec::Stored, _) => None,
        (Codec::Lz4, Encoder::Lz4Flex) => Some(lz4_flex::block::compress(chunk)),
        (Codec::Lz4, Encoder::Greedy(p)) => Some(cpu::lz4::encode::encode_block(chunk, &p)),
        (Codec::Glz, Encoder::Glz(p)) => Some(cpu::glz::encode_block(chunk, &p)),
        _ => unreachable!(),
    }
}

#[test]
fn filters_default_to_none() {
    assert_eq!(CompressOptions::default().filters, FilterMode::None);
}

#[test]
fn auto_files_round_trip_with_every_decoder_and_range() {
    for (name, input) in inputs() {
        for (codec, encoder) in encoders() {
            for level in [1, 2] {
                let file = compress(&input, &auto(codec, encoder, level)).unwrap();
                for decoder in decoders(codec) {
                    let what = format!("{name} {encoder:?} level {level} {decoder:?}");
                    let options = DecompressOptions {
                        decoder,
                        verify: true,
                    };
                    let out = decompress(&file, &options).unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert!(out == input, "{what}");
                    let n = input.len() as u64;
                    for (offset, len) in [
                        (0, n.min(10)),
                        (n / 3, n / 2),
                        (n.saturating_sub(7), 7.min(n)),
                    ] {
                        let got = decompress_range(&mut Cursor::new(&file), offset, len, &options)
                            .unwrap_or_else(|e| panic!("{what} range {offset}+{len}: {e}"));
                        assert!(
                            got == input[offset as usize..(offset + len) as usize],
                            "{what} range {offset}+{len}"
                        );
                    }
                }
            }
        }
    }
}

fn exhaustive(codec: Codec, encoder: Encoder, level: u8) -> CompressOptions {
    CompressOptions {
        filters: FilterMode::Exhaustive,
        ..auto(codec, encoder, level)
    }
}

/// The exhaustive rule: smallest final payload over the level's candidates,
/// ties to the earlier candidate; a chunk no candidate shrinks is stored raw
/// and unfiltered.
#[test]
fn exhaustive_gives_each_chunk_the_smallest_candidate_with_deterministic_ties() {
    for (name, input) in inputs() {
        for (codec, encoder) in encoders() {
            for level in [1, 2] {
                let file = compress(&input, &exhaustive(codec, encoder, level)).unwrap();
                let index = Index::parse(&file).unwrap();
                for (i, chunk) in input.chunks(CHUNK as usize).enumerate() {
                    let what = format!("{name} {encoder:?} level {level} chunk {i}");
                    let mut best: Option<(usize, Filter)> = None;
                    for &f in filter::candidates(level) {
                        let size = encode(codec, encoder, &filter::forward(f, chunk))
                            .unwrap()
                            .len();
                        if best.is_none_or(|(b, _)| size < b) {
                            best = Some((size, f));
                        }
                    }
                    let (size, f) = best.unwrap();
                    let entry = index.chunks[i];
                    if size >= chunk.len() {
                        assert!(entry.stored, "{what}");
                        assert_eq!(entry.filter, Filter::None, "{what}");
                    } else {
                        assert!(!entry.stored, "{what}");
                        assert_eq!(
                            (entry.comp_size as usize, entry.filter),
                            (size, f),
                            "{what}"
                        );
                    }
                }
            }
        }
    }
}

fn plain_and_auto(input: &[u8], codec: Codec, encoder: Encoder) -> (Vec<u8>, Vec<u8>) {
    let none = CompressOptions {
        filters: FilterMode::None,
        ..auto(codec, encoder, 1)
    };
    (
        compress(input, &none).unwrap(),
        compress(input, &auto(codec, encoder, 1)).unwrap(),
    )
}

#[test]
fn numeric_data_is_filtered_and_shrinks() {
    for input in [
        sorted_u32(16 * CHUNK as usize / 4),
        f32_points(4 * CHUNK as usize / 12),
    ] {
        for (codec, encoder) in encoders() {
            let (plain, filtered) = plain_and_auto(&input, codec, encoder);
            let index = Index::parse(&filtered).unwrap();
            assert!(
                index.chunks.iter().all(|c| c.filter != Filter::None),
                "{encoder:?}"
            );
            // At least 15% smaller (the margin depends on the match-finding
            // block size; GLZ at block 128 gives ~20% on the f32 points).
            assert!(
                filtered.len() * 100 < plain.len() * 85,
                "{encoder:?}: {} vs {}",
                filtered.len(),
                plain.len()
            );
        }
    }
}

#[test]
fn an_arithmetic_progression_with_an_irregular_stride_picks_delta() {
    // Every byte plane looks random, so only delta (constant differences) helps.
    let input: Vec<u8> = (0..4 * CHUNK)
        .flat_map(|i| i.wrapping_mul(0x9E37_79B1).to_le_bytes())
        .collect();
    for (codec, encoder) in encoders() {
        let (plain, filtered) = plain_and_auto(&input, codec, encoder);
        let index = Index::parse(&filtered).unwrap();
        assert!(
            index
                .chunks
                .iter()
                .all(|c| c.filter == Filter::Delta { width: 4 }),
            "{encoder:?}"
        );
        assert!(filtered.len() * 20 < plain.len(), "{encoder:?}");
    }
}

#[test]
fn zeros_tie_and_go_to_none() {
    let file = compress(
        &vec![0; 2 * CHUNK as usize],
        &auto(Codec::Lz4, Encoder::Lz4Flex, 2),
    )
    .unwrap();
    let index = Index::parse(&file).unwrap();
    assert!(index
        .chunks
        .iter()
        .all(|c| c.filter == Filter::None && !c.stored));
}

#[test]
fn auto_is_deterministic_and_none_mode_is_unchanged() {
    let input = [f32_points(2000), text(5000)].concat();
    for (codec, encoder) in encoders() {
        let a = compress(&input, &auto(codec, encoder, 2)).unwrap();
        assert_eq!(a, compress(&input, &auto(codec, encoder, 2)).unwrap());
        let none = CompressOptions {
            filters: FilterMode::None,
            ..auto(codec, encoder, 2)
        };
        let index = Index::parse(&compress(&input, &none).unwrap()).unwrap();
        assert!(index.chunks.iter().all(|c| c.filter == Filter::None));
    }
}

#[test]
fn stored_codec_ignores_auto() {
    let input = sorted_u32(3000);
    let file = compress(&input, &auto(Codec::Stored, Encoder::Lz4Flex, 2)).unwrap();
    let index = Index::parse(&file).unwrap();
    assert!(index
        .chunks
        .iter()
        .all(|c| c.stored && c.filter == Filter::None));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn arbitrary_inputs_round_trip_with_auto_filters(
        data in proptest::collection::vec(0u8..4, 0..20_000),
        level in 1u8..3,
    ) {
        for (codec, encoder) in encoders() {
            let file = compress(&data, &auto(codec, encoder, level)).unwrap();
            let out = decompress(&file, &DecompressOptions::default()).unwrap();
            prop_assert_eq!(&out, &data);
        }
    }
}

// ---- Auto: sampled trial encoding ----

const BIG_CHUNK: u32 = 64 << 10;

fn with_chunk(options: CompressOptions, chunk_size: u32) -> CompressOptions {
    CompressOptions {
        chunk_size,
        ..options
    }
}

#[test]
fn the_sample_is_an_eighth_of_the_chunk_but_at_least_4_kib() {
    assert_eq!(filter::sample_len(4096), 4096);
    assert_eq!(filter::sample_len(16 << 10), 4096);
    assert_eq!(filter::sample_len(64 << 10), 8192);
    assert_eq!(filter::sample_len(1 << 20), 128 << 10);
}

/// Auto's rule: candidates compete on the chunk's leading sample (filtered as
/// a buffer of its own); the winner encodes the whole chunk, which is stored
/// raw and unfiltered if that doesn't shrink it.
#[test]
fn auto_chooses_by_the_leading_sample() {
    let inputs = [
        ("numeric", [sorted_u32(40_000), f32_points(20_000)].concat()),
        ("text", text(300_000)),
        ("random", random(150_000, 7)),
    ];
    for (name, input) in inputs {
        for (codec, encoder) in encoders() {
            for level in [1, 2] {
                let options = with_chunk(auto(codec, encoder, level), BIG_CHUNK);
                let index = Index::parse(&compress(&input, &options).unwrap()).unwrap();
                for (i, chunk) in input.chunks(BIG_CHUNK as usize).enumerate() {
                    let what = format!("{name} {encoder:?} level {level} chunk {i}");
                    let sample =
                        &chunk[..(filter::sample_len(BIG_CHUNK) as usize).min(chunk.len())];
                    let mut best: Option<(usize, Filter)> = None;
                    for &f in filter::candidates(level) {
                        let size = encode(codec, encoder, &filter::forward(f, sample))
                            .unwrap()
                            .len();
                        if best.is_none_or(|(b, _)| size < b) {
                            best = Some((size, f));
                        }
                    }
                    let f = best.unwrap().1;
                    let block = encode(codec, encoder, &filter::forward(f, chunk)).unwrap();
                    let entry = index.chunks[i];
                    if block.len() >= chunk.len() {
                        assert!(entry.stored && entry.filter == Filter::None, "{what}");
                    } else {
                        assert!(!entry.stored, "{what}");
                        assert_eq!(
                            (entry.comp_size as usize, entry.filter),
                            (block.len(), f),
                            "{what}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn auto_and_exhaustive_disagree_when_the_sample_misleads() {
    // A chunk whose leading 8 KiB sample is text (no filter wins there) but
    // which is mostly sorted integers (shuffle wins the whole chunk).
    let chunk = [text(8192), sorted_u32((BIG_CHUNK as usize - 8192) / 4)].concat();
    let (codec, encoder) = encoders()[1];
    let filter_of = |options| {
        let file = compress(&chunk, &with_chunk(options, BIG_CHUNK)).unwrap();
        Index::parse(&file).unwrap().chunks[0].filter
    };
    assert_eq!(filter_of(auto(codec, encoder, 1)), Filter::None);
    assert_ne!(filter_of(exhaustive(codec, encoder, 1)), Filter::None);
}

#[test]
fn auto_equals_exhaustive_when_the_sample_is_the_whole_chunk() {
    for (name, input) in inputs() {
        for (codec, encoder) in encoders() {
            assert_eq!(
                compress(&input, &auto(codec, encoder, 2)).unwrap(),
                compress(&input, &exhaustive(codec, encoder, 2)).unwrap(),
                "{name} {encoder:?}"
            );
        }
    }
}
