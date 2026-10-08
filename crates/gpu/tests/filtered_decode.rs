//! M7: files whose chunks are filtered (CPU encoders with `FilterMode::Auto`)
//! decode exactly on the GPU, for both codecs, every decode kernel, whole
//! files and ranges.

mod common;

use std::io::Cursor;

use common::{context, fixtures, numeric, CHUNK};
use cpu::container::{compress, CompressOptions, Encoder, FilterMode};
use format::{Codec, Filter, Index};
use gpu::decode::{DecodeKernel, DecoderConfig, GpuDecoder};
use proptest::prelude::*;

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

fn encoders() -> [(Codec, Encoder); 3] {
    [
        (Codec::Lz4, Encoder::Lz4Flex),
        (Codec::Lz4, Encoder::Greedy(Default::default())),
        (Codec::Glz, Encoder::Glz(Default::default())),
    ]
}

fn decoders(ctx: &gpu::Context) -> Vec<(&'static str, GpuDecoder)> {
    let naive = DecoderConfig {
        kernel: DecodeKernel::Naive,
        ..DecoderConfig::default()
    };
    vec![
        ("naive", GpuDecoder::with_config(ctx, naive).unwrap()),
        ("cooperative", GpuDecoder::new(ctx)),
    ]
}

fn inputs() -> Vec<(String, Vec<u8>)> {
    numeric()
        .into_iter()
        .chain(fixtures())
        .map(|(n, d)| (n.to_string(), d))
        .collect()
}

#[test]
fn auto_filtered_files_decode_exactly_whole_and_in_ranges() {
    let Some(ctx) = context() else { return };
    let decoders = decoders(&ctx);
    let mut filtered_chunks = 0;
    for (name, input) in inputs() {
        for (codec, encoder) in encoders() {
            for level in [1, 2] {
                let file = compress(&input, &auto(codec, encoder, level)).unwrap();
                filtered_chunks += Index::parse(&file)
                    .unwrap()
                    .chunks
                    .iter()
                    .filter(|c| c.filter != Filter::None)
                    .count();
                for (label, decoder) in &decoders {
                    let what = format!("{name} {encoder:?} level {level} {label}");
                    let out = decoder
                        .decompress(&ctx, &file, true)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert!(out == input, "{what}");
                    let n = input.len() as u64;
                    let c = u64::from(CHUNK);
                    let mut ranges = vec![
                        (0, n.min(10)),
                        (n / 3, n / 2),
                        (n.saturating_sub(5), n.min(5)),
                    ];
                    if n > c + 20 {
                        ranges.push((c - 10, 20));
                    }
                    for (offset, len) in ranges {
                        let got = decoder
                            .decompress_range(&ctx, &mut Cursor::new(&file), offset, len, true)
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
    assert!(
        filtered_chunks > 50,
        "only {filtered_chunks} filtered chunks"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn arbitrary_auto_filtered_files_decode_exactly(
        data in proptest::collection::vec(0u8..8, 0..30_000),
        level in 1u8..3,
        which in 0usize..3,
    ) {
        let Some(ctx) = context() else { return Ok(()) };
        let (codec, encoder) = encoders()[which];
        let file = compress(&data, &auto(codec, encoder, level)).unwrap();
        let out = GpuDecoder::new(&ctx).decompress(&ctx, &file, true).unwrap();
        prop_assert_eq!(&out, &data);
    }
}
