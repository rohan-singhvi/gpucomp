//! M4: cross-path validation matrix. Every encoder × every decoder, on every
//! available GPU backend, at several chunk sizes, must reproduce the input
//! exactly, for whole files and byte ranges. Also runs the malformed-input
//! suite against the GPU decoder on each backend.

mod common;

use std::io::Cursor;

use common::{canterbury, contexts, fixtures, random, text};
use cpu::container::{
    compress, decompress, decompress_range, CompressOptions, Decoder, DecompressOptions, Encoder,
};
use gpu::decode::Lz4GpuDecoder;
use gpu::encode::{EncodeParams, GpuCompressOptions, Lz4GpuEncoder};
use gpu::Context;

#[derive(Clone, Copy, Debug)]
enum Enc {
    CpuLz4Flex,
    CpuGreedy,
    Gpu,
}

#[derive(Clone, Copy, Debug)]
enum Dec {
    CpuHandWritten,
    CpuLz4Flex,
    Gpu,
}

const ENCODERS: [Enc; 3] = [Enc::CpuLz4Flex, Enc::CpuGreedy, Enc::Gpu];
const DECODERS: [Dec; 3] = [Dec::CpuHandWritten, Dec::CpuLz4Flex, Dec::Gpu];
const CHUNK_SIZES: [u32; 3] = [4 << 10, 64 << 10, 1 << 20];

/// The M1 fixtures, inputs that stress LZ4's distance limit, and Canterbury.
fn corpus() -> Vec<(String, Vec<u8>)> {
    let mut inputs: Vec<(String, Vec<u8>)> = fixtures()
        .into_iter()
        .map(|(n, d)| (n.to_string(), d))
        .collect();
    let near = random(40_000, 11);
    let far = random(70_000, 12);
    inputs.push(("repeat at 40 KB".into(), [&near[..], &near[..]].concat()));
    inputs.push((
        "repeat at 70 KB (beyond max offset)".into(),
        [&far[..], &far[..]].concat(),
    ));
    inputs.push(("long text".into(), text(3 << 20)));
    inputs.extend(canterbury());
    inputs
}

struct Paths<'a> {
    ctx: &'a Context,
    encoder: Lz4GpuEncoder,
    decoder: Lz4GpuDecoder,
}

impl Paths<'_> {
    fn compress(&self, enc: Enc, input: &[u8], chunk_size: u32) -> Vec<u8> {
        let cpu = |encoder| CompressOptions {
            chunk_size,
            encoder,
            checksums: true,
            ..CompressOptions::default()
        };
        match enc {
            Enc::CpuLz4Flex => compress(input, &cpu(Encoder::Lz4Flex)).unwrap(),
            Enc::CpuGreedy => compress(input, &cpu(Encoder::Greedy(Default::default()))).unwrap(),
            Enc::Gpu => self
                .encoder
                .compress(
                    self.ctx,
                    input,
                    &GpuCompressOptions {
                        chunk_size,
                        checksums: true,
                        level: 1,
                    },
                )
                .unwrap(),
        }
    }

    fn decompress(&self, dec: Dec, file: &[u8]) -> Result<Vec<u8>, String> {
        let cpu = |decoder| DecompressOptions {
            decoder,
            verify: true,
        };
        match dec {
            Dec::CpuHandWritten => {
                decompress(file, &cpu(Decoder::HandWritten)).map_err(|e| e.to_string())
            }
            Dec::CpuLz4Flex => decompress(file, &cpu(Decoder::Lz4Flex)).map_err(|e| e.to_string()),
            Dec::Gpu => self
                .decoder
                .decompress(self.ctx, file, true)
                .map_err(|e| e.to_string()),
        }
    }

    fn range(&self, dec: Dec, file: &[u8], offset: u64, len: u64) -> Result<Vec<u8>, String> {
        let mut reader = Cursor::new(file);
        match dec {
            Dec::CpuHandWritten | Dec::CpuLz4Flex => {
                let decoder = match dec {
                    Dec::CpuLz4Flex => Decoder::Lz4Flex,
                    _ => Decoder::HandWritten,
                };
                let options = DecompressOptions {
                    decoder,
                    verify: true,
                };
                decompress_range(&mut reader, offset, len, &options).map_err(|e| e.to_string())
            }
            Dec::Gpu => self
                .decoder
                .decompress_range(self.ctx, &mut reader, offset, len, true)
                .map_err(|e| e.to_string()),
        }
    }
}

/// A few ranges that start, straddle chunk boundaries and end the input.
fn ranges(n: u64, chunk_size: u32) -> Vec<(u64, u64)> {
    let c = u64::from(chunk_size);
    let mut out = vec![(0, n.min(100)), (n.saturating_sub(77), n.min(77))];
    if n > c + 50 {
        out.push((c - 50, 100.min(n - (c - 50))));
    }
    if n > 3 {
        out.push((n / 3, n / 3));
    }
    out
}

#[test]
fn every_encoder_decodes_exactly_with_every_decoder_on_every_backend() {
    let corpus = corpus();
    for (backend, ctx) in contexts() {
        let paths = Paths {
            ctx: &ctx,
            encoder: Lz4GpuEncoder::new(&ctx, EncodeParams::default()).unwrap(),
            decoder: Lz4GpuDecoder::new(&ctx),
        };
        let mut checked = 0;
        for chunk_size in CHUNK_SIZES {
            for (name, input) in &corpus {
                for enc in ENCODERS {
                    let file = paths.compress(enc, input, chunk_size);
                    for dec in DECODERS {
                        let what =
                            format!("{backend} chunk={chunk_size} {name}: {enc:?} -> {dec:?}");
                        let out = paths
                            .decompress(dec, &file)
                            .unwrap_or_else(|e| panic!("{what}: {e}"));
                        assert!(out == *input, "{what}: output differs");
                        for (offset, len) in ranges(input.len() as u64, chunk_size) {
                            let got = paths
                                .range(dec, &file, offset, len)
                                .unwrap_or_else(|e| panic!("{what} range {offset}+{len}: {e}"));
                            assert!(
                                got == input[offset as usize..(offset + len) as usize],
                                "{what}: range {offset}+{len} differs"
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
        eprintln!("{backend}: {checked} encoder/decoder/input/chunk-size combinations exact");
    }
}

#[test]
fn gpu_and_cpu_greedy_files_are_identical_on_every_backend() {
    let corpus = corpus();
    for (backend, ctx) in contexts() {
        let encoder = Lz4GpuEncoder::new(&ctx, EncodeParams::default()).unwrap();
        for chunk_size in CHUNK_SIZES {
            for (name, input) in &corpus {
                let gpu = encoder
                    .compress(
                        &ctx,
                        input,
                        &GpuCompressOptions {
                            chunk_size,
                            checksums: true,
                            level: 1,
                        },
                    )
                    .unwrap();
                let cpu = compress(
                    input,
                    &CompressOptions {
                        chunk_size,
                        encoder: Encoder::Greedy(Default::default()),
                        checksums: true,
                        ..CompressOptions::default()
                    },
                )
                .unwrap();
                assert!(
                    gpu == cpu,
                    "{backend} chunk={chunk_size} {name}: GPU file differs from CPU twin"
                );
            }
        }
    }
}

#[test]
fn malformed_input_suite_fails_cleanly_on_every_backend() {
    let input = [text(9_000), random(3_000, 5)].concat();
    let file = compress(
        &input,
        &CompressOptions {
            chunk_size: 4096,
            encoder: Encoder::Greedy(Default::default()),
            checksums: true,
            ..CompressOptions::default()
        },
    )
    .unwrap();
    for (backend, ctx) in contexts() {
        let decoder = Lz4GpuDecoder::new(&ctx);
        // Every truncation is an error.
        for len in 0..file.len() {
            assert!(
                decoder.decompress(&ctx, &file[..len], true).is_err(),
                "{backend}: truncation to {len} bytes decoded"
            );
        }
        // Every single-byte corruption is an error or (thanks to checksums) exact.
        for i in 0..file.len() {
            let mut bad = file.clone();
            bad[i] ^= 0x5A;
            if let Ok(out) = decoder.decompress(&ctx, &bad, true) {
                assert!(
                    out == input,
                    "{backend}: corrupting byte {i} decoded to wrong data"
                );
            }
        }
    }
}
