//! M8: large inputs in bounded GPU memory. Batching by a memory budget,
//! streaming compress/decompress, and range reads that touch only what they
//! need. A 4 GiB round trip is `#[ignore]`d (run it with
//! `cargo test --release -p gpu --test stream -- --ignored`).

mod common;

use std::io::{Cursor, Read, Seek, SeekFrom, Write};

use common::{context, fixtures, text, CHUNK};
use format::{Codec, Index};
use gpu::decode::GpuDecoder;
use gpu::encode::{FilterMode, GpuCompressOptions, GpuEncoder};

/// Deterministic, moderately compressible bytes (text runs, numbers, noise),
/// produced on the fly so huge inputs never sit in memory.
struct Generator {
    state: u64,
    pos: u64,
    len: u64,
}

impl Generator {
    fn new(len: u64) -> Self {
        Generator {
            state: 0x9E37_79B9_7F4A_7C15,
            pos: 0,
            len,
        }
    }

    fn next_byte(&mut self) -> u8 {
        let p = self.pos;
        self.pos += 1;
        // 64 KiB regions alternating between kinds of data.
        match (p >> 16) % 4 {
            0 | 1 => {
                b"the quick brown fox jumps over the lazy dog. "[(p % 45) as usize]
                    ^ ((p >> 20) as u8 & 1)
            }
            2 => ((p / 4) as u32).wrapping_mul(3).to_le_bytes()[(p % 4) as usize],
            _ => {
                self.state ^= self.state << 13;
                self.state ^= self.state >> 7;
                self.state ^= self.state << 17;
                self.state as u8
            }
        }
    }
}

impl Read for Generator {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = buf.len().min((self.len - self.pos) as usize);
        for b in &mut buf[..n] {
            *b = self.next_byte();
        }
        Ok(n)
    }
}

/// Checks written bytes against a fresh generator instead of storing them.
struct Verifier {
    expected: Generator,
    written: u64,
}

impl Write for Verifier {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut want = vec![0; buf.len()];
        self.expected.read_exact(&mut want)?;
        assert!(
            want == buf,
            "output differs within bytes {}..{}",
            self.written,
            self.written + buf.len() as u64
        );
        self.written += buf.len() as u64;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn generated(len: u64) -> Vec<u8> {
    let mut v = Vec::new();
    Generator::new(len).read_to_end(&mut v).unwrap();
    v
}

fn opts(codec: Codec, filters: FilterMode) -> GpuCompressOptions {
    GpuCompressOptions {
        codec,
        chunk_size: CHUNK,
        checksums: true,
        filters,
        ..GpuCompressOptions::default()
    }
}

const MODES: [(Codec, FilterMode); 4] = [
    (Codec::Lz4, FilterMode::None),
    (Codec::Glz, FilterMode::None),
    (Codec::Lz4, FilterMode::Auto),
    (Codec::Glz, FilterMode::Exhaustive),
];

#[test]
fn streamed_compression_equals_in_memory_compression() {
    let Some(ctx) = context() else { return };
    let small = GpuEncoder::new(&ctx, Default::default())
        .unwrap()
        .with_memory_budget(256 << 10);
    let encoder = GpuEncoder::new(&ctx, Default::default()).unwrap();
    let mut inputs: Vec<(&str, Vec<u8>)> = fixtures();
    inputs.push(("generated", generated(3 << 20)));
    for (codec, filters) in MODES {
        for (name, input) in &inputs {
            let expected = encoder
                .compress(&ctx, input, &opts(codec, filters))
                .unwrap();
            // A writer that already holds a prefix: the file starts mid-stream.
            let mut out = Cursor::new(b"prefix".to_vec());
            out.seek(SeekFrom::End(0)).unwrap();
            let n = small
                .compress_stream(
                    &ctx,
                    &mut Cursor::new(input),
                    input.len() as u64,
                    &mut out,
                    &opts(codec, filters),
                )
                .unwrap();
            let out = out.into_inner();
            assert_eq!(n, expected.len() as u64, "{name} {codec:?} {filters:?}");
            assert!(
                &out[..6] == b"prefix" && out[6..] == expected,
                "{name} {codec:?} {filters:?}"
            );
            // The small budget really did split the work.
            assert_eq!(
                small.compress(&ctx, input, &opts(codec, filters)).unwrap(),
                expected
            );
        }
    }
}

#[test]
fn a_short_reader_is_an_error() {
    let Some(ctx) = context() else { return };
    let encoder = GpuEncoder::new(&ctx, Default::default()).unwrap();
    let mut out = Cursor::new(Vec::new());
    let result = encoder.compress_stream(
        &ctx,
        &mut Cursor::new(vec![1u8; 100]),
        200,
        &mut out,
        &opts(Codec::Lz4, FilterMode::None),
    );
    assert!(result.is_err());
}

#[test]
fn small_budgets_decode_in_many_batches() {
    let Some(ctx) = context() else { return };
    let encoder = GpuEncoder::new(&ctx, Default::default()).unwrap();
    let decoder = GpuDecoder::new(&ctx).with_memory_budget(64 << 10);
    let input = generated(2 << 20);
    for (codec, filters) in MODES {
        let file = encoder
            .compress(&ctx, &input, &opts(codec, filters))
            .unwrap();
        assert!(
            decoder.decompress(&ctx, &file, true).unwrap() == input,
            "{codec:?} {filters:?}"
        );
        let mut out = Vec::new();
        let n = decoder
            .decompress_stream(&ctx, &mut Cursor::new(&file), &mut out, true)
            .unwrap();
        assert_eq!(n, input.len() as u64);
        assert!(out == input, "stream {codec:?} {filters:?}");
        for (offset, len) in [
            (0u64, 10u64),
            (100_000, 700_000),
            (input.len() as u64 - 5, 5),
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
}

#[test]
fn decoding_empty_and_tiny_files_streams_correctly() {
    let Some(ctx) = context() else { return };
    let encoder = GpuEncoder::new(&ctx, Default::default()).unwrap();
    let decoder = GpuDecoder::new(&ctx).with_memory_budget(1);
    for (name, input) in fixtures() {
        let file = encoder
            .compress(&ctx, &input, &opts(Codec::Lz4, FilterMode::None))
            .unwrap();
        let mut out = Vec::new();
        decoder
            .decompress_stream(&ctx, &mut Cursor::new(&file), &mut out, true)
            .unwrap();
        assert!(out == input, "{name}");
    }
}

/// Counts the bytes actually read.
struct Counting<R> {
    inner: R,
    read: u64,
}

impl<R: Read> Read for Counting<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.read += n as u64;
        Ok(n)
    }
}

impl<R: Seek> Seek for Counting<R> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

/// Round trip of `len` generated bytes through a temporary file, in bounded
/// GPU memory; then a range read near the end must read only the index and
/// the chunks it needs.
fn round_trip_through_a_file(len: u64, budget: u64) {
    let Some(ctx) = context() else { return };
    let encoder = GpuEncoder::new(&ctx, Default::default())
        .unwrap()
        .with_memory_budget(budget);
    let decoder = GpuDecoder::new(&ctx).with_memory_budget(budget);
    let path =
        std::env::temp_dir().join(format!("gpucomp-stream-{len}-{}.gpcz", std::process::id()));
    let options = GpuCompressOptions {
        chunk_size: 64 << 10,
        checksums: true,
        ..GpuCompressOptions::default()
    };
    {
        let mut file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
        encoder
            .compress_stream(&ctx, &mut Generator::new(len), len, &mut file, &options)
            .unwrap();
        file.flush().unwrap();
    }
    let mut reader = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
    let mut verifier = Verifier {
        expected: Generator::new(len),
        written: 0,
    };
    let n = decoder
        .decompress_stream(&ctx, &mut reader, &mut verifier, true)
        .unwrap();
    assert_eq!((n, verifier.written), (len, len));

    let index = format::read_index(&mut std::fs::File::open(&path).unwrap()).unwrap();
    let (offset, want) = (len - 200_000, 150_000);
    let mut counting = Counting {
        inner: std::fs::File::open(&path).unwrap(),
        read: 0,
    };
    let got = decoder
        .decompress_range(&ctx, &mut counting, offset, want, true)
        .unwrap();
    let mut expected = Generator::new(len);
    std::io::copy(&mut (&mut expected).take(offset), &mut std::io::sink()).unwrap();
    let mut want_bytes = vec![0; want as usize];
    expected.read_exact(&mut want_bytes).unwrap();
    assert!(got == want_bytes);
    let needed = index.chunks_for_range(offset, want).unwrap();
    let (first, last) = (index.chunks[needed.start], index.chunks[needed.end - 1]);
    let span = format::pad4(last.comp_offset + u64::from(last.comp_size)) - first.comp_offset;
    assert!(
        counting.read <= index.data_offset() + span,
        "read {} bytes",
        counting.read
    );
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn a_96_mib_stream_round_trips_in_8_mib_of_gpu_memory() {
    round_trip_through_a_file(96 << 20, 8 << 20);
}

#[test]
#[ignore = "4 GiB: run explicitly in release mode"]
fn a_4_gib_stream_round_trips_in_bounded_gpu_memory() {
    round_trip_through_a_file((4 << 30) + 12_345, 512 << 20);
}

#[test]
fn compressed_index_of_a_generated_stream_is_valid() {
    let Some(ctx) = context() else { return };
    let encoder = GpuEncoder::new(&ctx, Default::default())
        .unwrap()
        .with_memory_budget(1 << 20);
    let input = text(5 << 20);
    let mut out = Cursor::new(Vec::new());
    encoder
        .compress_stream(
            &ctx,
            &mut Cursor::new(&input),
            input.len() as u64,
            &mut out,
            &opts(Codec::Lz4, FilterMode::None),
        )
        .unwrap();
    let file = out.into_inner();
    let index = Index::parse(&file).unwrap();
    index
        .validate(file.len() as u64 - index.data_offset())
        .unwrap();
}
