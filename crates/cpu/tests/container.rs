//! Container round trips, cross-decoder checks, range reads and malformed input.

use std::io::{Cursor, Read, Seek, SeekFrom};

use cpu::container::{
    compress, decompress, decompress_range, read_index, CompressOptions, CpuError, Decoder,
    DecompressOptions, Encoder,
};
use cpu::lz4::encode::Params;
use format::{Codec, Filter, Index};
use proptest::prelude::*;

const CHUNK: u32 = 4096;

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
    let words = [
        "lorem ", "ipsum ", "dolor ", "sit ", "amet, ", "gpu ", "chunk ", "\n",
    ];
    let mut out = Vec::with_capacity(n + 8);
    let mut i = 0usize;
    while out.len() < n {
        out.extend_from_slice(words[(i * 7 + i / 3) % words.len()].as_bytes());
        i += 1;
    }
    out.truncate(n);
    out
}

fn fixtures() -> Vec<(&'static str, Vec<u8>)> {
    let c = CHUNK as usize;
    vec![
        ("empty", vec![]),
        ("one byte", vec![42]),
        ("exactly one chunk", text(c)),
        ("chunk - 1", text(c - 1)),
        ("chunk + 1", text(c + 1)),
        ("random", random(10 * c + 123, 1)),
        ("zeros", vec![0; 10 * c]),
        ("text", text(10 * c + 7)),
        (
            "mixed",
            [text(3 * c), random(2 * c, 2), vec![7; 3 * c]].concat(),
        ),
    ]
}

const ENCODERS: [Encoder; 2] = [
    Encoder::Lz4Flex,
    Encoder::Greedy(Params {
        block: 64,
        hash_log: 12,
        probe_len: 16,
    }),
];
const DECODERS: [Decoder; 2] = [Decoder::HandWritten, Decoder::Lz4Flex];

fn opts(encoder: Encoder) -> CompressOptions {
    CompressOptions {
        chunk_size: CHUNK,
        encoder,
        checksums: true,
        ..CompressOptions::default()
    }
}

fn dopts(decoder: Decoder) -> DecompressOptions {
    DecompressOptions {
        decoder,
        verify: true,
    }
}

#[test]
fn every_encoder_round_trips_with_every_decoder() {
    for (name, input) in fixtures() {
        for encoder in ENCODERS {
            let file = compress(&input, &opts(encoder)).unwrap();
            for decoder in DECODERS {
                let out = decompress(&file, &dopts(decoder))
                    .unwrap_or_else(|e| panic!("{name} {encoder:?} {decoder:?}: {e}"));
                assert!(out == input, "{name} {encoder:?} {decoder:?}");
            }
        }
    }
}

#[test]
fn compressed_files_have_a_valid_index() {
    for (name, input) in fixtures() {
        let file = compress(&input, &opts(Encoder::Lz4Flex)).unwrap();
        let index = Index::parse(&file).unwrap();
        index
            .validate(file.len() as u64 - index.data_offset())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(index.header.total_size, input.len() as u64, "{name}");
    }
}

#[test]
fn incompressible_chunks_are_stored_raw() {
    let file = compress(&random(3 * CHUNK as usize, 9), &opts(Encoder::Lz4Flex)).unwrap();
    let index = Index::parse(&file).unwrap();
    assert!(index
        .chunks
        .iter()
        .all(|c| c.stored && c.comp_size == CHUNK));
}

#[test]
fn compressible_chunks_are_not_stored() {
    let file = compress(&vec![0; 3 * CHUNK as usize], &opts(Encoder::Lz4Flex)).unwrap();
    let index = Index::parse(&file).unwrap();
    assert!(index.chunks.iter().all(|c| !c.stored && c.comp_size < 100));
}

#[test]
fn stored_codec_stores_every_chunk() {
    let input = vec![0; 3 * CHUNK as usize];
    let options = CompressOptions {
        codec: Codec::Stored,
        ..opts(Encoder::Lz4Flex)
    };
    let file = compress(&input, &options).unwrap();
    let index = Index::parse(&file).unwrap();
    assert_eq!(index.header.codec, Codec::Stored);
    assert!(index.chunks.iter().all(|c| c.stored));
    assert_eq!(
        decompress(&file, &dopts(Decoder::HandWritten)).unwrap(),
        input
    );
}

#[test]
fn checksums_are_zero_when_disabled() {
    let options = CompressOptions {
        checksums: false,
        ..opts(Encoder::Lz4Flex)
    };
    let file = compress(&text(20_000), &options).unwrap();
    let index = Index::parse(&file).unwrap();
    assert!(!index.header.checksums);
    assert!(index.chunks.iter().all(|c| c.checksum == 0));
}

#[test]
fn compression_is_deterministic() {
    for encoder in ENCODERS {
        let input = text(50_000);
        assert_eq!(
            compress(&input, &opts(encoder)).unwrap(),
            compress(&input, &opts(encoder)).unwrap()
        );
    }
}

#[test]
fn invalid_chunk_size_is_rejected() {
    let options = CompressOptions {
        chunk_size: 5000,
        ..opts(Encoder::Lz4Flex)
    };
    assert!(matches!(
        compress(b"abc", &options),
        Err(CpuError::Format(format::FormatError::BadChunkSize(5000)))
    ));
}

#[test]
fn verify_detects_a_corrupted_stored_chunk() {
    let input = random(2 * CHUNK as usize, 3);
    let mut file = compress(&input, &opts(Encoder::Lz4Flex)).unwrap();
    let data = Index::parse(&file).unwrap().data_offset() as usize;
    file[data + 10] ^= 0xFF;
    assert!(matches!(
        decompress(&file, &dopts(Decoder::HandWritten)),
        Err(CpuError::Checksum { chunk: 0 })
    ));
    let unverified = DecompressOptions {
        verify: false,
        ..dopts(Decoder::HandWritten)
    };
    assert_ne!(decompress(&file, &unverified).unwrap(), input);
}

#[test]
fn filtered_chunks_are_rejected_until_filters_exist() {
    let file = compress(&text(10_000), &opts(Encoder::Lz4Flex)).unwrap();
    let mut index = Index::parse(&file).unwrap();
    index.chunks[1].filter = Filter::Shuffle { width: 4 };
    let header_len = index.data_offset() as usize;
    let patched = [index.to_bytes(), file[header_len..].to_vec()].concat();
    assert!(matches!(
        decompress(&patched, &dopts(Decoder::HandWritten)),
        Err(CpuError::UnsupportedFilter { chunk: 1 })
    ));
}

#[test]
fn every_truncation_is_an_error_not_a_panic() {
    let file = compress(
        &text(3 * CHUNK as usize),
        &opts(Encoder::Greedy(Params::default())),
    )
    .unwrap();
    for len in 0..file.len() {
        for decoder in DECODERS {
            assert!(
                decompress(&file[..len], &dopts(decoder)).is_err(),
                "len {len}"
            );
        }
    }
}

// ---- random access ----

/// Counts the bytes actually read, to prove a range read skips other chunks.
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

fn range(file: &[u8], offset: u64, len: u64) -> Result<Vec<u8>, CpuError> {
    decompress_range(
        &mut Cursor::new(file),
        offset,
        len,
        &dopts(Decoder::HandWritten),
    )
}

#[test]
fn range_reads_match_slices_of_the_original() {
    let input = text(10 * CHUNK as usize + 1234);
    let file = compress(&input, &opts(Encoder::Greedy(Params::default()))).unwrap();
    let c = u64::from(CHUNK);
    let n = input.len() as u64;
    let cases = [
        (0, 10),            // inside the first chunk
        (c + 5, 100),       // inside a middle chunk
        (c - 50, 100),      // spanning one boundary
        (c - 1, 2 * c + 2), // spanning several chunks
        (0, c),             // exactly one chunk
        (n - 1234, 1234),   // exactly the short last chunk
        (n - 10, 10),       // tail of the last chunk
        (0, n),             // everything
        (3 * c, 0),         // empty
    ];
    for (offset, len) in cases {
        let got = range(&file, offset, len).unwrap();
        assert!(
            got == input[offset as usize..(offset + len) as usize],
            "range {offset}+{len}"
        );
    }
}

#[test]
fn out_of_bounds_ranges_are_rejected() {
    let file = compress(&text(10_000), &opts(Encoder::Lz4Flex)).unwrap();
    for (offset, len) in [(9_999, 2), (10_001, 0), (u64::MAX, 1)] {
        assert!(matches!(
            range(&file, offset, len),
            Err(CpuError::Format(
                format::FormatError::RangeOutOfBounds { .. }
            ))
        ));
    }
}

#[test]
fn range_read_touches_only_the_index_and_needed_chunks() {
    let input = text(64 * CHUNK as usize);
    let file = compress(&input, &opts(Encoder::Lz4Flex)).unwrap();
    let index = Index::parse(&file).unwrap();
    let mut reader = Counting {
        inner: Cursor::new(&file),
        read: 0,
    };
    let offset = 40 * u64::from(CHUNK) + 100;
    let got = decompress_range(&mut reader, offset, 5000, &dopts(Decoder::HandWritten)).unwrap();
    assert!(got == input[offset as usize..offset as usize + 5000]);
    // Chunks 40 and 41 are contiguous; the span includes chunk 40's padding.
    let (first, last) = (index.chunks[40], index.chunks[41]);
    let needed = last.comp_offset + u64::from(last.comp_size) - first.comp_offset;
    assert_eq!(reader.read, index.data_offset() + needed);
}

#[test]
fn read_index_validates_against_the_file_length() {
    let file = compress(&text(10_000), &opts(Encoder::Lz4Flex)).unwrap();
    assert!(read_index(&mut Cursor::new(&file)).is_ok());
    assert!(read_index(&mut Cursor::new(&file[..file.len() - 4])).is_err());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn arbitrary_inputs_round_trip(
        input in prop::collection::vec(prop_oneof![3 => 0u8..4, 1 => any::<u8>()], 0..40_000),
        chunk_shift in 12u32..=16,
        greedy: bool,
    ) {
        let encoder = if greedy { ENCODERS[1] } else { ENCODERS[0] };
        let options = CompressOptions { chunk_size: 1 << chunk_shift, ..opts(encoder) };
        let file = compress(&input, &options).unwrap();
        for decoder in DECODERS {
            prop_assert_eq!(&decompress(&file, &dopts(decoder)).unwrap(), &input);
        }
    }

    #[test]
    fn corrupted_files_never_panic(
        flips in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..8),
    ) {
        let input = text(3 * CHUNK as usize);
        let mut file = compress(&input, &opts(Encoder::Greedy(Params::default()))).unwrap();
        for (at, byte) in flips {
            let i = at.index(file.len());
            file[i] ^= byte | 1;
        }
        for decoder in DECODERS {
            // Errors are fine; with checksums verified, success means exact output.
            if let Ok(out) = decompress(&file, &dopts(decoder)) {
                prop_assert_eq!(&out, &input);
            }
        }
    }
}
