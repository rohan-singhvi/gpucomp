//! Streaming file-to-file compression and decompression: the same files as
//! the in-memory functions, produced and consumed a group of chunks at a time.

use std::io::{Cursor, Seek, SeekFrom, Write};

use cpu::container::{
    compress, compress_into, compress_stream, compress_stream_data, compress_stream_grouped,
    decompress_stream, decompress_stream_grouped, CompressOptions, CpuError, DecompressOptions,
    Encoder, FilterMode,
};
use cpu::glz::GlzParams;
use cpu::lz4::encode::Params;
use format::Codec;

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
        "lorem ", "ipsum ", "dolor ", "sit ", "amet, ", "chunk ", "\n",
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

fn inputs() -> Vec<(&'static str, Vec<u8>)> {
    let c = CHUNK as usize;
    vec![
        ("empty", vec![]),
        ("one byte", vec![42]),
        ("one chunk", text(c)),
        ("chunk + 1", text(c + 1)),
        (
            "mixed",
            [text(7 * c), random(3 * c, 2), vec![0; 4 * c], text(99)].concat(),
        ),
    ]
}

fn option_sets() -> Vec<CompressOptions> {
    let base = CompressOptions {
        chunk_size: CHUNK,
        ..CompressOptions::default()
    };
    let glz = Encoder::Glz(GlzParams {
        lz: Params::for_level(2),
        independent_groups: None,
    });
    vec![
        base,
        CompressOptions {
            checksums: true,
            encoder: Encoder::Greedy(Params::for_level(1)),
            ..base
        },
        CompressOptions {
            codec: Codec::Glz,
            encoder: glz,
            filters: FilterMode::Auto,
            ..base
        },
        CompressOptions {
            codec: Codec::GlzE,
            encoder: glz,
            checksums: true,
            level: 2,
            ..base
        },
        CompressOptions {
            codec: Codec::Stored,
            ..base
        },
    ]
}

/// Group sizes in bytes: under one chunk (rounds up to one), a few chunks
/// that don't divide the input, and more than any input.
const GROUPS: [usize; 4] = [1, 3 * CHUNK as usize, 5 * CHUNK as usize + 17, 1 << 20];

#[test]
fn streamed_files_equal_in_memory_files_for_any_group_size() {
    for (name, input) in inputs() {
        for options in option_sets() {
            let expected = compress(&input, &options).unwrap();
            for group in GROUPS {
                let mut out = Cursor::new(Vec::new());
                let len = input.len() as u64;
                let written = compress_stream_grouped(
                    &mut Cursor::new(&input),
                    len,
                    &mut out,
                    &options,
                    group,
                )
                .unwrap();
                assert_eq!(written, expected.len() as u64, "{name} {options:?} {group}");
                assert!(out.into_inner() == expected, "{name} {options:?} {group}");
            }
        }
    }
}

#[test]
fn a_stream_starts_at_the_writers_position_and_ends_after_the_file() {
    let input = text(9 * CHUNK as usize);
    let options = option_sets()[0];
    let expected = compress(&input, &options).unwrap();
    let mut out = Cursor::new(Vec::new());
    out.write_all(b"prefix").unwrap();
    let written = compress_stream(
        &mut Cursor::new(&input),
        input.len() as u64,
        &mut out,
        &options,
    )
    .unwrap();
    assert_eq!(written, expected.len() as u64);
    assert_eq!(out.stream_position().unwrap(), 6 + written);
    assert!(out.into_inner()[6..] == expected[..]);
}

#[test]
fn a_reader_shorter_than_its_length_is_an_error() {
    let input = text(5 * CHUNK as usize);
    let mut out = Cursor::new(Vec::new());
    let result = compress_stream_grouped(
        &mut Cursor::new(&input),
        input.len() as u64 + 1,
        &mut out,
        &option_sets()[0],
        CHUNK as usize,
    );
    assert!(matches!(result, Err(CpuError::Io(_))), "{result:?}");
}

#[test]
fn streamed_decompression_restores_the_input_for_any_group_size() {
    for (name, input) in inputs() {
        for options in option_sets() {
            let file = compress(&input, &options).unwrap();
            for group in GROUPS {
                let mut out = Vec::new();
                let n = decompress_stream_grouped(
                    &mut Cursor::new(&file),
                    &mut out,
                    &DecompressOptions::default(),
                    group,
                )
                .unwrap();
                assert_eq!(n, input.len() as u64, "{name} {options:?} {group}");
                assert!(out == input, "{name} {options:?} {group}");
            }
        }
    }
    let input = text(20 * CHUNK as usize);
    let file = compress(&input, &option_sets()[1]).unwrap();
    let mut out = Vec::new();
    decompress_stream(
        &mut Cursor::new(&file),
        &mut out,
        &DecompressOptions::default(),
    )
    .unwrap();
    assert!(out == input);
}

#[test]
fn streamed_decompression_reports_corrupt_chunks_and_truncated_files() {
    let input = text(12 * CHUNK as usize);
    let options = option_sets()[1]; // checksums
    let mut file = compress(&input, &options).unwrap();
    let truncated = &file[..file.len() - 5];
    let result = decompress_stream_grouped(
        &mut Cursor::new(truncated),
        &mut Vec::new(),
        &DecompressOptions::default(),
        CHUNK as usize,
    );
    assert!(result.is_err());
    // Corrupt the middle chunk's payload: the error names it, whatever the
    // group size, and nothing after it is written.
    let index = format::Index::parse(&file).unwrap();
    let entry = index.chunks[6];
    let at = (index.data_offset() + entry.comp_offset) as usize + entry.comp_size as usize / 2;
    file[at] ^= 0x55;
    for group in GROUPS {
        let mut out = Vec::new();
        let result = decompress_stream_grouped(
            &mut Cursor::new(&file),
            &mut out,
            &DecompressOptions::default(),
            group,
        );
        match result {
            Err(CpuError::Checksum { chunk: 6 } | CpuError::Decode { chunk: 6, .. }) => {}
            other => panic!("group {group}: {other:?}"),
        }
        assert!(out.len() <= 6 * CHUNK as usize, "group {group}");
        assert!(out[..] == input[..out.len()], "group {group}");
    }
}

#[test]
fn streamed_decompression_reads_from_the_start_of_a_seekable_file() {
    let input = text(3 * CHUNK as usize);
    let file = compress(&input, &option_sets()[0]).unwrap();
    let mut reader = Cursor::new(&file);
    reader.seek(SeekFrom::End(0)).unwrap();
    let mut out = Vec::new();
    decompress_stream(&mut reader, &mut out, &DecompressOptions::default()).unwrap();
    assert!(out == input);
}

#[test]
fn a_data_section_streams_on_any_pool_as_compress_into_appends_it() {
    // One worker: the reader isn't a worker, so the pipeline can't deadlock.
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    for (name, input) in inputs() {
        for options in option_sets() {
            let mut expected = Vec::new();
            let entries = compress_into(&input, &options, &mut expected).unwrap();
            for (group, pool) in [(CHUNK as usize, None), (3 * CHUNK as usize, Some(&one))] {
                let mut out = Vec::new();
                let (got, len) = compress_stream_data(
                    &mut Cursor::new(&input),
                    input.len() as u64,
                    &mut out,
                    &options,
                    group,
                    pool,
                )
                .unwrap();
                assert_eq!(got, entries, "{name} {options:?} {group}");
                assert_eq!(len, expected.len() as u64, "{name} {options:?} {group}");
                assert!(out == expected, "{name} {options:?} {group}");
            }
        }
    }
}
