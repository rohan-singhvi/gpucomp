use proptest::prelude::*;

use super::*;

fn header() -> Header {
    Header {
        codec: Codec::Lz4,
        chunk_size: 4096,
        chunk_count: 3,
        total_size: 10_000,
        checksums: true,
        level: 1,
    }
}

fn entry(comp_offset: u64, comp_size: u32, stored: bool, uncomp_size: u32) -> ChunkEntry {
    ChunkEntry {
        comp_offset,
        comp_size,
        stored,
        uncomp_size,
        checksum: 0xABCD_0123,
        filter: Filter::None,
    }
}

/// 3 chunks (4096, 4096, 1808 bytes); the middle one stored.
fn index() -> Index {
    Index {
        header: header(),
        chunks: vec![
            entry(0, 101, false, 4096),
            entry(104, 4096, true, 4096),
            entry(4200, 50, false, 1808),
        ],
    }
}
const DATA_LEN: u64 = 4252;

// ---- helpers ----

#[test]
fn pad4_rounds_up_to_multiple_of_four() {
    assert_eq!([0, 1, 3, 4, 5, 8].map(pad4), [0, 4, 4, 4, 8, 8]);
}

#[test]
fn chunk_count_is_ceiling_of_total_over_chunk_size() {
    assert_eq!(chunk_count_for(0, 4096), 0);
    assert_eq!(chunk_count_for(1, 4096), 1);
    assert_eq!(chunk_count_for(4096, 4096), 1);
    assert_eq!(chunk_count_for(4097, 4096), 2);
}

#[test]
fn last_chunk_holds_the_remainder() {
    let h = header();
    assert_eq!(
        [0, 1, 2].map(|i| h.uncomp_size_of(i)),
        [4096, 4096, 10_000 - 8192]
    );
}

// ---- header ----

#[test]
fn header_layout_matches_spec() {
    let b = header().to_bytes();
    assert_eq!(&b[0..4], b"GPCZ");
    assert_eq!(&b[4..6], &1u16.to_le_bytes()); // version
    assert_eq!(&b[6..8], &1u16.to_le_bytes()); // codec
    assert_eq!(&b[8..12], &4096u32.to_le_bytes());
    assert_eq!(&b[12..16], &3u32.to_le_bytes());
    assert_eq!(&b[16..24], &10_000u64.to_le_bytes());
    assert_eq!(&b[24..28], &1u32.to_le_bytes()); // flags: checksums
    assert_eq!(b[28], 1); // level
    assert_eq!(&b[29..32], &[0, 0, 0]);
}

#[test]
fn header_round_trips() {
    assert_eq!(Header::parse(&header().to_bytes()), Ok(header()));
}

#[test]
fn header_rejects_short_input() {
    assert_eq!(
        Header::parse(&header().to_bytes()[..31]),
        Err(FormatError::Truncated { need: 32, have: 31 })
    );
}

#[test]
fn header_rejects_bad_magic() {
    let mut b = header().to_bytes();
    b[0] = b'X';
    assert_eq!(Header::parse(&b), Err(FormatError::BadMagic));
}

#[test]
fn header_rejects_other_versions() {
    let mut b = header().to_bytes();
    b[4] = 2;
    assert_eq!(Header::parse(&b), Err(FormatError::UnsupportedVersion(2)));
}

#[test]
fn header_rejects_unknown_codec() {
    let mut b = header().to_bytes();
    b[6] = 9;
    assert_eq!(Header::parse(&b), Err(FormatError::UnknownCodec(9)));
}

#[test]
fn header_rejects_unknown_flags() {
    let mut b = header().to_bytes();
    b[24] = 0b11;
    assert_eq!(Header::parse(&b), Err(FormatError::UnknownFlags(0b11)));
}

#[test]
fn header_rejects_nonzero_reserved() {
    let mut b = header().to_bytes();
    b[31] = 1;
    assert_eq!(Header::parse(&b), Err(FormatError::NonzeroReserved));
}

// ---- chunk entries ----

#[test]
fn entry_layout_matches_spec() {
    let e = ChunkEntry {
        filter: Filter::Delta { width: 4 },
        ..entry(0x1_0000_0004, 77, true, 4096)
    };
    let b = e.to_bytes();
    assert_eq!(&b[0..8], &0x1_0000_0004u64.to_le_bytes());
    assert_eq!(&b[8..12], &(77 | STORED_BIT).to_le_bytes());
    assert_eq!(&b[12..16], &4096u32.to_le_bytes());
    assert_eq!(&b[16..20], &0xABCD_0123u32.to_le_bytes());
    assert_eq!((b[20], b[21]), (2, 4)); // filter id, width
    assert_eq!(&b[22..24], &[0, 0]);
}

#[test]
fn entries_round_trip_for_every_filter() {
    for filter in [
        Filter::None,
        Filter::Shuffle { width: 8 },
        Filter::Delta { width: 2 },
    ] {
        let e = ChunkEntry {
            filter,
            ..entry(8, 5, false, 9)
        };
        assert_eq!(ChunkEntry::parse(&e.to_bytes()), Ok(e));
    }
}

#[test]
fn entry_rejects_unknown_filter() {
    let mut b = entry(0, 1, false, 1).to_bytes();
    b[20] = 7;
    assert_eq!(ChunkEntry::parse(&b), Err(FormatError::UnknownFilter(7)));
}

#[test]
fn entry_rejects_bad_filter_widths() {
    for (id, width) in [(1, 3), (2, 0), (2, 16), (0, 4)] {
        let mut b = entry(0, 1, false, 1).to_bytes();
        (b[20], b[21]) = (id, width);
        assert_eq!(
            ChunkEntry::parse(&b),
            Err(FormatError::BadFilterWidth { id, width })
        );
    }
}

#[test]
fn entry_rejects_nonzero_reserved() {
    let mut b = entry(0, 1, false, 1).to_bytes();
    b[23] = 1;
    assert_eq!(ChunkEntry::parse(&b), Err(FormatError::NonzeroReserved));
}

// ---- index ----

#[test]
fn index_round_trips_and_data_follows_the_table() {
    let bytes = index().to_bytes();
    assert_eq!(bytes.len(), 32 + 3 * 24);
    assert_eq!(index().data_offset(), 32 + 3 * 24);
    assert_eq!(Index::parse(&bytes), Ok(index()));
}

#[test]
fn index_parse_rejects_truncated_table() {
    let bytes = index().to_bytes();
    assert_eq!(
        Index::parse(&bytes[..bytes.len() - 1]),
        Err(FormatError::Truncated {
            need: 104,
            have: 103
        })
    );
}

#[test]
fn valid_index_validates() {
    assert_eq!(index().validate(DATA_LEN), Ok(()));
}

#[test]
fn empty_file_validates() {
    let idx = Index {
        header: Header {
            chunk_count: 0,
            total_size: 0,
            ..header()
        },
        chunks: vec![],
    };
    assert_eq!(idx.validate(0), Ok(()));
}

#[test]
fn validate_rejects_bad_chunk_sizes() {
    for size in [0, 2048, 4097, 3 << 12, 2 << 20] {
        let mut idx = index();
        idx.header.chunk_size = size;
        assert_eq!(
            idx.validate(DATA_LEN),
            Err(FormatError::BadChunkSize(size)),
            "{size}"
        );
    }
}

#[test]
fn validate_rejects_chunk_count_mismatch() {
    let mut idx = index();
    idx.header.total_size = 20_000;
    assert_eq!(
        idx.validate(DATA_LEN),
        Err(FormatError::ChunkCountMismatch {
            expected: 5,
            actual: 3
        })
    );
}

fn rejects_chunk(idx: Index, data_len: u64, index: usize) {
    assert!(
        matches!(idx.validate(data_len), Err(FormatError::BadChunk { index: i, .. }) if i == index),
        "{:?}",
        idx.validate(data_len)
    );
}

#[test]
fn validate_rejects_wrong_uncompressed_size() {
    let mut idx = index();
    idx.chunks[2].uncomp_size = 1807;
    rejects_chunk(idx, DATA_LEN, 2);
}

#[test]
fn validate_rejects_misaligned_offset() {
    let mut idx = index();
    idx.chunks[2].comp_offset = 4202;
    rejects_chunk(idx, DATA_LEN + 4, 2);
}

#[test]
fn validate_rejects_overlapping_payloads() {
    let mut idx = index();
    idx.chunks[1].comp_offset = 100; // chunk 0 occupies 0..104 after padding
    rejects_chunk(idx, DATA_LEN, 1);
}

#[test]
fn validate_rejects_payload_past_end_of_data() {
    rejects_chunk(index(), DATA_LEN - 4, 2);
}

#[test]
fn validate_rejects_missing_final_padding() {
    // Last payload ends at 4250; its padding runs to 4252.
    rejects_chunk(index(), DATA_LEN - 1, 2);
}

#[test]
fn validate_rejects_stored_chunk_with_different_size() {
    let mut idx = index();
    idx.chunks[1].comp_size = 4000;
    rejects_chunk(idx, DATA_LEN, 1);
}

#[test]
fn validate_rejects_nonzero_checksum_without_flag() {
    let mut idx = index();
    idx.header.checksums = false;
    rejects_chunk(idx, DATA_LEN, 0);
}

#[test]
fn validate_rejects_compressed_chunk_in_stored_codec() {
    let mut idx = index();
    idx.header.codec = Codec::Stored;
    rejects_chunk(idx, DATA_LEN, 0);
}

// ---- random access ----

#[test]
fn range_inside_one_chunk_needs_only_that_chunk() {
    assert_eq!(index().chunks_for_range(4100, 10), Ok(1..2));
}

#[test]
fn range_spanning_a_boundary_needs_both_chunks() {
    assert_eq!(index().chunks_for_range(4000, 200), Ok(0..2));
}

#[test]
fn range_ending_exactly_on_a_boundary_stops_there() {
    assert_eq!(index().chunks_for_range(0, 4096), Ok(0..1));
}

#[test]
fn range_covering_the_short_last_chunk() {
    assert_eq!(index().chunks_for_range(9000, 1000), Ok(2..3));
}

#[test]
fn zero_length_range_needs_no_chunks() {
    assert_eq!(index().chunks_for_range(5000, 0), Ok(0..0));
    assert_eq!(index().chunks_for_range(10_000, 0), Ok(0..0));
}

#[test]
fn out_of_bounds_ranges_are_rejected() {
    for (offset, len) in [(9_999, 2), (10_001, 0), (u64::MAX, 2)] {
        assert_eq!(
            index().chunks_for_range(offset, len),
            Err(FormatError::RangeOutOfBounds {
                offset,
                len,
                total: 10_000
            })
        );
    }
}

proptest! {
    #[test]
    fn any_header_round_trips(
        codec in prop_oneof![Just(Codec::Stored), Just(Codec::Lz4)],
        chunk_shift in 12u32..=20,
        total_size in 0u64..(1 << 40),
        checksums: bool,
        level: u8,
    ) {
        let chunk_size = 1 << chunk_shift;
        let h = Header {
            codec,
            chunk_size,
            chunk_count: chunk_count_for(total_size, chunk_size) as u32,
            total_size,
            checksums,
            level,
        };
        prop_assert_eq!(Header::parse(&h.to_bytes()), Ok(h));
    }

    #[test]
    fn chunks_for_range_covers_exactly_the_overlapping_chunks(
        offset in 0u64..10_000,
        len in 0u64..10_000,
    ) {
        prop_assume!(offset + len <= 10_000);
        let range = index().chunks_for_range(offset, len).unwrap();
        for i in 0..3usize {
            let start = i as u64 * 4096;
            let end = (start + 4096).min(10_000);
            let overlaps = len > 0 && start < offset + len && offset < end;
            prop_assert_eq!(range.contains(&i), overlaps, "chunk {}", i);
        }
    }
}
