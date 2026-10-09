//! Greedy LZ4 block encoder that runs the same algorithm as the GPU encoder
//! (M3), phase by phase, so GPU intermediate buffers can be checked against it.
//!
//! 1. **Match finding** (parallel on the GPU): positions are processed in blocks
//!    of `block` positions (= the GPU workgroup size). Each position hashes its
//!    next 4 bytes and looks up a candidate in a hash table that holds only
//!    positions from *earlier* blocks, verifies it and extends the match up to
//!    `probe_len` bytes. Then every position of the block is inserted; when
//!    several hit one bucket, the latest position wins (`atomicMax` on the GPU).
//! 2. **Parse**: walk forward taking any match of length ≥ 4, except (lazy)
//!    when the next position's match is longer.
//!    A match that hit the `probe_len` cap is extended here, so total extension
//!    work is O(n) instead of O(n × match length).
//! 3. **Emit**: write the LZ4 sequences.

use super::{LAST_LITERALS, MAX_OFFSET, MFLIMIT, MIN_MATCH};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    /// Positions per match-finding block (the GPU workgroup size).
    pub block: usize,
    /// Hash table has `1 << hash_log` buckets.
    pub hash_log: u32,
    /// Phase 1 extends each match at most this far; phase 2 extends taken ones.
    pub probe_len: usize,
    /// Lazy parse: skip a match when the next position's (phase-1) match is
    /// longer. Otherwise greedy.
    pub lazy: bool,
    /// Candidates per position: the latest earlier-block position with the
    /// same hash, then that position's own candidate, and so on (a hash chain).
    pub depth: usize,
}

impl Default for Params {
    fn default() -> Self {
        // Same as the GPU encoder's level-1 default (gpu::encode::EncodeParams).
        Params {
            block: 128,
            hash_log: 12,
            probe_len: 16,
            lazy: true,
            depth: 1,
        }
    }
}

impl Params {
    /// The encoder for compression `level` (clamped to 1..=3): hash-chain
    /// candidates 1, 4 and 16 (DECISIONS.md, "Hash chains and levels").
    pub fn for_level(level: u8) -> Self {
        Params {
            depth: match level {
                0 | 1 => 1,
                2 => 4,
                _ => 16,
            },
            ..Params::default()
        }
    }
}

/// Best match found for one position in phase 1. `len == 0` means none.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Match {
    pub len: u32,
    pub offset: u32,
}

/// One LZ4 sequence: literals `lit_start..lit_start + lit_len`, then a match.
/// The final sequence of a block has `match_len == 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sequence {
    pub lit_start: u32,
    pub lit_len: u32,
    pub match_len: u32,
    pub offset: u32,
}

/// Multiplicative hash of a little-endian 4-byte word into `hash_log` bits.
pub fn hash(word: u32, hash_log: u32) -> u32 {
    word.wrapping_mul(2_654_435_761) >> (32 - hash_log)
}

fn read_u32(input: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(input[p..p + 4].try_into().unwrap())
}

/// Phase 1: the best (probe-capped) match for every position.
pub fn find_matches(input: &[u8], params: &Params) -> Vec<Match> {
    let n = input.len();
    let mut matches = vec![Match::default(); n];
    if n < MFLIMIT {
        return matches;
    }
    let last_start = n - MFLIMIT; // last position a match may start at
    let match_limit = n - LAST_LITERALS; // matches end at or before this
                                         // Bucket value = position + 1; 0 = empty (the GPU zero-initialises its table).
    let mut table = vec![0u32; 1 << params.hash_log];
    // Per position, its first candidate (position + 1; 0 = none): following
    // these links walks back through earlier blocks' positions with the hash.
    let mut chain = vec![0u32; if params.depth > 1 { n } else { 0 }];

    for block_start in (0..=last_start).step_by(params.block) {
        let block = block_start..(block_start + params.block).min(last_start + 1);
        for p in block.clone() {
            let mut entry = table[hash(read_u32(input, p), params.hash_log) as usize];
            let limit = (p + params.probe_len).min(match_limit);
            if params.depth <= 1 {
                // Level 1: one candidate.
                if entry == 0 {
                    continue;
                }
                let offset = p - (entry as usize - 1);
                if offset > MAX_OFFSET {
                    continue;
                }
                let mut end = p;
                while end < limit && input[end - offset] == input[end] {
                    end += 1;
                }
                if end - p >= MIN_MATCH {
                    matches[p] = Match {
                        len: (end - p) as u32,
                        offset: offset as u32,
                    };
                }
                continue;
            }
            chain[p] = entry;
            // The longest of up to `depth` candidates; ties go to the nearest.
            for _ in 0..params.depth {
                if entry == 0 {
                    break;
                }
                let candidate = entry as usize - 1;
                let offset = p - candidate;
                if offset > MAX_OFFSET {
                    break;
                }
                let mut end = p;
                while end < limit && input[end - offset] == input[end] {
                    end += 1;
                }
                if end - p >= MIN_MATCH && end - p > matches[p].len as usize {
                    matches[p] = Match {
                        len: (end - p) as u32,
                        offset: offset as u32,
                    };
                }
                entry = chain[candidate];
            }
        }
        for p in block {
            let bucket = &mut table[hash(read_u32(input, p), params.hash_log) as usize];
            *bucket = (*bucket).max(p as u32 + 1);
        }
    }
    matches
}

/// Lazy matching: skip the match at `p` (emit a literal) when the next
/// position's phase-1 match is longer. Phase-1 lengths are probe-capped, so
/// this needs no extra extension. Positions past the last match start have no
/// match (the GPU never writes them, so `p + 1` must be a valid start).
pub fn defer_match(matches: &[Match], p: usize, n: usize) -> bool {
    p + 1 + MFLIMIT <= n && matches[p + 1].len > matches[p].len
}

/// Phase 2: greedy (or, with `lazy`, lazy) parse over the phase-1 matches.
/// Each step depends only on the matches at `p` and `p + 1`, which is what lets
/// the GPU parse segments of a chunk in parallel.
pub fn parse(input: &[u8], matches: &[Match], lazy: bool) -> Vec<Sequence> {
    let n = input.len();
    let match_limit = n.saturating_sub(LAST_LITERALS);
    let mut sequences = Vec::new();
    let mut anchor = 0;
    let mut p = 0;
    while p + MFLIMIT <= n {
        let m = matches[p];
        if (m.len as usize) < MIN_MATCH || (lazy && defer_match(matches, p, n)) {
            p += 1;
            continue;
        }
        // Extend past the probe cap. A match that ended on a mismatch or the
        // limit in phase 1 stops again immediately.
        let offset = m.offset as usize;
        let mut end = p + m.len as usize;
        while end < match_limit && input[end - offset] == input[end] {
            end += 1;
        }
        sequences.push(Sequence {
            lit_start: anchor as u32,
            lit_len: (p - anchor) as u32,
            match_len: (end - p) as u32,
            offset: m.offset,
        });
        p = end;
        anchor = end;
    }
    sequences.push(Sequence {
        lit_start: anchor as u32,
        lit_len: (n - anchor) as u32,
        match_len: 0,
        offset: 0,
    });
    sequences
}

/// Phase 3: appends the LZ4 encoding of `sequences` to `out`.
pub fn emit(input: &[u8], sequences: &[Sequence], out: &mut Vec<u8>) {
    for s in sequences {
        let lit_len = s.lit_len as usize;
        let match_code = (s.match_len as usize).saturating_sub(MIN_MATCH);
        out.push(((lit_len.min(15) as u8) << 4) | match_code.min(15) as u8);
        write_length(out, lit_len);
        let start = s.lit_start as usize;
        out.extend_from_slice(&input[start..start + lit_len]);
        if s.match_len > 0 {
            out.extend_from_slice(&(s.offset as u16).to_le_bytes());
            write_length(out, match_code);
        }
    }
}

/// Writes the continuation bytes of a length whose 4-bit field is 15.
fn write_length(out: &mut Vec<u8>, len: usize) {
    if len < 15 {
        return;
    }
    let mut rest = len - 15;
    while rest >= 255 {
        out.push(255);
        rest -= 255;
    }
    out.push(rest as u8);
}

/// Compresses `input` as one LZ4 block.
pub fn encode_block(input: &[u8], params: &Params) -> Vec<u8> {
    let mut out = Vec::with_capacity(super::max_compressed_size(input.len()));
    emit(
        input,
        &parse(input, &find_matches(input, params), params.lazy),
        &mut out,
    );
    out
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::lz4::decode::decode_block;
    use crate::lz4::max_compressed_size;

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

    /// These tests hand-compute block-64 behaviour (positions 0..64 form the
    /// first match-finding block), independent of the default block size.
    fn block64() -> Params {
        Params {
            block: 64,
            ..Params::default()
        }
    }

    fn decode(block: &[u8], n: usize) -> Vec<u8> {
        let mut out = vec![0; n];
        decode_block(block, &mut out).unwrap();
        out
    }

    #[test]
    fn hash_takes_the_top_bits_of_a_multiplicative_hash() {
        assert_eq!(hash(1, 12), 2_654_435_761u32 >> 20);
        assert_eq!(
            hash(0xFFFF_FFFF, 16),
            0xFFFF_FFFFu32.wrapping_mul(2_654_435_761) >> 16
        );
    }

    #[test]
    fn finds_a_repeat_from_an_earlier_block() {
        let mut input = random(64, 1);
        input.extend_from_within(0..64);
        input.extend(random(32, 2));
        let m = find_matches(&input, &block64());
        assert_eq!(
            m[64],
            Match {
                len: 16,
                offset: 64
            }
        );
    }

    #[test]
    fn ignores_repeats_within_the_same_block() {
        let input = b"abcdefgh"
            .repeat(8)
            .into_iter()
            .chain(random(64, 3))
            .collect::<Vec<_>>();
        let m = find_matches(&input, &Params::default());
        assert!(m[..64].iter().all(|m| m.len == 0), "{:?}", &m[..64]);
    }

    #[test]
    fn latest_position_wins_a_bucket() {
        // Zeros: every position hashes alike; the candidate for block 2 is position 63.
        let m = find_matches(&[0u8; 200], &block64());
        assert_eq!(m[64].offset, 1);
        assert_eq!(m[100].offset, 37);
    }

    fn chained(depth: usize) -> Params {
        Params { depth, ..block64() }
    }

    /// "abcdefghijklmnop" at 0 (block 0), "abcd" then noise at 70 (block 1),
    /// and "abcdefghijklmnop" again at 140 (block 2).
    fn chain_input() -> Vec<u8> {
        let mut input = random(200, 21);
        input[..16].copy_from_slice(b"abcdefghijklmnop");
        input[70..74].copy_from_slice(b"abcd");
        input[140..156].copy_from_slice(b"abcdefghijklmnop");
        input
    }

    #[test]
    fn depth_one_takes_the_latest_candidate() {
        let m = find_matches(&chain_input(), &chained(1));
        assert_eq!(m[140], Match { len: 4, offset: 70 });
    }

    #[test]
    fn deeper_chains_find_older_longer_matches() {
        let m = find_matches(&chain_input(), &chained(2));
        assert_eq!(
            m[140],
            Match {
                len: 16,
                offset: 140
            }
        );
    }

    #[test]
    fn equal_lengths_go_to_the_nearest_candidate() {
        // "abcdefgh" + noise at 0 and at 70: both give 8 bytes at 140.
        let mut input = random(200, 22);
        input[..8].copy_from_slice(b"abcdefgh");
        input[70..78].copy_from_slice(b"abcdefgh");
        input[140..148].copy_from_slice(b"abcdefgh");
        let m = find_matches(&input, &chained(4));
        assert_eq!(m[140], Match { len: 8, offset: 70 });
    }

    #[test]
    fn chains_stop_beyond_max_offset() {
        // The older copy is 66 000 back: out of reach even with depth 4.
        let mut input = random(66_200, 23);
        input[..16].copy_from_slice(b"abcdefghijklmnop");
        input[66_010..66_014].copy_from_slice(b"abcd");
        input[66_100..66_116].copy_from_slice(b"abcdefghijklmnop");
        let m = find_matches(&input, &chained(4));
        assert_eq!(m[66_100], Match { len: 4, offset: 90 });
    }

    #[test]
    fn levels_add_hash_chain_candidates() {
        assert_eq!(Params::for_level(1), Params::default());
        assert_eq!(
            Params::for_level(2),
            Params {
                depth: 4,
                ..Params::default()
            }
        );
        assert_eq!(
            Params::for_level(3),
            Params {
                depth: 16,
                ..Params::default()
            }
        );
        assert_eq!(Params::for_level(0), Params::for_level(1));
        assert_eq!(Params::for_level(9), Params::for_level(3));
    }

    #[test]
    fn higher_levels_compress_smaller() {
        let input: Vec<u8> = (0..20_000u32)
            .flat_map(|i| format!("row {} col {}; ", (i * 7919) % 1013, i % 37).into_bytes())
            .take(65_536)
            .collect();
        let size = |level| encode_block(&input, &Params::for_level(level)).len();
        assert!(size(1) > size(2), "{} vs {}", size(1), size(2));
        assert!(size(2) > size(3), "{} vs {}", size(2), size(3));
    }

    #[test]
    fn matches_are_capped_at_probe_len() {
        let m = find_matches(&[0u8; 1000], &block64());
        assert!(m.iter().all(|m| m.len <= 16));
        assert_eq!(m[100].len, 16);
    }

    #[test]
    fn no_match_starts_in_the_last_mflimit_bytes() {
        let params = Params {
            block: 4,
            ..Params::default()
        };
        let m = find_matches(&[0u8; 200], &params);
        assert!(m[189..].iter().all(|m| m.len == 0), "{:?}", &m[189..]);
        assert!(m[188].len > 0);
    }

    #[test]
    fn matches_stop_before_the_last_literals() {
        let params = Params {
            block: 4,
            probe_len: 1000,
            ..Params::default()
        };
        let m = find_matches(&[0u8; 200], &params);
        for (p, m) in m.iter().enumerate() {
            assert!(
                m.len == 0 || p + m.len as usize <= 200 - LAST_LITERALS,
                "{p}: {m:?}"
            );
        }
    }

    #[test]
    fn candidates_beyond_max_offset_are_ignored() {
        let mut input = random(66_000, 4);
        input.extend_from_within(0..64);
        input.extend(random(64, 5));
        let params = Params {
            hash_log: 24,
            ..Params::default()
        };
        // Same 4 bytes as position 0, but 66 000 back.
        assert_eq!(find_matches(&input, &params)[66_000], Match::default());
    }

    #[test]
    fn parse_takes_matches_greedily_and_extends_capped_ones() {
        let input = [0u8; 1000];
        let seqs = parse(&input, &find_matches(&input, &block64()), false);
        assert_eq!(
            seqs,
            [
                Sequence {
                    lit_start: 0,
                    lit_len: 64,
                    match_len: 1000 - 5 - 64,
                    offset: 1
                },
                Sequence {
                    lit_start: 995,
                    lit_len: 5,
                    match_len: 0,
                    offset: 0
                },
            ]
        );
    }

    /// Block 0 holds "abcd" and, apart, "bcdefghij"; position 128 starts
    /// "abcdefghij", so it has a 4-byte match and position 129 a 9-byte one.
    fn lazy_input() -> Vec<u8> {
        let mut input = random(128, 11);
        input[10..14].copy_from_slice(b"abcd");
        input[30] = b'X';
        input[31..40].copy_from_slice(b"bcdefghij");
        input.extend_from_slice(b"abcdefghij");
        input.extend(random(64, 12));
        input
    }

    #[test]
    fn greedy_parse_takes_the_first_match() {
        let input = lazy_input();
        let seqs = parse(&input, &find_matches(&input, &block64()), false);
        assert_eq!(
            (seqs[0].lit_len, seqs[0].match_len, seqs[0].offset),
            (128, 4, 118)
        );
    }

    #[test]
    fn lazy_parse_skips_a_match_when_the_next_one_is_longer() {
        let input = lazy_input();
        let seqs = parse(&input, &find_matches(&input, &block64()), true);
        assert_eq!(
            (seqs[0].lit_len, seqs[0].match_len, seqs[0].offset),
            (129, 9, 98)
        );
        let mut out = Vec::new();
        emit(&input, &seqs, &mut out);
        assert_eq!(decode(&out, input.len()), input);
    }

    #[test]
    fn parse_without_matches_is_one_literal_run() {
        let input = random(100, 6);
        let seqs = parse(&input, &find_matches(&input, &Params::default()), true);
        assert_eq!(
            seqs,
            [Sequence {
                lit_start: 0,
                lit_len: 100,
                match_len: 0,
                offset: 0
            }]
        );
    }

    #[test]
    fn empty_input_encodes_to_a_single_zero_token() {
        assert_eq!(encode_block(&[], &Params::default()), [0x00]);
    }

    #[test]
    fn zeros_compress_to_almost_nothing() {
        let input = vec![0u8; 64 << 10];
        let block = encode_block(&input, &Params::default());
        assert!(block.len() < 400, "{}", block.len());
        assert_eq!(decode(&block, input.len()), input);
    }

    #[test]
    fn random_data_stays_within_the_worst_case_bound() {
        let input = random(64 << 10, 7);
        let block = encode_block(&input, &Params::default());
        assert!(block.len() <= max_compressed_size(input.len()));
        assert_eq!(decode(&block, input.len()), input);
    }

    /// Compressible data: random-length runs drawn from a small alphabet,
    /// with copies of earlier slices.
    fn compressible() -> impl Strategy<Value = Vec<u8>> {
        prop::collection::vec((0u8..4, 1usize..40, any::<bool>()), 0..300).prop_map(|parts| {
            let mut out: Vec<u8> = Vec::new();
            for (byte, len, copy) in parts {
                if copy && out.len() > len {
                    let start = out.len() - len - (byte as usize * 7 % (out.len() - len + 1));
                    out.extend_from_within(start..start + len);
                } else {
                    out.extend(std::iter::repeat_n(b'a' + byte, len));
                }
            }
            out
        })
    }

    fn params() -> impl Strategy<Value = Params> {
        (
            prop_oneof![Just(4usize), Just(64), Just(256)],
            8u32..=14,
            4usize..=64,
            1usize..=8,
        )
            .prop_map(|(block, hash_log, probe_len, depth)| Params {
                block,
                hash_log,
                probe_len,
                depth,
                ..Params::default()
            })
    }

    proptest! {
        #[test]
        fn encoded_blocks_round_trip_through_both_decoders(input in compressible(), params in params()) {
            let block = encode_block(&input, &params);
            prop_assert_eq!(decode(&block, input.len()), input.clone());
            prop_assert_eq!(lz4_flex::block::decompress(&block, input.len()).unwrap(), input);
        }

        #[test]
        fn parse_obeys_lz4_end_of_block_rules(input in compressible(), params in params()) {
            let n = input.len();
            let seqs = parse(&input, &find_matches(&input, &params), params.lazy);
            let (last, body) = seqs.split_last().unwrap();
            prop_assert_eq!(last.match_len, 0);
            prop_assert_eq!((last.lit_start + last.lit_len) as usize, n);
            for s in body {
                let start = (s.lit_start + s.lit_len) as usize;
                prop_assert!(s.match_len as usize >= MIN_MATCH);
                prop_assert!(start + MFLIMIT <= n, "match starts too late: {:?}", s);
                prop_assert!(start + s.match_len as usize <= n - LAST_LITERALS, "{:?}", s);
                prop_assert!((1..=MAX_OFFSET as u32).contains(&s.offset));
            }
            prop_assert!(n < LAST_LITERALS || last.lit_len as usize >= LAST_LITERALS.min(n));
        }
    }
}
