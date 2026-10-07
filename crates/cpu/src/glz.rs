//! GLZ (codec 2): an LZ77 block format designed for GPUs. A sequence's
//! fields live in separate arrays rather than one interleaved token stream, so
//! any sequence's lengths, output position and literal source are a prefix sum
//! away and both directions parallelise without a serial parse. Layout
//! (little-endian), see `format/FORMAT.md`:
//!
//! ```text
//! u32      seq_count | WIDE_BIT   (WIDE_BIT: extension values are u32, else u16)
//! u32      ext_count              number of extension values
//! tokens   [seq_count] u8         lit nibble << 4 | match nibble; padded to 4 bytes
//! offsets  [seq_count] u16        padded to 4 bytes; 0 in the last sequence
//! ext      [ext_count] u16/u32    padded to 4 bytes
//! literals [sum(lit_len)] u8      all literal bytes, in order
//! ```
//!
//! As in LZ4, the literal nibble is `min(lit_len, 15)` and the match nibble is
//! `min(match_len - 4, 15)`; a nibble of 15 takes the next extension value
//! (literal before match within a sequence) and adds it. The last sequence is
//! literals only: its match nibble and offset are 0. A typical sequence costs
//! 3 bytes (token + offset), like LZ4, but a sequence's extension slots are an
//! exclusive prefix sum of escape counts, so decoding needs no serial pass.
//!
//! The encoder reuses the greedy (GPU-twin) match finder. Optionally it
//! refuses matches whose source overlaps another match's output in the same
//! group of `G` sequences, so a decoder can resolve a whole group in one
//! parallel step (Gompresso's dependency elimination).

use crate::lz4::encode::{find_matches, Params, Sequence};
use crate::lz4::{LAST_LITERALS, MFLIMIT, MIN_MATCH};

pub const WIDE_BIT: u32 = 1 << 31;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlzParams {
    /// Match finding (same as the LZ4 greedy encoder).
    pub lz: Params,
    /// Dependency elimination: no match may copy from another match's output
    /// within the same group of this many sequences.
    pub independent_groups: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GlzError {
    #[error("block is shorter than its headers or literals claim")]
    Truncated,
    #[error("invalid sequence (count, match length or final sequence)")]
    BadSequence,
    #[error("match offset is zero")]
    ZeroOffset,
    #[error("match offset points before the start of the block")]
    OffsetBeforeStart,
    #[error("block decodes to more bytes than expected")]
    OutputOverflow,
    #[error("block decodes to a different size, or has unused literal bytes")]
    SizeMismatch,
}

/// Greedy parse over phase-1 matches, as in `lz4::encode::parse`, optionally
/// refusing matches that would depend on another match in the same group.
pub fn parse(
    input: &[u8],
    matches: &[crate::lz4::encode::Match],
    groups: Option<u32>,
) -> Vec<Sequence> {
    let n = input.len();
    let match_limit = n.saturating_sub(LAST_LITERALS);
    let group_size = groups.map(|g| g.max(1) as usize);
    let mut sequences = Vec::new();
    // Output ranges of the matches in the current group.
    let mut group_outputs: Vec<(usize, usize)> = Vec::new();
    let mut anchor = 0;
    let mut p = 0;
    while p + MFLIMIT <= n {
        let m = matches[p];
        if (m.len as usize) < MIN_MATCH {
            p += 1;
            continue;
        }
        let offset = m.offset as usize;
        // With dependency elimination, the match's source pattern
        // [src, src + min(offset, len)) must avoid this group's match outputs,
        // so the match may be capped (or, below MIN_MATCH, dropped).
        let mut cap = usize::MAX;
        if let Some(g) = group_size {
            if sequences.len() % g == 0 {
                group_outputs.clear();
            }
            match source_room(&group_outputs, p - offset) {
                Some(room) if offset > room => cap = room,
                Some(_) => {}
                None => {
                    p += 1;
                    continue;
                }
            }
            if cap < MIN_MATCH {
                p += 1;
                continue;
            }
        }
        let limit = match_limit.min(p.saturating_add(cap));
        let mut end = p + (m.len as usize).min(cap);
        while end < limit && input[end - offset] == input[end] {
            end += 1;
        }
        if group_size.is_some() {
            group_outputs.push((p, end));
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

/// How many bytes from `src` on are free of the group's match outputs
/// (ascending, disjoint ranges): `None` if `src` itself is inside one,
/// `usize::MAX` if no output starts after it.
fn source_room(outputs: &[(usize, usize)], src: usize) -> Option<usize> {
    // First output that ends after `src` (outputs are sorted).
    match outputs.get(outputs.partition_point(|&(_, b)| b <= src)) {
        None => Some(usize::MAX),
        Some(&(a, _)) if a <= src => None,
        Some(&(a, _)) => Some(a - src),
    }
}

fn pad4(v: &mut Vec<u8>) {
    v.resize(v.len().div_ceil(4) * 4, 0);
}

/// Byte offsets of the arrays: (tokens, offsets, ext, literals).
fn field_offsets(count: usize, ext_count: usize, wide: bool) -> (usize, usize, usize, usize) {
    let pad = |n: usize| n.div_ceil(4) * 4;
    let tokens = 8;
    let offsets = tokens + pad(count);
    let ext = offsets + pad(count * 2);
    let literals = ext + pad(ext_count * if wide { 4 } else { 2 });
    (tokens, offsets, ext, literals)
}

/// The nibble for a length, and its extension value if it escapes.
fn nibble(len: u32) -> (u8, Option<u32>) {
    if len >= 15 {
        (15, Some(len - 15))
    } else {
        (len as u8, None)
    }
}

/// Serialises `sequences` (literals taken from `input`) as a GLZ block.
pub fn emit(input: &[u8], sequences: &[Sequence]) -> Vec<u8> {
    let mut tokens = Vec::with_capacity(sequences.len());
    let mut ext = Vec::new();
    for s in sequences {
        let (lit, lit_ext) = nibble(s.lit_len);
        let (mat, mat_ext) = if s.match_len > 0 {
            nibble(s.match_len - MIN_MATCH as u32)
        } else {
            (0, None)
        };
        tokens.push(lit << 4 | mat);
        ext.extend(lit_ext);
        ext.extend(mat_ext);
    }
    let wide = ext.iter().any(|&v| v > 0xFFFF);
    let mut out = (sequences.len() as u32 | if wide { WIDE_BIT } else { 0 })
        .to_le_bytes()
        .to_vec();
    out.extend_from_slice(&(ext.len() as u32).to_le_bytes());
    out.extend_from_slice(&tokens);
    pad4(&mut out);
    for s in sequences {
        out.extend_from_slice(&(s.offset as u16).to_le_bytes());
    }
    pad4(&mut out);
    for &v in &ext {
        if wide {
            out.extend_from_slice(&v.to_le_bytes());
        } else {
            out.extend_from_slice(&(v as u16).to_le_bytes());
        }
    }
    pad4(&mut out);
    for s in sequences {
        let start = s.lit_start as usize;
        out.extend_from_slice(&input[start..start + s.lit_len as usize]);
    }
    out
}

/// Compresses `input` as one GLZ block.
pub fn encode_block(input: &[u8], params: &GlzParams) -> Vec<u8> {
    let matches = find_matches(input, &params.lz);
    emit(input, &parse(input, &matches, params.independent_groups))
}

/// Decodes one GLZ block into `dst` (exactly the uncompressed size). Header
/// checks come first, then checks run sequence by sequence in a fixed order;
/// the GPU decoder reports the same error for the first failing sequence.
pub fn decode_block(src: &[u8], dst: &mut [u8]) -> Result<(), GlzError> {
    let word = |at: usize| {
        src.get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    };
    let (Some(head), Some(ext_count)) = (word(0), word(4)) else {
        return Err(GlzError::Truncated);
    };
    let count = (head & !WIDE_BIT) as usize;
    let ext_count = ext_count as usize;
    let wide = head & WIDE_BIT != 0;
    // Every sequence but the last outputs >= MIN_MATCH bytes, and each has at
    // most two extension values.
    if count == 0 || count > dst.len() / MIN_MATCH + 1 || ext_count > 2 * count {
        return Err(GlzError::BadSequence);
    }
    let (tokens_at, offsets_at, ext_at, literals_at) = field_offsets(count, ext_count, wide);
    if literals_at > src.len() {
        return Err(GlzError::Truncated);
    }
    let tokens = &src[tokens_at..tokens_at + count];
    let escapes: usize = tokens
        .iter()
        .map(|t| usize::from(t >> 4 == 15) + usize::from(t & 15 == 15))
        .sum();
    if escapes != ext_count {
        return Err(GlzError::BadSequence);
    }
    let ext = |k: usize| -> usize {
        if wide {
            u32::from_le_bytes(src[ext_at + 4 * k..ext_at + 4 * k + 4].try_into().unwrap()) as usize
        } else {
            usize::from(u16::from_le_bytes([
                src[ext_at + 2 * k],
                src[ext_at + 2 * k + 1],
            ]))
        }
    };
    let literals = &src[literals_at..];
    let (mut lp, mut op, mut next_ext) = (0usize, 0usize, 0usize);
    for (i, &token) in tokens.iter().enumerate() {
        let mut length = |nibble: u8| {
            let mut len = usize::from(nibble);
            if nibble == 15 {
                len += ext(next_ext);
                next_ext += 1;
            }
            len
        };
        let lit_len = length(token >> 4);
        let match_code = length(token & 15);
        let offset = usize::from(u16::from_le_bytes([
            src[offsets_at + 2 * i],
            src[offsets_at + 2 * i + 1],
        ]));
        if lit_len > literals.len() - lp {
            return Err(GlzError::Truncated);
        }
        if lit_len > dst.len() - op {
            return Err(GlzError::OutputOverflow);
        }
        dst[op..op + lit_len].copy_from_slice(&literals[lp..lp + lit_len]);
        lp += lit_len;
        op += lit_len;
        if i == count - 1 {
            if match_code != 0 || offset != 0 {
                return Err(GlzError::BadSequence);
            }
            break;
        }
        let match_len = match_code + MIN_MATCH;
        if offset == 0 {
            return Err(GlzError::ZeroOffset);
        }
        if offset > op {
            return Err(GlzError::OffsetBeforeStart);
        }
        if match_len > dst.len() - op {
            return Err(GlzError::OutputOverflow);
        }
        for k in op..op + match_len {
            dst[k] = dst[k - offset];
        }
        op += match_len;
    }
    if op != dst.len() || lp != literals.len() {
        return Err(GlzError::SizeMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

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

    fn decode(block: &[u8], n: usize) -> Result<Vec<u8>, GlzError> {
        let mut out = vec![0xEE; n];
        decode_block(block, &mut out).map(|()| out)
    }

    fn seq(lit_start: u32, lit_len: u32, match_len: u32, offset: u32) -> Sequence {
        Sequence {
            lit_start,
            lit_len,
            match_len,
            offset,
        }
    }

    /// Builds a narrow block from raw fields, for malformed-input tests.
    fn raw_block(tokens: &[u8], offsets: &[u16], ext: &[u16], literals: &[u8]) -> Vec<u8> {
        let mut out = (tokens.len() as u32).to_le_bytes().to_vec();
        out.extend_from_slice(&(ext.len() as u32).to_le_bytes());
        out.extend_from_slice(tokens);
        out.resize(out.len().div_ceil(4) * 4, 0);
        for array in [offsets, ext] {
            for v in array {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.resize(out.len().div_ceil(4) * 4, 0);
        }
        out.extend_from_slice(literals);
        out
    }

    #[test]
    fn layout_matches_the_spec() {
        // "abcd" + match(4 @ 4) + "xy"  ->  abcdabcdxy
        let input = b"abcdabcdxy";
        let block = emit(input, &[seq(0, 4, 4, 4), seq(8, 2, 0, 0)]);
        let expected = [
            &2u32.to_le_bytes()[..], // count, narrow
            &0u32.to_le_bytes(),     // no extension values
            &[0x40, 0x20, 0, 0],     // tokens: (4 lits, match 4), (2 lits)
            &[4, 0, 0, 0],           // offsets
            b"abcdxy",               // literals
        ]
        .concat();
        assert_eq!(block, expected);
        assert_eq!(decode(&block, 10).unwrap(), input);
    }

    #[test]
    fn long_lengths_escape_to_extension_values() {
        // 20 literals, match of 4 + 15 + 3 = 22 at offset 1, then 1 literal.
        let input = [&[7u8; 20][..], &[7; 22], b"!"].concat();
        let block = emit(&input, &[seq(0, 20, 22, 1), seq(42, 1, 0, 0)]);
        let expected = [
            &2u32.to_le_bytes()[..],
            &2u32.to_le_bytes(), // two extension values
            &[0xFF, 0x10, 0, 0], // both nibbles escaped; then (1 lit)
            &[1, 0, 0, 0],       // offsets
            &[5, 0, 3, 0],       // ext: lit 20 - 15, match (22 - 4) - 15
            &input[..20],
            b"!",
        ]
        .concat();
        assert_eq!(block, expected);
        assert_eq!(decode(&block, input.len()).unwrap(), input);
    }

    #[test]
    fn odd_counts_pad_each_array_to_four_bytes() {
        let block = emit(b"hello", &[seq(0, 5, 0, 0)]);
        assert_eq!(block.len(), 8 + 4 + 4 + 5);
        assert_eq!(decode(&block, 5).unwrap(), b"hello");
    }

    #[test]
    fn long_lengths_switch_to_wide_fields() {
        let input = vec![0u8; 1 << 20];
        let block = encode_block(&input, &GlzParams::default());
        let head = u32::from_le_bytes(block[..4].try_into().unwrap());
        assert_ne!(head & WIDE_BIT, 0, "a ~1 MiB match needs u32 lengths");
        assert_eq!(decode(&block, input.len()).unwrap(), input);
    }

    #[test]
    fn parse_without_groups_matches_the_lz4_parse() {
        let input = [b"the cat sat on the mat. ".repeat(40), random(500, 3)].concat();
        let matches = find_matches(&input, &Params::default());
        assert_eq!(
            parse(&input, &matches, None),
            crate::lz4::encode::parse(&input, &matches)
        );
    }

    #[test]
    fn independent_groups_never_copy_from_a_group_mates_match() {
        let input = b"abcabcabcabc-xyz-".repeat(200);
        let matches = find_matches(&input, &Params::default());
        let seqs = parse(&input, &matches, Some(4));
        assert_independent(&seqs, 4);
        let block = emit(&input, &seqs);
        assert_eq!(decode(&block, input.len()).unwrap(), input);
    }

    fn assert_independent(seqs: &[Sequence], g: usize) {
        for group in seqs.chunks(g) {
            let outputs: Vec<(u32, u32)> = group
                .iter()
                .filter(|s| s.match_len > 0)
                .map(|s| {
                    (
                        s.lit_start + s.lit_len,
                        s.lit_start + s.lit_len + s.match_len,
                    )
                })
                .collect();
            for s in group.iter().filter(|s| s.match_len > 0) {
                let m = s.lit_start + s.lit_len;
                let (src, src_end) = (m - s.offset, m - s.offset + s.offset.min(s.match_len));
                for &(a, b) in &outputs {
                    if a != m {
                        assert!(
                            src_end <= a || b <= src,
                            "{s:?} reads a group mate's output {a}..{b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_empty_input() {
        assert_eq!(decode(&[], 0), Err(GlzError::Truncated));
    }

    #[test]
    fn rejects_zero_sequences() {
        assert_eq!(
            decode(&raw_block(&[], &[], &[], &[]), 0),
            Err(GlzError::BadSequence)
        );
    }

    #[test]
    fn rejects_arrays_past_the_end() {
        let mut block = raw_block(&[0x10, 0x00], &[1, 0], &[], b"a");
        block.truncate(14);
        assert_eq!(decode(&block, 5), Err(GlzError::Truncated));
    }

    #[test]
    fn rejects_extension_count_mismatch() {
        let mut block = raw_block(&[0xF0], &[0], &[0], &[0; 15]);
        block[4] = 2; // claims two extension values, the tokens escape once
        assert_eq!(decode(&block, 15), Err(GlzError::BadSequence));
    }

    #[test]
    fn rejects_more_literals_than_present() {
        assert_eq!(
            decode(&raw_block(&[0x30], &[0], &[], b"ab"), 3),
            Err(GlzError::Truncated)
        );
    }

    #[test]
    fn rejects_literals_overflowing_output() {
        assert_eq!(
            decode(&raw_block(&[0x30], &[0], &[], b"abc"), 2),
            Err(GlzError::OutputOverflow)
        );
    }

    #[test]
    fn rejects_a_match_in_the_final_sequence() {
        assert_eq!(
            decode(&raw_block(&[0x11], &[1], &[], b"a"), 6),
            Err(GlzError::BadSequence)
        );
        assert_eq!(
            decode(&raw_block(&[0x10], &[1], &[], b"a"), 1),
            Err(GlzError::BadSequence)
        );
    }

    #[test]
    fn rejects_zero_offset() {
        assert_eq!(
            decode(&raw_block(&[0x10, 0x00], &[0, 0], &[], b"a"), 5),
            Err(GlzError::ZeroOffset)
        );
    }

    #[test]
    fn rejects_offset_before_start() {
        assert_eq!(
            decode(&raw_block(&[0x10, 0x00], &[2, 0], &[], b"a"), 5),
            Err(GlzError::OffsetBeforeStart)
        );
    }

    #[test]
    fn rejects_match_overflowing_output() {
        assert_eq!(
            decode(&raw_block(&[0x10, 0x00], &[1, 0], &[], b"a"), 4),
            Err(GlzError::OutputOverflow)
        );
    }

    #[test]
    fn rejects_short_output_or_unused_literals() {
        assert_eq!(
            decode(&raw_block(&[0x20], &[0], &[], b"ab"), 3),
            Err(GlzError::SizeMismatch)
        );
        assert_eq!(
            decode(&raw_block(&[0x20], &[0], &[], b"abc"), 2),
            Err(GlzError::SizeMismatch)
        );
    }

    #[test]
    fn rejects_implausibly_many_sequences() {
        // Every sequence but the last outputs >= 4 bytes, so 4 output bytes
        // allow at most 2 sequences.
        let block = raw_block(&[0x00, 0x00, 0x40], &[1, 1, 0], &[], b"abcd");
        assert_eq!(decode(&block, 4), Err(GlzError::BadSequence));
    }

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

    proptest! {
        #[test]
        fn blocks_round_trip(input in compressible(), groups in prop_oneof![Just(None), (1u32..=64).prop_map(Some)]) {
            let params = GlzParams { independent_groups: groups, ..GlzParams::default() };
            let block = encode_block(&input, &params);
            prop_assert_eq!(decode(&block, input.len()).unwrap(), input.clone());
            if let Some(g) = groups {
                let matches = find_matches(&input, &params.lz);
                assert_independent(&parse(&input, &matches, Some(g)), g as usize);
            }
        }

        #[test]
        fn garbage_never_panics(block in prop::collection::vec(any::<u8>(), 0..200), n in 0usize..300) {
            let _ = decode(&block, n);
        }
    }
}
