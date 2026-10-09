//! Order-0 Huffman streams for GLZ-E (FORMAT.md, "GLZ-E block"): a stream of
//! `n` byte symbols is stored raw, as a single repeated symbol (RLE), or
//! Huffman-coded in up to 32 independent lanes. Everything here is integer
//! arithmetic with fixed tie-breaks, so the GPU encoder reproduces it exactly.

/// Longest code length: the decode table has `1 << MAX_LEN` entries.
pub const MAX_LEN: u32 = 11;
/// Most lanes per stream (one SIMD group).
pub const MAX_LANES: usize = 32;
/// A lane is added per this many symbols, up to `MAX_LANES`.
pub const LANE_SYMBOLS: usize = 512;

pub const MODE_RAW: u32 = 0;
pub const MODE_RLE: u32 = 1;
pub const MODE_HUFFMAN: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum StreamError {
    #[error("stream is shorter than its header, table or lanes claim")]
    Truncated,
    #[error("unknown stream mode or nonzero reserved bits")]
    BadMode,
    #[error("code lengths over the limit or not a complete prefix code")]
    BadTable,
    #[error("a lane decodes past the end of its words")]
    LaneOverrun,
}

/// Lanes of a Huffman stream of `n` symbols.
pub fn lanes(n: usize) -> usize {
    n.div_ceil(LANE_SYMBOLS).clamp(1, MAX_LANES)
}

/// Symbols `[start, end)` of lane `k` of `lanes` in an `n`-symbol stream.
pub fn lane_range(n: usize, lanes: usize, k: usize) -> std::ops::Range<usize> {
    k * n / lanes..(k + 1) * n / lanes
}

/// Deepest unlimited tree: with at most 2^20 symbols, Fibonacci weights stop
/// growing the tree before depth 29.
const MAX_DEPTH: usize = 32;

/// Code lengths (0 = unused, else 1..=MAX_LEN) for symbol `counts` with at
/// least two used symbols and a total below 2^20 + 1.
///
/// 1. Used symbols sorted by (count, symbol).
/// 2. Two-queue Huffman: repeatedly merge the two lightest of the leaf queue
///    and the internal-node queue, taking the leaf on equal weights.
/// 3. Depths over MAX_LEN are folded back (JPEG Annex K.3, which keeps the
///    code complete): per overlong length i, two codes move up to i - 1 and a
///    code at the deepest j < i - 1 splits into two at j + 1.
/// 4. Lengths are reassigned from the per-length counts in sorted order:
///    the longest codes to the first (least frequent) symbols.
pub fn code_lengths(counts: &[u32; 256]) -> [u8; 256] {
    let mut syms: Vec<usize> = (0..256).filter(|&s| counts[s] > 0).collect();
    syms.sort_by_key(|&s| (counts[s], s));
    let m = syms.len();
    assert!(m >= 2, "Huffman coding needs two used symbols");
    // Internal nodes in creation order: weight and parent. Leaves' parents.
    let mut weight: Vec<u32> = Vec::with_capacity(m - 1);
    let mut node_parent = vec![0usize; m - 1];
    let mut leaf_parent = vec![0usize; m];
    let (mut li, mut ni) = (0, 0);
    for k in 0..m - 1 {
        let mut pick = || {
            if li < m && (ni >= weight.len() || counts[syms[li]] <= weight[ni]) {
                li += 1;
                (true, li - 1, counts[syms[li - 1]])
            } else {
                ni += 1;
                (false, ni - 1, weight[ni - 1])
            }
        };
        let a = pick();
        let b = pick();
        for (leaf, i, _) in [a, b] {
            if leaf {
                leaf_parent[i] = k;
            } else {
                node_parent[i] = k;
            }
        }
        weight.push(a.2 + b.2);
    }
    // Parents are created after their children, so depths go root-first.
    let mut depth = vec![0usize; m - 1];
    for k in (0..m - 2).rev() {
        depth[k] = depth[node_parent[k]] + 1;
    }
    let mut bl = [0u32; MAX_DEPTH + 1];
    for &p in &leaf_parent {
        bl[depth[p] + 1] += 1;
    }
    let max_len = MAX_LEN as usize;
    for i in (max_len + 1..=MAX_DEPTH).rev() {
        while bl[i] > 0 {
            let mut j = i - 2;
            while bl[j] == 0 {
                j -= 1;
            }
            bl[i] -= 2;
            bl[i - 1] += 1;
            bl[j + 1] += 2;
            bl[j] -= 1;
        }
    }
    let mut lengths = [0u8; 256];
    let mut next = syms.iter();
    for len in (1..=max_len).rev() {
        for _ in 0..bl[len] {
            lengths[*next.next().unwrap()] = len as u8;
        }
    }
    lengths
}

/// Canonical codes for `lengths` (shorter codes first, then by symbol),
/// bit-reversed so they can be written least significant bit first.
pub fn codes(lengths: &[u8; 256]) -> [u16; 256] {
    let mut bl = [0u32; MAX_LEN as usize + 1];
    for &l in lengths.iter().filter(|&&l| l > 0) {
        bl[l as usize] += 1;
    }
    let mut next = [0u32; MAX_LEN as usize + 1];
    let mut code = 0;
    for len in 1..=MAX_LEN as usize {
        code = (code + bl[len - 1]) << 1;
        next[len] = code;
    }
    let mut codes = [0u16; 256];
    for s in 0..256 {
        let len = u32::from(lengths[s]);
        if len > 0 {
            codes[s] = (next[len as usize].reverse_bits() >> (32 - len)) as u16;
            next[len as usize] += 1;
        }
    }
    codes
}

/// Appends the smallest encoding of `symbols` (raw, RLE or Huffman); all
/// three are whole words.
pub fn encode_stream(symbols: &[u8], out: &mut Vec<u8>) {
    let n = symbols.len();
    let mut counts = [0u32; 256];
    for &s in symbols {
        counts[s as usize] += 1;
    }
    let used = counts.iter().filter(|&&c| c > 0).count();
    if used == 1 {
        out.extend_from_slice(&(MODE_RLE | u32::from(symbols[0]) << 8).to_le_bytes());
        return;
    }
    let raw_size = 4 + n.next_multiple_of(4);
    if used >= 2 {
        let lengths = code_lengths(&counts);
        let lanes = lanes(n);
        let words: Vec<usize> = (0..lanes)
            .map(|k| {
                let bits: usize = symbols[lane_range(n, lanes, k)]
                    .iter()
                    .map(|&s| usize::from(lengths[s as usize]))
                    .sum();
                bits.div_ceil(32)
            })
            .collect();
        let huffman_size =
            4 + 128 + (2 * lanes).next_multiple_of(4) + 4 * words.iter().sum::<usize>();
        if huffman_size < raw_size {
            let codes = codes(&lengths);
            out.extend_from_slice(&MODE_HUFFMAN.to_le_bytes());
            out.extend(lengths.chunks(2).map(|p| p[0] | p[1] << 4));
            for &w in &words {
                out.extend_from_slice(&(w as u16).to_le_bytes());
            }
            out.resize(out.len() + (2 * lanes).next_multiple_of(4) - 2 * lanes, 0);
            for k in 0..lanes {
                let (mut acc, mut fill) = (0u64, 0u32);
                for &s in &symbols[lane_range(n, lanes, k)] {
                    acc |= u64::from(codes[s as usize]) << fill;
                    fill += u32::from(lengths[s as usize]);
                    if fill >= 32 {
                        out.extend_from_slice(&(acc as u32).to_le_bytes());
                        acc >>= 32;
                        fill -= 32;
                    }
                }
                if fill > 0 {
                    out.extend_from_slice(&(acc as u32).to_le_bytes());
                }
            }
            return;
        }
    }
    out.extend_from_slice(&MODE_RAW.to_le_bytes());
    out.extend_from_slice(symbols);
    out.resize(out.len() + raw_size - 4 - n, 0);
}

/// Decodes the `n`-symbol stream starting at `src[at..]` onto `out`, and
/// returns the position after it. Checks, in order: the mode word, the code
/// table (present, then valid), the lane sizes and words (present), then each
/// lane in order (no bits past its words).
pub fn decode_stream(
    src: &[u8],
    at: usize,
    n: usize,
    out: &mut Vec<u8>,
) -> Result<usize, StreamError> {
    let word = |at: usize| {
        src.get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .ok_or(StreamError::Truncated)
    };
    let head = word(at)?;
    match head & 0xFF {
        MODE_RAW if head >> 8 == 0 => {
            let end = at + 4 + n.next_multiple_of(4);
            let bytes = src.get(at + 4..at + 4 + n).ok_or(StreamError::Truncated)?;
            if end > src.len() {
                return Err(StreamError::Truncated);
            }
            out.extend_from_slice(bytes);
            Ok(end)
        }
        MODE_RLE if head >> 16 == 0 => {
            out.resize(out.len() + n, (head >> 8) as u8);
            Ok(at + 4)
        }
        MODE_HUFFMAN if head >> 8 == 0 => {
            let table = src.get(at + 4..at + 132).ok_or(StreamError::Truncated)?;
            let mut lengths = [0u8; 256];
            for (i, &b) in table.iter().enumerate() {
                lengths[2 * i] = b & 15;
                lengths[2 * i + 1] = b >> 4;
            }
            let kraft: u32 = lengths
                .iter()
                .filter(|&&l| l > 0)
                .map(|&l| 1u32 << (MAX_LEN.saturating_sub(u32::from(l))))
                .sum();
            if lengths.iter().any(|&l| u32::from(l) > MAX_LEN) || kraft != 1 << MAX_LEN {
                return Err(StreamError::BadTable);
            }
            let lanes = lanes(n);
            let sizes_at = at + 132;
            let sizes = src
                .get(sizes_at..sizes_at + 2 * lanes)
                .ok_or(StreamError::Truncated)?;
            let words: Vec<usize> = sizes
                .chunks(2)
                .map(|p| usize::from(u16::from_le_bytes([p[0], p[1]])))
                .collect();
            let data_at = sizes_at + (2 * lanes).next_multiple_of(4);
            let end = data_at + 4 * words.iter().sum::<usize>();
            if end > src.len() {
                return Err(StreamError::Truncated);
            }
            // Decode table: entry = symbol | length << 8, indexed by the next
            // MAX_LEN bits (least significant first).
            let codes = codes(&lengths);
            let mut lookup = vec![0u16; 1 << MAX_LEN];
            for s in 0..256 {
                let len = u32::from(lengths[s]);
                if len > 0 {
                    for fill in 0..1u32 << (MAX_LEN - len) {
                        lookup[(u32::from(codes[s]) | fill << len) as usize] =
                            s as u16 | (len as u16) << 8;
                    }
                }
            }
            let mut lane_at = data_at;
            for (k, &w) in words.iter().enumerate() {
                let lane = &src[lane_at..lane_at + 4 * w];
                let bit = |i: usize| -> u32 {
                    lane.get(i / 32 * 4..i / 32 * 4 + 4)
                        .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
                };
                let mut pos = 0usize;
                for _ in lane_range(n, lanes, k) {
                    let shift = (pos % 32) as u32;
                    let mut peek = bit(pos) >> shift;
                    if shift > 0 {
                        peek |= bit(pos + 32) << (32 - shift);
                    }
                    let e = lookup[(peek & ((1 << MAX_LEN) - 1)) as usize];
                    pos += usize::from(e >> 8);
                    if pos > 32 * w {
                        return Err(StreamError::LaneOverrun);
                    }
                    out.push(e as u8);
                }
                lane_at += 4 * w;
            }
            Ok(end)
        }
        _ => Err(StreamError::BadMode),
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn counts(pairs: &[(u8, u32)]) -> [u32; 256] {
        let mut c = [0; 256];
        for &(s, n) in pairs {
            c[s as usize] = n;
        }
        c
    }

    fn kraft(lengths: &[u8; 256]) -> u32 {
        lengths
            .iter()
            .filter(|&&l| l > 0)
            .map(|&l| 1 << (MAX_LEN - u32::from(l)))
            .sum()
    }

    fn round_trip(symbols: &[u8]) -> Vec<u8> {
        let mut enc = vec![0xEE; 3]; // streams may start anywhere
        encode_stream(symbols, &mut enc);
        let mut out = Vec::new();
        let end = decode_stream(&enc, 3, symbols.len(), &mut out).unwrap();
        assert_eq!(end, enc.len(), "decoder stops where the stream ends");
        assert_eq!(out, symbols);
        enc[3..].to_vec()
    }

    fn text(n: usize) -> Vec<u8> {
        b"the quick brown fox jumps over the lazy dog; "
            .iter()
            .cycle()
            .take(n)
            .copied()
            .collect()
    }

    #[test]
    fn lanes_grow_with_the_stream() {
        assert_eq!(lanes(0), 1);
        assert_eq!(lanes(512), 1);
        assert_eq!(lanes(513), 2);
        assert_eq!(lanes(1 << 20), 32);
    }

    #[test]
    fn classic_distribution_gets_classic_lengths() {
        let l = code_lengths(&counts(&[(0, 1), (1, 1), (2, 2), (3, 4)]));
        assert_eq!(&l[..4], &[3, 3, 2, 1]);
        assert!(l[4..].iter().all(|&x| x == 0));
    }

    #[test]
    fn ties_give_the_shortest_code_to_the_highest_symbol() {
        let l = code_lengths(&counts(&[(5, 9), (6, 9), (7, 9)]));
        assert_eq!(&l[5..8], &[2, 2, 1]);
    }

    #[test]
    fn lengths_are_limited_and_complete() {
        // Fibonacci counts make the unlimited tree as deep as possible.
        let (mut a, mut b) = (1u32, 1u32);
        let mut c = [0; 256];
        for s in c.iter_mut().take(30) {
            *s = a;
            (a, b) = (b, a + b);
        }
        let l = code_lengths(&c);
        assert!(l.iter().all(|&x| u32::from(x) <= MAX_LEN));
        assert_eq!(kraft(&l), 1 << MAX_LEN);
    }

    #[test]
    fn canonical_codes_are_bit_reversed() {
        // Lengths 3, 3, 2, 1: codes 110, 111, 10, 0 (MSB first).
        let mut l = [0; 256];
        l[..4].copy_from_slice(&[3, 3, 2, 1]);
        let c = codes(&l);
        assert_eq!(&c[..4], &[0b011, 0b111, 0b01, 0b0]);
    }

    #[test]
    fn small_and_degenerate_streams() {
        assert_eq!(round_trip(&[]), MODE_RAW.to_le_bytes());
        assert_eq!(
            round_trip(&[7; 1000]),
            (MODE_RLE | 7 << 8).to_le_bytes(),
            "one symbol: RLE"
        );
        let mut s = 1u64;
        let random: Vec<u8> = (0..1000)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s as u8
            })
            .collect();
        let enc = round_trip(&random);
        assert_eq!(enc[0], MODE_RAW as u8, "random data stays raw");
        assert_eq!(enc.len(), 4 + 1000);
    }

    #[test]
    fn skewed_streams_are_huffman_coded_and_smaller() {
        let t = text(100_000);
        let enc = round_trip(&t);
        assert_eq!(enc[0], MODE_HUFFMAN as u8);
        assert!(enc.len() < t.len() * 5 / 8, "{} bytes", enc.len());
    }

    #[test]
    fn corrupt_streams_are_rejected() {
        let t = text(5000);
        let mut enc = Vec::new();
        encode_stream(&t, &mut enc);
        let decode = |bytes: &[u8]| decode_stream(bytes, 0, t.len(), &mut Vec::new());
        let mut bad = enc.clone();
        bad[0] = 7;
        assert_eq!(decode(&bad), Err(StreamError::BadMode));
        assert_eq!(decode(&enc[..60]), Err(StreamError::Truncated));
        // A length of 12, then an incomplete code (one length dropped).
        let mut bad = enc.clone();
        let sym = t[0] as usize;
        bad[4 + sym / 2] |= 0xC << (4 * (sym % 2));
        assert_eq!(decode(&bad), Err(StreamError::BadTable));
        let mut bad = enc.clone();
        bad[4 + sym / 2] &= !(0xF << (4 * (sym % 2)));
        assert_eq!(decode(&bad), Err(StreamError::BadTable));
        // Lane 0 claims one word fewer than it uses.
        let mut bad = enc.clone();
        let at = 4 + 128;
        let words = u16::from_le_bytes([bad[at], bad[at + 1]]);
        bad[at..at + 2].copy_from_slice(&(words - 1).to_le_bytes());
        assert_eq!(decode(&bad), Err(StreamError::LaneOverrun));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn any_stream_round_trips(
            symbols in prop::collection::vec(prop_oneof![8 => 0u8..6, 2 => 0u8..40, 1 => any::<u8>()], 0..40_000),
        ) {
            round_trip(&symbols);
        }

        #[test]
        fn any_counts_give_complete_limited_codes(
            c in prop::collection::vec(0u32..100_000, 256),
        ) {
            let mut counts = [0u32; 256];
            counts.copy_from_slice(&c);
            prop_assume!(counts.iter().filter(|&&x| x > 0).count() >= 2);
            let l = code_lengths(&counts);
            prop_assert!(l.iter().all(|&x| u32::from(x) <= MAX_LEN));
            prop_assert_eq!(kraft(&l), 1 << MAX_LEN);
            for s in 0..256 {
                prop_assert_eq!(l[s] > 0, counts[s] > 0);
            }
        }
    }
}
