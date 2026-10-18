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
    // Four histograms, summed: runs of one symbol don't wait on one counter.
    let mut partial = [[0u32; 256]; 4];
    let (quads, rest) = symbols.as_chunks::<4>();
    for q in quads {
        partial[0][q[0] as usize] += 1;
        partial[1][q[1] as usize] += 1;
        partial[2][q[2] as usize] += 1;
        partial[3][q[3] as usize] += 1;
    }
    for &s in rest {
        partial[0][s as usize] += 1;
    }
    let counts: [u32; 256] =
        std::array::from_fn(|s| partial[0][s] + partial[1][s] + partial[2][s] + partial[3][s]);
    let used = counts.iter().filter(|&&c| c > 0).count();
    if used == 1 {
        out.extend_from_slice(&(MODE_RLE | u32::from(symbols[0]) << 8).to_le_bytes());
        return;
    }
    let raw_size = 4 + n.next_multiple_of(4);
    if used >= 2 {
        // Written speculatively in one pass (lanes, then the header in front
        // of them); kept only if smaller than raw.
        let lengths = code_lengths(&counts);
        let codes = codes(&lengths);
        let entries: [u32; 256] =
            std::array::from_fn(|s| u32::from(codes[s]) | u32::from(lengths[s]) << 16);
        let lanes = lanes(n);
        let start = out.len();
        let data_at = start + 4 + 128 + (2 * lanes).next_multiple_of(4);
        // At most MAX_LEN bits per symbol, plus a partial word per lane.
        out.resize(
            data_at + 4 * ((n * MAX_LEN as usize).div_ceil(32) + lanes),
            0,
        );
        let mut at = data_at;
        for k in 0..lanes {
            let lane_start = at;
            let (mut acc, mut fill) = (0u64, 0u32);
            // Two symbols per word check: after a flush fewer than 32 bits
            // are pending, and two codes add at most 2 * MAX_LEN = 22.
            let lane = &symbols[lane_range(n, lanes, k)];
            let (pairs, last) = lane.as_chunks::<2>();
            for pair in pairs.iter().map(|p| &p[..]).chain(std::iter::once(last)) {
                for &s in pair {
                    let e = entries[s as usize];
                    acc |= u64::from(e & 0xFFFF) << fill;
                    fill += e >> 16;
                }
                if fill >= 32 {
                    out[at..at + 4].copy_from_slice(&(acc as u32).to_le_bytes());
                    at += 4;
                    acc >>= 32;
                    fill -= 32;
                }
            }
            if fill > 0 {
                out[at..at + 4].copy_from_slice(&(acc as u32).to_le_bytes());
                at += 4;
            }
            let words = (at - lane_start) / 4;
            out[start + 132 + 2 * k..start + 134 + 2 * k]
                .copy_from_slice(&(words as u16).to_le_bytes());
        }
        if at - start < raw_size {
            out[start..start + 4].copy_from_slice(&MODE_HUFFMAN.to_le_bytes());
            for (i, p) in lengths.chunks(2).enumerate() {
                out[start + 4 + i] = p[0] | p[1] << 4;
            }
            out.truncate(at);
            return;
        }
        out.truncate(start);
    }
    out.extend_from_slice(&MODE_RAW.to_le_bytes());
    out.extend_from_slice(symbols);
    out.resize(out.len() + raw_size - 4 - n, 0);
}

/// Decodes the `n`-symbol stream starting at `src[at..]` onto `out`, and
/// returns the position after it. See [`decode_stream_into`].
pub fn decode_stream(
    src: &[u8],
    at: usize,
    n: usize,
    out: &mut Vec<u8>,
) -> Result<usize, StreamError> {
    let start = out.len();
    out.resize(start + n, 0);
    decode_stream_into(src, at, &mut out[start..])
}

/// Decodes the stream starting at `src[at..]` into `out` (as many symbols as
/// it holds), and returns the position after it. Checks, in order: the mode
/// word, the code table (present, then valid), the lane sizes and words
/// (present), then each lane in order (no bits past its words).
pub fn decode_stream_into(src: &[u8], at: usize, out: &mut [u8]) -> Result<usize, StreamError> {
    let n = out.len();
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
            out.copy_from_slice(bytes);
            Ok(end)
        }
        MODE_RLE if head >> 16 == 0 => {
            out.fill((head >> 8) as u8);
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
            let data_at = sizes_at + (2 * lanes).next_multiple_of(4);
            let end = data_at
                + 4 * sizes
                    .chunks(2)
                    .map(|p| usize::from(u16::from_le_bytes([p[0], p[1]])))
                    .sum::<usize>();
            if end > src.len() {
                return Err(StreamError::Truncated);
            }
            // The code is complete (Kraft sum exactly 1), so every entry of
            // the decode table gets a symbol.
            let mut lookup = [0u16; 1 << MAX_LEN];
            build_lookup(&lengths, &mut lookup);
            // Lanes are independent: decode them four at a time, interleaved,
            // so the CPU overlaps their dependent lookup chains.
            let mut lane_at = data_at;
            let mut rest = out;
            let mut group: Vec<(Lane, &mut [u8])> = Vec::with_capacity(4);
            for (k, size) in sizes.chunks(2).enumerate() {
                let words = usize::from(u16::from_le_bytes([size[0], size[1]]));
                let (dst, tail) =
                    std::mem::take(&mut rest).split_at_mut(lane_range(n, lanes, k).len());
                rest = tail;
                group.push((Lane::new(src, lane_at, 4 * words), dst));
                lane_at += 4 * words;
                if group.len() == 4 || k + 1 == lanes {
                    decode_group(&mut group, &lookup)?;
                    group.clear();
                }
            }
            Ok(end)
        }
        _ => Err(StreamError::BadMode),
    }
}

/// Fills `lookup`, the decode table of a complete code with these `lengths`
/// (each 1..=`MAX_LEN`, or 0 for unused): entry = symbol | length << 8,
/// indexed by the next `MAX_LEN` bits, least significant first.
///
/// One entry per codeword, in canonical order (by length, then symbol), into
/// a table that doubles as lengths grow: each doubling copies the shorter
/// codes' entries into the new half. Codewords are kept bit-reversed, so the
/// next one in canonical order sets the highest clear bit and clears those
/// above it. Writing every index of every code separately took ~2.3 µs per
/// table, a quarter of GLZ-E's entropy decoding.
fn build_lookup(lengths: &[u8; 256], lookup: &mut [u16; 1 << MAX_LEN]) {
    const BITS: usize = MAX_LEN as usize;
    let mut count = [0usize; BITS + 1];
    for &l in lengths {
        count[usize::from(l)] += 1;
    }
    count[0] = 0;
    let mut next = [0usize; BITS + 1];
    for l in 1..BITS {
        next[l + 1] = next[l] + count[l];
    }
    let mut sorted = [0u8; 256];
    for (s, &l) in lengths.iter().enumerate() {
        if l > 0 {
            sorted[next[usize::from(l)]] = s as u8;
            next[usize::from(l)] += 1;
        }
    }
    let Some(mut len) = (1..=BITS).find(|&l| count[l] > 0) else {
        return;
    };
    let mut end = 1usize << len;
    let mut codeword = 0usize;
    let mut symbols = sorted.iter();
    loop {
        for _ in 0..count[len] {
            let &s = symbols.next().expect("a symbol per code");
            lookup[codeword] = u16::from(s) | (len as u16) << 8;
            if codeword == end - 1 {
                // The last codeword (all ones): fill the rest of the table.
                while end < 1 << BITS {
                    lookup.copy_within(0..end, end);
                    end <<= 1;
                }
                return;
            }
            let bit = 1 << (usize::BITS - 1 - (codeword ^ (end - 1)).leading_zeros());
            codeword = (codeword & (bit - 1)) | bit;
        }
        // Longer codes: double the table until the next length in use.
        loop {
            len += 1;
            if len > BITS {
                return;
            }
            lookup.copy_within(0..end, end);
            end <<= 1;
            if count[len] > 0 {
                break;
            }
        }
    }
}

/// One lane's bit reader: its words, LSB first, through a 64-bit buffer
/// refilled whole bytes at a time. A refill may read past the lane into what
/// follows it in `src` (zeros past `src`'s end): those bits are used only if a
/// symbol runs past the lane, which is an overrun either way. Bits only
/// accumulate, so checking once at the end ([`Lane::finish`]) catches exactly
/// what a check per symbol would.
struct Lane<'a> {
    src: &'a [u8],
    /// The lane's first byte, and the byte after it, in `src`.
    start: usize,
    end: usize,
    /// The next byte to load, and the buffer: `avail` bits, LSB first, plus
    /// copies of the bytes after them above (refills OR the same bits in).
    next: usize,
    bits: u64,
    avail: u32,
}

/// Symbols a lane decodes per refill: a refill leaves at least 56 bits, and
/// a code has at most `MAX_LEN` = 11.
const PER_REFILL: usize = 5;

impl<'a> Lane<'a> {
    fn new(src: &'a [u8], start: usize, len: usize) -> Self {
        Lane {
            src,
            start,
            end: start + len,
            next: start,
            bits: 0,
            avail: 0,
        }
    }

    /// Loads whole bytes until at least 56 bits are buffered.
    #[inline(always)]
    fn refill(&mut self) {
        let chunk = match self.src.get(self.next..self.next + 8) {
            Some(b) => u64::from_le_bytes(b.try_into().unwrap()),
            None => {
                let mut b = [0u8; 8];
                let tail = self.src.get(self.next..).unwrap_or(&[]);
                let n = tail.len().min(8);
                b[..n].copy_from_slice(&tail[..n]);
                u64::from_le_bytes(b)
            }
        };
        self.bits |= chunk << self.avail;
        let take = (63 - self.avail) >> 3;
        self.next += take as usize;
        self.avail += 8 * take;
    }

    /// One symbol, with at least `MAX_LEN` bits buffered.
    #[inline(always)]
    fn step(&mut self, lookup: &[u16; 1 << MAX_LEN]) -> u8 {
        let entry = lookup[(self.bits & ((1 << MAX_LEN) - 1)) as usize];
        let len = u32::from(entry >> 8);
        self.bits >>= len;
        self.avail -= len;
        entry as u8
    }

    #[inline(always)]
    fn decode(&mut self, lookup: &[u16; 1 << MAX_LEN]) -> u8 {
        if self.avail < MAX_LEN {
            self.refill();
        }
        self.step(lookup)
    }

    fn finish(&self) -> Result<(), StreamError> {
        // Bits consumed: every byte loaded, less what's still buffered.
        let used = 8 * (self.next - self.start) - self.avail as usize;
        if used > 8 * (self.end - self.start) {
            return Err(StreamError::LaneOverrun);
        }
        Ok(())
    }
}

/// Decodes up to four lanes into their outputs: interleaved while all have
/// symbols left (their lengths differ by at most one), [`PER_REFILL`]
/// symbols per lane between refills, then each one's rest.
fn decode_group(
    group: &mut [(Lane, &mut [u8])],
    lookup: &[u16; 1 << MAX_LEN],
) -> Result<(), StreamError> {
    let common = group.iter().map(|(_, out)| out.len()).min().unwrap_or(0);
    let mut done = 0;
    if let [(a, oa), (b, ob), (c, oc), (d, od)] = group {
        let rounds = common / PER_REFILL;
        let (oa, ob, oc, od) = (
            &mut oa[..rounds * PER_REFILL],
            &mut ob[..rounds * PER_REFILL],
            &mut oc[..rounds * PER_REFILL],
            &mut od[..rounds * PER_REFILL],
        );
        for (((ra, rb), rc), rd) in oa
            .as_chunks_mut::<PER_REFILL>()
            .0
            .iter_mut()
            .zip(ob.as_chunks_mut::<PER_REFILL>().0.iter_mut())
            .zip(oc.as_chunks_mut::<PER_REFILL>().0.iter_mut())
            .zip(od.as_chunks_mut::<PER_REFILL>().0.iter_mut())
        {
            a.refill();
            b.refill();
            c.refill();
            d.refill();
            for i in 0..PER_REFILL {
                ra[i] = a.step(lookup);
                rb[i] = b.step(lookup);
                rc[i] = c.step(lookup);
                rd[i] = d.step(lookup);
            }
        }
        done = rounds * PER_REFILL;
    }
    for (lane, out) in group.iter_mut() {
        for symbol in &mut out[done..] {
            *symbol = lane.decode(lookup);
        }
    }
    group.iter().try_for_each(|(lane, _)| lane.finish())
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

    /// The decode table the plain way: every index whose low `len` bits are
    /// a symbol's (bit-reversed) code gets that symbol.
    fn reference_lookup(lengths: &[u8; 256]) -> [u16; 1 << MAX_LEN] {
        let codes = codes(lengths);
        let mut lookup = [0u16; 1 << MAX_LEN];
        for s in 0..256 {
            let len = u32::from(lengths[s]);
            if len > 0 {
                let mut index = u32::from(codes[s]);
                while index < 1 << MAX_LEN {
                    lookup[index as usize] = s as u16 | (len as u16) << 8;
                    index += 1 << len;
                }
            }
        }
        lookup
    }

    #[test]
    fn decode_tables_cover_every_index_of_complete_codes() {
        // Two symbols, one bit each; and the longest codes the format allows.
        let mut two = [0u8; 256];
        (two[3], two[200]) = (1, 1);
        let mut deep = [0u8; 256];
        for (s, l) in deep.iter_mut().enumerate().take(12) {
            *l = (s + 1).min(11) as u8;
        }
        for lengths in [two, deep] {
            let mut lookup = [0u16; 1 << MAX_LEN];
            build_lookup(&lengths, &mut lookup);
            assert_eq!(lookup, reference_lookup(&lengths));
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn decode_tables_equal_the_plain_construction(
            c in prop::collection::vec(0u32..100_000, 256),
            used in 2usize..256,
        ) {
            let mut counts = [0u32; 256];
            for (i, &n) in c.iter().take(used).enumerate() {
                counts[(i * 97) % 256] = n.max(1);
            }
            let lengths = code_lengths(&counts);
            let mut lookup = [0xEEEEu16; 1 << MAX_LEN];
            build_lookup(&lengths, &mut lookup);
            prop_assert_eq!(lookup, reference_lookup(&lengths));
        }

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
