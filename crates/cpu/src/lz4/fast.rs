//! A fast path for the GPU twin's match finding and parse: the same sequences
//! as `parse(input, &find_matches(input, params), params.lazy)`, several times
//! faster. Every position is still inserted into the hash table block by block
//! (the GPU's semantics: a position sees only earlier blocks, latest wins), but
//! matches are computed only where the parse looks (about 40% of positions),
//! comparing 8 bytes at a time.

use super::encode::{hash, Match, Params, Sequence};
use super::{LAST_LITERALS, MAX_OFFSET, MFLIMIT, MIN_MATCH};

fn read_u32(input: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(input[p..p + 4].try_into().unwrap())
}

fn read_u64(input: &[u8], p: usize) -> u64 {
    u64::from_le_bytes(input[p..p + 8].try_into().unwrap())
}

/// How many leading bytes `input[q..]` and `input[p..]` share, up to `max`
/// (`p + max` within the input).
fn common(input: &[u8], q: usize, p: usize, max: usize) -> usize {
    let mut i = 0;
    // 8 bytes at a time, the last step clamped to `max`; bytes only where
    // fewer than 8 remain in the input (q < p, so p's side ends first).
    while i < max {
        if p + i + 8 > input.len() {
            while i < max && input[q + i] == input[p + i] {
                i += 1;
            }
            return i;
        }
        let x = read_u64(input, q + i) ^ read_u64(input, p + i);
        if x != 0 {
            return (i + (x.trailing_zeros() / 8) as usize).min(max);
        }
        i += 8;
    }
    max
}

/// With depth > 1, table and chain entries carry an 11-bit tag of their
/// position's first 4 bytes above the position + 1 (as the GPU's chain links
/// do): a candidate whose tag differs from the position's can't match, so the
/// walk skips it without loading its bytes.
const LINK_BITS: u32 = 21;
const LINK_MASK: u32 = (1 << LINK_BITS) - 1;

fn tag(word: u32) -> u32 {
    word.wrapping_mul(0x85EB_CA77) >> LINK_BITS
}

/// Phase-1 matches on demand, in nondecreasing position order.
struct Matcher<'a> {
    input: &'a [u8],
    params: &'a Params,
    last_start: usize,
    match_limit: usize,
    table: Vec<u32>,
    /// With depth > 1: per position, its first candidate (position + 1).
    chain: Vec<u32>,
    /// Positions below this (whole blocks) are in the table.
    inserted: usize,
    /// The last match computed: the lazy look-ahead's, reused by the next step.
    cached: Option<(usize, Match)>,
}

impl<'a> Matcher<'a> {
    fn new(input: &'a [u8], params: &'a Params) -> Self {
        let n = input.len();
        Matcher {
            input,
            params,
            last_start: n - MFLIMIT,
            match_limit: n - LAST_LITERALS,
            table: vec![0; 1 << params.hash_log],
            chain: vec![0; if params.depth > 1 { n } else { 0 }],
            inserted: 0,
            cached: None,
        }
    }

    /// Inserts every block before the one holding `p`.
    fn insert_before(&mut self, p: usize) {
        let block = self.params.block;
        let hash_log = self.params.hash_log;
        while self.inserted + block <= p {
            let end = (self.inserted + block).min(self.last_start + 1);
            if self.params.depth > 1 {
                // A position's first candidate is the bucket before its own
                // block's inserts (the GPU's chain links); four per load, as
                // below.
                let (input, table) = (self.input, &self.table[..]);
                let chain = &mut self.chain[self.inserted..end];
                let mut k = 0;
                while k + 4 <= chain.len() && self.inserted + k + 8 <= input.len() {
                    let v = read_u64(input, self.inserted + k);
                    for j in 0..4 {
                        chain[k + j] = table[hash((v >> (8 * j)) as u32, hash_log) as usize];
                    }
                    k += 4;
                }
                for (j, link) in chain.iter_mut().enumerate().skip(k) {
                    *link = table[hash(read_u32(input, self.inserted + j), hash_log) as usize];
                }
            }
            // Positions only grow, so the latest wins without a max. Four
            // positions per 8-byte load: position q + k's word is its bytes
            // k..k + 4.
            let input = self.input;
            let table = &mut self.table[..];
            let tagged = self.params.depth > 1;
            let entry = |q: usize, word: u32| {
                let tag = if tagged { tag(word) << LINK_BITS } else { 0 };
                (q as u32 + 1) | tag
            };
            let mut q = self.inserted;
            while q + 4 <= end && q + 8 <= input.len() {
                let v = read_u64(input, q);
                for k in 0..4 {
                    let word = (v >> (8 * k)) as u32;
                    table[hash(word, hash_log) as usize] = entry(q + k, word);
                }
                q += 4;
            }
            for q in q..end {
                let word = read_u32(input, q);
                table[hash(word, hash_log) as usize] = entry(q, word);
            }
            self.inserted += block;
        }
    }

    /// `find_matches(input, params)[p]`.
    fn at(&mut self, p: usize) -> Match {
        if let Some((q, m)) = self.cached {
            if q == p {
                return m;
            }
        }
        let m = self.compute(p);
        self.cached = Some((p, m));
        m
    }

    fn compute(&mut self, p: usize) -> Match {
        if p > self.last_start {
            return Match::default();
        }
        self.insert_before(p);
        let input = self.input;
        let max = (p + self.params.probe_len).min(self.match_limit) - p;
        let word = read_u32(input, p);
        let mut entry = self.table[hash(word, self.params.hash_log) as usize];
        if self.params.depth <= 1 {
            // One candidate: its first word decides most cases (a hash
            // collision, or fewer than MIN_MATCH = 4 equal bytes).
            if entry == 0 {
                return Match::default();
            }
            let candidate = entry as usize - 1;
            let offset = p - candidate;
            if offset > MAX_OFFSET || read_u32(input, candidate) != word {
                return Match::default();
            }
            let len =
                MIN_MATCH + common(input, candidate + MIN_MATCH, p + MIN_MATCH, max - MIN_MATCH);
            return Match {
                len: len as u32,
                offset: offset as u32,
            };
        }
        let mut best = Match::default();
        let ptag = tag(word);
        for _ in 0..self.params.depth.max(1) {
            if entry & LINK_MASK == 0 {
                break;
            }
            let candidate = (entry & LINK_MASK) as usize - 1;
            let offset = p - candidate;
            if offset > MAX_OFFSET {
                break;
            }
            // Only a longer match counts, so byte `best.len` must match too;
            // and a candidate with another prefix can't match at all.
            let b = best.len as usize;
            if entry >> LINK_BITS == ptag && (b == 0 || input[candidate + b] == input[p + b]) {
                let len = common(input, candidate, p, max);
                if len >= MIN_MATCH && len > b {
                    best = Match {
                        len: len as u32,
                        offset: offset as u32,
                    };
                    if len == max {
                        // Capped: nothing can be longer.
                        break;
                    }
                }
            }
            if self.params.depth <= 1 {
                break;
            }
            entry = self.chain[candidate];
        }
        best
    }
}

/// The sequences `parse(input, &find_matches(input, params), params.lazy)`
/// returns.
pub fn sequences(input: &[u8], params: &Params) -> Vec<Sequence> {
    let n = input.len();
    let mut sequences = Vec::new();
    let mut anchor = 0;
    if n >= MFLIMIT {
        let mut matcher = Matcher::new(input, params);
        let match_limit = n - LAST_LITERALS;
        let mut p = 0;
        while p + MFLIMIT <= n {
            let m = matcher.at(p);
            let len = m.len as usize;
            if len < MIN_MATCH
                || (params.lazy && p + 1 + MFLIMIT <= n && matcher.at(p + 1).len > m.len)
            {
                p += 1;
                continue;
            }
            // Extend past the probe cap.
            let offset = m.offset as usize;
            let from = p + len;
            let end = from + common(input, from - offset, from, match_limit - from);
            sequences.push(Sequence {
                lit_start: anchor as u32,
                lit_len: (p - anchor) as u32,
                match_len: (end - p) as u32,
                offset: m.offset,
            });
            p = end;
            anchor = end;
        }
    }
    sequences.push(Sequence {
        lit_start: anchor as u32,
        lit_len: (n - anchor) as u32,
        match_len: 0,
        offset: 0,
    });
    sequences
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::lz4::encode::{find_matches, parse};

    fn reference(input: &[u8], params: &Params) -> Vec<Sequence> {
        parse(input, &find_matches(input, params), params.lazy)
    }

    fn random(n: usize, seed: u64, alphabet: u8) -> Vec<u8> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % u64::from(alphabet)) as u8
            })
            .collect()
    }

    /// Text-like data, runs, incompressible bytes, and repeats both within and
    /// beyond MAX_OFFSET.
    fn fixtures() -> Vec<(&'static str, Vec<u8>)> {
        let text: Vec<u8> = b"the quick brown fox jumps over the lazy dog; "
            .iter()
            .copied()
            .cycle()
            .take(20_000)
            .collect();
        let far = [
            random(30_000, 1, 255),
            random(50_000, 2, 4),
            random(30_000, 1, 255),
        ]
        .concat();
        vec![
            ("empty", vec![]),
            ("short", b"abcabcabcab".to_vec()),
            ("mflimit", random(12, 3, 2)),
            ("text", text),
            ("zeros", vec![0; 70_000]),
            ("random", random(10_000, 4, 255)),
            ("small alphabet", random(40_000, 5, 3)),
            ("repeat beyond max offset", far),
            (
                "mixed",
                [random(5_000, 6, 255), vec![7; 3_000], random(9_000, 7, 6)].concat(),
            ),
        ]
    }

    #[test]
    fn matches_the_reference_on_fixtures_at_every_level() {
        for (name, input) in fixtures() {
            for level in 1..=3 {
                for lazy in [false, true] {
                    let params = Params {
                        lazy,
                        ..Params::for_level(level)
                    };
                    assert_eq!(
                        sequences(&input, &params),
                        reference(&input, &params),
                        "{name}, level {level}, lazy {lazy}"
                    );
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]
        #[test]
        fn matches_the_reference_for_any_params(
            input in prop_oneof![
                proptest::collection::vec(0u8..4, 0..3_000),
                proptest::collection::vec(any::<u8>(), 0..1_000),
                (proptest::collection::vec(any::<u8>(), 1..200), 1usize..40)
                    .prop_map(|(unit, reps)| unit.repeat(reps)),
            ],
            block in prop_oneof![Just(1usize), Just(7), Just(64), Just(128)],
            hash_log in prop_oneof![Just(6u32), Just(12)],
            probe_len in prop_oneof![Just(4usize), Just(7), Just(16), Just(40)],
            lazy in any::<bool>(),
            depth in prop_oneof![Just(1usize), Just(2), Just(4), Just(16)],
        ) {
            let params = Params { block, hash_log, probe_len, lazy, depth };
            prop_assert_eq!(sequences(&input, &params), reference(&input, &params));
        }
    }
}
