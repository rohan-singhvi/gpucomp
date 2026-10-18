//! Reversible per-chunk filters (plan §4a): byte-shuffle, delta and their
//! combinations over `width`-byte elements (`crates/format/FORMAT.md`). With `n` bytes there are `n / width` whole elements;
//! the trailing `n % width` bytes always pass through unchanged.

use format::Filter;

const LEVEL1: [Filter; 6] = [
    Filter::None,
    Filter::Shuffle { width: 4 },
    Filter::Delta { width: 4 },
    Filter::DeltaShuffle { width: 4 },
    Filter::DeltaZigzagShuffle { width: 2 },
    Filter::DeltaBitShuffle { width: 8 },
];
const LEVEL2: [Filter; 13] = [
    Filter::None,
    Filter::Shuffle { width: 2 },
    Filter::Shuffle { width: 4 },
    Filter::Shuffle { width: 8 },
    Filter::Delta { width: 2 },
    Filter::Delta { width: 4 },
    Filter::Delta { width: 8 },
    Filter::DeltaShuffle { width: 2 },
    Filter::DeltaShuffle { width: 4 },
    Filter::DeltaShuffle { width: 8 },
    Filter::DeltaZigzagShuffle { width: 2 },
    Filter::DeltaBitShuffle { width: 4 },
    Filter::DeltaBitShuffle { width: 8 },
];

/// Candidate filters tried per chunk at `level`, in tie-break order (none <
/// shuffle < delta < delta-shuffle < delta-zigzag-shuffle < delta-bitshuffle,
/// then smaller width first). Levels 0 and 1 try width 4 of the first four,
/// delta-zigzag-shuffle-2 (16-bit samples) and delta-bitshuffle-8 (64-bit
/// values, pairs of floats); level 2 and above try widths 2, 4 and 8 of the
/// first four, delta-zigzag-shuffle-2 and delta-bitshuffle-4 and -8. The
/// choice comes from measurements (DECISIONS.md, "Float filters").
pub fn candidates(level: u8) -> &'static [Filter] {
    if level <= 1 {
        &LEVEL1
    } else {
        &LEVEL2
    }
}

/// Bytes of each chunk's leading sample used by `FilterMode::Auto`: an
/// eighth of the chunk, at least 4 KiB (so chunks up to 4 KiB are sampled
/// whole). Always a valid chunk size, so a GPU can encode samples as chunks.
pub fn sample_len(chunk_size: u32) -> u32 {
    (chunk_size / 8).max(format::MIN_CHUNK_SIZE)
}

/// Reads a little-endian `w`-byte element as a u64.
fn load(bytes: &[u8]) -> u64 {
    let mut b = [0u8; 8];
    b[..bytes.len()].copy_from_slice(bytes);
    u64::from_le_bytes(b)
}

fn store(dst: &mut [u8], value: u64) {
    let w = dst.len();
    dst.copy_from_slice(&value.to_le_bytes()[..w]);
}

/// Applies `filter` to `src`, writing `dst` (same length).
pub fn forward_into(filter: Filter, src: &[u8], dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len());
    match filter {
        Filter::None => dst.copy_from_slice(src),
        Filter::Shuffle { width } => {
            let w = usize::from(width);
            let m = src.len() / w;
            for (i, elem) in src[..m * w].chunks_exact(w).enumerate() {
                for (j, &b) in elem.iter().enumerate() {
                    dst[j * m + i] = b;
                }
            }
            dst[m * w..].copy_from_slice(&src[m * w..]);
        }
        Filter::Delta { width } => {
            let w = usize::from(width);
            let whole = src.len() / w * w;
            let mut prev = 0u64;
            for (s, d) in src[..whole].chunks_exact(w).zip(dst.chunks_exact_mut(w)) {
                let v = load(s);
                store(d, v.wrapping_sub(prev));
                prev = v;
            }
            dst[whole..].copy_from_slice(&src[whole..]);
        }
        Filter::DeltaShuffle { width } => delta_shuffle(width, src, dst, |d| d),
        Filter::DeltaZigzagShuffle { width } => {
            delta_shuffle(width, src, dst, |d| zigzag(d, width))
        }
        Filter::DeltaBitShuffle { width } => {
            let w = usize::from(width);
            let m = src.len() / w;
            let m32 = bit_planes(m);
            let plane = m32 / 8;
            let mut prev = 0u64;
            // Eight elements at a time: byte j of their deltas, as an 8×8
            // bit matrix, transposed into bit planes 8j..8j + 8.
            for (g, group) in src[..m32 * w].chunks_exact(8 * w).enumerate() {
                let mut rows = [0u64; 8];
                for (r, elem) in group.chunks_exact(w).enumerate() {
                    let v = load(elem);
                    let delta = v.wrapping_sub(prev);
                    prev = v;
                    for (j, row) in rows[..w].iter_mut().enumerate() {
                        *row |= ((delta >> (8 * j)) & 0xFF) << (8 * r);
                    }
                }
                for (j, &row) in rows[..w].iter().enumerate() {
                    for (b, byte) in transpose8(row).to_le_bytes().into_iter().enumerate() {
                        dst[(8 * j + b) * plane + g] = byte;
                    }
                }
            }
            for (s, d) in src[m32 * w..m * w]
                .chunks_exact(w)
                .zip(dst[m32 * w..].chunks_exact_mut(w))
            {
                let v = load(s);
                store(d, v.wrapping_sub(prev));
                prev = v;
            }
            dst[m * w..].copy_from_slice(&src[m * w..]);
        }
    }
}

/// Delta then shuffle in one pass, with `map` applied to each delta: byte j
/// of element i's mapped delta goes to j * m + i.
fn delta_shuffle(width: u8, src: &[u8], dst: &mut [u8], map: impl Fn(u64) -> u64) {
    let w = usize::from(width);
    let m = src.len() / w;
    let mut prev = 0u64;
    for (i, elem) in src[..m * w].chunks_exact(w).enumerate() {
        let v = load(elem);
        let delta = map(v.wrapping_sub(prev)).to_le_bytes();
        prev = v;
        for (j, &b) in delta[..w].iter().enumerate() {
            dst[j * m + i] = b;
        }
    }
    dst[m * w..].copy_from_slice(&src[m * w..]);
}

/// The inverse of [`delta_shuffle`] (`unmap` undoes `map`): gather element
/// i's delta, then the running sum.
fn delta_unshuffle(width: u8, src: &[u8], dst: &mut [u8], unmap: impl Fn(u64) -> u64) {
    let w = usize::from(width);
    let m = src.len() / w;
    let mut sum = 0u64;
    for (i, elem) in dst[..m * w].chunks_exact_mut(w).enumerate() {
        let mut delta = [0u8; 8];
        for (j, b) in delta[..w].iter_mut().enumerate() {
            *b = src[j * m + i];
        }
        sum = sum.wrapping_add(unmap(u64::from_le_bytes(delta)));
        elem.copy_from_slice(&sum.to_le_bytes()[..w]);
    }
    dst[m * w..].copy_from_slice(&src[m * w..]);
}

/// Zigzag of a `width`-byte two's-complement value (its low bytes): 0, -1,
/// 1, -2, ... become 0, 1, 2, 3, ...
fn zigzag(d: u64, width: u8) -> u64 {
    let shift = 64 - 8 * u32::from(width);
    let signed = ((d << shift) as i64) >> shift;
    ((signed << 1) ^ (signed >> 63)) as u64
}

fn unzigzag(z: u64) -> u64 {
    (z >> 1) ^ (z & 1).wrapping_neg()
}

/// Elements that delta-bitshuffle spreads over bit planes: whole groups of
/// 32, so every plane is a whole number of 32-bit words.
fn bit_planes(m: usize) -> usize {
    m / 32 * 32
}

/// Transposes an 8×8 bit matrix: bit `8 * r + c` and bit `8 * c + r` swap
/// (Hacker's Delight, 7-3). Its own inverse.
fn transpose8(mut x: u64) -> u64 {
    let mut t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
    x ^= t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
    x ^= t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
    x ^ t ^ (t << 28)
}

/// Undoes `filter`: `inverse(f, &forward(f, x)) == x`.
pub fn inverse_into(filter: Filter, src: &[u8], dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len());
    match filter {
        Filter::None => dst.copy_from_slice(src),
        Filter::Shuffle { width } => {
            let w = usize::from(width);
            let m = src.len() / w;
            for (i, elem) in dst[..m * w].chunks_exact_mut(w).enumerate() {
                for (j, b) in elem.iter_mut().enumerate() {
                    *b = src[j * m + i];
                }
            }
            dst[m * w..].copy_from_slice(&src[m * w..]);
        }
        Filter::Delta { .. } => {
            dst.copy_from_slice(src);
            inverse_in_place(filter, dst);
        }
        Filter::DeltaShuffle { width } => delta_unshuffle(width, src, dst, |d| d),
        Filter::DeltaZigzagShuffle { width } => delta_unshuffle(width, src, dst, unzigzag),
        Filter::DeltaBitShuffle { width } => {
            let w = usize::from(width);
            let m = src.len() / w;
            let m32 = bit_planes(m);
            let plane = m32 / 8;
            let mut sum = 0u64;
            for (g, group) in dst[..m32 * w].chunks_exact_mut(8 * w).enumerate() {
                let mut rows = [0u64; 8];
                for (j, row) in rows[..w].iter_mut().enumerate() {
                    let mut bits = [0u8; 8];
                    for (b, byte) in bits.iter_mut().enumerate() {
                        *byte = src[(8 * j + b) * plane + g];
                    }
                    *row = transpose8(u64::from_le_bytes(bits));
                }
                for (r, elem) in group.chunks_exact_mut(w).enumerate() {
                    let mut delta = 0u64;
                    for (j, &row) in rows[..w].iter().enumerate() {
                        delta |= ((row >> (8 * r)) & 0xFF) << (8 * j);
                    }
                    sum = sum.wrapping_add(delta);
                    store(elem, sum);
                }
            }
            for (s, d) in src[m32 * w..m * w]
                .chunks_exact(w)
                .zip(dst[m32 * w..].chunks_exact_mut(w))
            {
                sum = sum.wrapping_add(load(s));
                store(d, sum);
            }
            dst[m * w..].copy_from_slice(&src[m * w..]);
        }
    }
}

/// [`forward_into`] into a new vector.
pub fn forward(filter: Filter, src: &[u8]) -> Vec<u8> {
    let mut dst = vec![0; src.len()];
    forward_into(filter, src, &mut dst);
    dst
}

/// [`inverse_into`] into a new vector.
pub fn inverse(filter: Filter, src: &[u8]) -> Vec<u8> {
    let mut dst = vec![0; src.len()];
    inverse_into(filter, src, &mut dst);
    dst
}

/// Undoes `filter` in place (shuffle goes through a temporary copy).
pub fn inverse_in_place(filter: Filter, buf: &mut [u8]) {
    match filter {
        Filter::None => {}
        Filter::Shuffle { .. } => {
            let src = buf.to_vec();
            inverse_into(filter, &src, buf);
        }
        Filter::DeltaShuffle { .. }
        | Filter::DeltaZigzagShuffle { .. }
        | Filter::DeltaBitShuffle { .. } => {
            let src = buf.to_vec();
            inverse_into(filter, &src, buf);
        }
        Filter::Delta { width } => {
            let w = usize::from(width);
            let whole = buf.len() / w * w;
            let mut sum = 0u64;
            for elem in buf[..whole].chunks_exact_mut(w) {
                sum = sum.wrapping_add(load(elem));
                store(elem, sum);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const WIDTHS: [u8; 4] = [1, 2, 4, 8];

    fn all_filters() -> Vec<Filter> {
        let mut out = vec![Filter::None];
        out.extend(WIDTHS.map(|width| Filter::Shuffle { width }));
        out.extend(WIDTHS.map(|width| Filter::Delta { width }));
        out.extend(WIDTHS.map(|width| Filter::DeltaShuffle { width }));
        out.extend(WIDTHS.map(|width| Filter::DeltaZigzagShuffle { width }));
        out.extend(WIDTHS.map(|width| Filter::DeltaBitShuffle { width }));
        out
    }

    #[test]
    fn delta_zigzag_shuffle_zigzags_each_delta_before_shuffling() {
        // u16 elements 0x0102, 0x0101, 0x0300 and a 1-byte tail: deltas 258,
        // -1, 511; zigzag 516 (0x0204), 1, 1022 (0x03FE); low bytes first.
        let src = [0x02, 0x01, 0x01, 0x01, 0x00, 0x03, 0xEE];
        let out = forward(Filter::DeltaZigzagShuffle { width: 2 }, &src);
        assert_eq!(out, [0x04, 0x01, 0xFE, 0x02, 0x00, 0x03, 0xEE]);
        assert_eq!(inverse(Filter::DeltaZigzagShuffle { width: 2 }, &out), src);
        // Bytes 5, 7, 6: deltas 5, 2, -1 become 10, 4, 1.
        assert_eq!(
            forward(Filter::DeltaZigzagShuffle { width: 1 }, &[5, 7, 6]),
            [10, 4, 1]
        );
        // u64 deltas 5, -2 and 2^63 (the most negative) become 10, 3, 2^64 - 1.
        let src = [5u64, 3, 3 + (1 << 63)].map(u64::to_le_bytes).concat();
        let zigzagged = [10u64, 3, u64::MAX].map(u64::to_le_bytes).concat();
        assert_eq!(
            forward(Filter::DeltaZigzagShuffle { width: 8 }, &src),
            forward(Filter::Shuffle { width: 8 }, &zigzagged)
        );
    }

    /// Delta-bitshuffle by its definition: bit `k` of delta `i < m32` (the
    /// first `m / 32 * 32` elements) goes to bit `i % 8` of byte
    /// `k * m32 / 8 + i / 8`; later deltas and the tail stay in place.
    fn bitshuffle_reference(width: u8, src: &[u8]) -> Vec<u8> {
        let w = usize::from(width);
        let deltas = forward(Filter::Delta { width }, src);
        let m32 = src.len() / w / 32 * 32;
        let mut out = deltas.clone();
        out[..m32 * w].fill(0);
        for i in 0..m32 {
            for k in 0..8 * w {
                let bit = (deltas[i * w + k / 8] >> (k % 8)) & 1;
                out[k * m32 / 8 + i / 8] |= bit << (i % 8);
            }
        }
        out
    }

    #[test]
    fn delta_bitshuffle_puts_bit_k_of_every_delta_in_plane_k() {
        // Bytes 1..=33: 32 deltas of 1 fill plane 0 with ones; element 32 (a
        // 33rd) isn't in a whole group of 32 and keeps its delta.
        let src: Vec<u8> = (1..=33).collect();
        let out = forward(Filter::DeltaBitShuffle { width: 1 }, &src);
        let mut expected = vec![0u8; 33];
        expected[..4].fill(0xFF);
        expected[32] = 1;
        assert_eq!(out, expected);
        let data: Vec<u8> = (0..3001u32).map(|i| (i * i / 7 + i / 3) as u8).collect();
        for width in WIDTHS {
            for len in [
                0,
                7,
                31 * usize::from(width),
                32 * usize::from(width) + 3,
                1000,
                3001,
            ] {
                assert_eq!(
                    forward(Filter::DeltaBitShuffle { width }, &data[..len]),
                    bitshuffle_reference(width, &data[..len]),
                    "width {width} len {len}"
                );
            }
        }
    }

    #[test]
    fn delta_shuffle_is_delta_then_shuffle() {
        // u16 elements 0x0102, 0x0101, 0x0300 and a 1-byte tail: deltas 0x0102,
        // 0xFFFF, 0x01FF, then low bytes before high bytes.
        let src = [0x02, 0x01, 0x01, 0x01, 0x00, 0x03, 0xEE];
        let out = forward(Filter::DeltaShuffle { width: 2 }, &src);
        assert_eq!(out, [0x02, 0xFF, 0xFF, 0x01, 0xFF, 0x01, 0xEE]);
        assert_eq!(inverse(Filter::DeltaShuffle { width: 2 }, &out), src);
        let data: Vec<u8> = (0..1003u32).map(|i| (i * i / 7) as u8).collect();
        for width in WIDTHS {
            let delta = forward(Filter::Delta { width }, &data);
            assert_eq!(
                forward(Filter::DeltaShuffle { width }, &data),
                forward(Filter::Shuffle { width }, &delta),
                "{width}"
            );
        }
    }

    #[test]
    fn shuffle_transposes_elements_and_keeps_the_tail() {
        // Three 2-byte elements (a0 a1)(b0 b1)(c0 c1), then a 1-byte tail.
        let src = [0xA0, 0xA1, 0xB0, 0xB1, 0xC0, 0xC1, 0x77];
        let out = forward(Filter::Shuffle { width: 2 }, &src);
        assert_eq!(out, [0xA0, 0xB0, 0xC0, 0xA1, 0xB1, 0xC1, 0x77]);
        assert_eq!(inverse(Filter::Shuffle { width: 2 }, &out), src);
    }

    #[test]
    fn shuffle_4_groups_byte_planes() {
        let src: Vec<u8> = (0..10).collect(); // two elements + 2-byte tail
        assert_eq!(
            forward(Filter::Shuffle { width: 4 }, &src),
            [0, 4, 1, 5, 2, 6, 3, 7, 8, 9]
        );
    }

    #[test]
    fn delta_takes_wrapping_little_endian_differences() {
        // u16 elements 0x0102, 0x0101, 0x0300; tail 0xEE.
        let src = [0x02, 0x01, 0x01, 0x01, 0x00, 0x03, 0xEE];
        let out = forward(Filter::Delta { width: 2 }, &src);
        // 0x0102 unchanged; 0x0101 - 0x0102 = 0xFFFF; 0x0300 - 0x0101 = 0x01FF.
        assert_eq!(out, [0x02, 0x01, 0xFF, 0xFF, 0xFF, 0x01, 0xEE]);
        assert_eq!(inverse(Filter::Delta { width: 2 }, &out), src);
    }

    #[test]
    fn delta_1_and_8_on_small_examples() {
        assert_eq!(
            forward(Filter::Delta { width: 1 }, &[5, 7, 6, 255, 0]),
            [5, 2, 255, 249, 1]
        );
        let a = 0x0000_0001_FFFF_FFFFu64;
        let b = 0x0000_0002_0000_0001u64;
        let src = [a.to_le_bytes(), b.to_le_bytes()].concat();
        let out = forward(Filter::Delta { width: 8 }, &src);
        assert_eq!(out[..8], a.to_le_bytes());
        assert_eq!(out[8..], 2u64.to_le_bytes());
    }

    #[test]
    fn inputs_shorter_than_one_element_pass_through() {
        for f in all_filters() {
            if !matches!(
                f,
                Filter::Shuffle { width: 1 | 2 }
                    | Filter::Delta { width: 1 | 2 }
                    | Filter::DeltaShuffle { width: 1 | 2 }
                    | Filter::DeltaZigzagShuffle { width: 1 | 2 }
                    | Filter::DeltaBitShuffle { width: 1 | 2 }
            ) {
                assert_eq!(forward(f, &[1, 2, 3]), [1, 2, 3], "{f:?}");
            }
            assert_eq!(forward(f, &[]), Vec::<u8>::new(), "{f:?}");
        }
    }

    #[test]
    fn none_is_the_identity() {
        assert_eq!(forward(Filter::None, &[9, 8, 7]), [9, 8, 7]);
    }

    #[test]
    fn level_1_tries_width_4_and_higher_levels_add_widths_2_and_8() {
        use Filter::*;
        assert_eq!(
            candidates(1),
            [
                None,
                Shuffle { width: 4 },
                Delta { width: 4 },
                DeltaShuffle { width: 4 },
                DeltaZigzagShuffle { width: 2 },
                DeltaBitShuffle { width: 8 },
            ]
        );
        assert_eq!(
            candidates(2),
            [
                None,
                Shuffle { width: 2 },
                Shuffle { width: 4 },
                Shuffle { width: 8 },
                Delta { width: 2 },
                Delta { width: 4 },
                Delta { width: 8 },
                DeltaShuffle { width: 2 },
                DeltaShuffle { width: 4 },
                DeltaShuffle { width: 8 },
                DeltaZigzagShuffle { width: 2 },
                DeltaBitShuffle { width: 4 },
                DeltaBitShuffle { width: 8 },
            ]
        );
        assert_eq!(candidates(9), candidates(2));
        assert_eq!(candidates(0), candidates(1));
    }

    proptest! {
        #[test]
        fn inverse_undoes_forward_for_every_filter(
            data in proptest::collection::vec(any::<u8>(), 0..300),
        ) {
            for f in all_filters() {
                let filtered = forward(f, &data);
                prop_assert_eq!(&inverse(f, &filtered), &data, "{:?}", f);
                let mut in_place = filtered.clone();
                inverse_in_place(f, &mut in_place);
                prop_assert_eq!(&in_place, &data, "{:?} in place", f);
                // The tail is untouched.
                if let Filter::Shuffle { width }
                | Filter::Delta { width }
                | Filter::DeltaShuffle { width }
                | Filter::DeltaZigzagShuffle { width }
                | Filter::DeltaBitShuffle { width } = f
                {
                    let whole = data.len() / usize::from(width) * usize::from(width);
                    prop_assert_eq!(&filtered[whole..], &data[whole..]);
                }
            }
        }
    }
}
