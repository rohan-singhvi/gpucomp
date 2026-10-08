//! Reversible per-chunk filters (plan §4a): byte-shuffle and delta over
//! `width`-byte elements. With `n` bytes there are `n / width` whole elements;
//! the trailing `n % width` bytes always pass through unchanged.

use format::Filter;

const LEVEL1: [Filter; 3] = [
    Filter::None,
    Filter::Shuffle { width: 4 },
    Filter::Delta { width: 4 },
];
const LEVEL2: [Filter; 7] = [
    Filter::None,
    Filter::Shuffle { width: 2 },
    Filter::Shuffle { width: 4 },
    Filter::Shuffle { width: 8 },
    Filter::Delta { width: 2 },
    Filter::Delta { width: 4 },
    Filter::Delta { width: 8 },
];

/// Candidate filters tried per chunk at `level`, in tie-break order
/// (none < shuffle < delta, then smaller width first). Levels 0 and 1 try
/// `{none, shuffle-4, delta-4}`; level 2 and above add widths 2 and 8.
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
    }
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
        out
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
                Filter::Shuffle { width: 1 | 2 } | Filter::Delta { width: 1 | 2 }
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
    fn level_1_tries_none_shuffle4_delta4_and_higher_levels_add_widths_2_and_8() {
        use Filter::*;
        assert_eq!(
            candidates(1),
            [None, Shuffle { width: 4 }, Delta { width: 4 }]
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
                if let Filter::Shuffle { width } | Filter::Delta { width } = f {
                    let whole = data.len() / usize::from(width) * usize::from(width);
                    prop_assert_eq!(&filtered[whole..], &data[whole..]);
                }
            }
        }
    }
}
