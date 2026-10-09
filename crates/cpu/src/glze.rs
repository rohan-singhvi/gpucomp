//! GLZ-E (codec 3): a GLZ block whose byte streams are entropy-coded (see
//! `format/FORMAT.md`, "GLZ-E block"). Layout, every part a whole number of
//! words:
//!
//! ```text
//! u32     seq_count | WIDE_BIT
//! u32     ext_count
//! u32     lit_count               literal bytes
//! stream  tokens   [seq_count]    (crate::huffman streams: raw, RLE or Huffman)
//! stream  off_lo   [seq_count]    offset & 0xFF
//! stream  off_hi   [seq_count]    offset >> 8
//! ext     [ext_count] u16/u32     raw, padded to 4 bytes (as in GLZ)
//! stream  literals [lit_count]
//! ```
//!
//! The encoder turns a GLZ block into this ([`transcode`]); the decoder turns
//! it back into the identical GLZ block ([`to_glz`]) and decodes that, so
//! sequence checks and errors are GLZ's. The GPU works the same way.

use crate::glz::{field_offsets, GlzError, GlzParams, WIDE_BIT};
use crate::huffman;
use crate::lz4::MIN_MATCH;

/// Compresses `input` as one GLZ-E block.
pub fn encode_block(input: &[u8], params: &GlzParams) -> Vec<u8> {
    transcode(&crate::glz::encode_block(input, params))
}

fn word(src: &[u8], at: usize) -> Option<u32> {
    src.get(at..at + 4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
}

/// The GLZ-E encoding of a valid GLZ block.
pub fn transcode(glz: &[u8]) -> Vec<u8> {
    let head = word(glz, 0).expect("a GLZ block");
    let ext_count = word(glz, 4).expect("a GLZ block") as usize;
    let count = (head & !WIDE_BIT) as usize;
    let (tokens_at, offsets_at, ext_at, literals_at) =
        field_offsets(count, ext_count, head & WIDE_BIT != 0);
    let mut out = Vec::with_capacity(glz.len());
    out.extend_from_slice(&head.to_le_bytes());
    out.extend_from_slice(&(ext_count as u32).to_le_bytes());
    out.extend_from_slice(&((glz.len() - literals_at) as u32).to_le_bytes());
    huffman::encode_stream(&glz[tokens_at..tokens_at + count], &mut out);
    let offsets = &glz[offsets_at..offsets_at + 2 * count];
    let lo: Vec<u8> = offsets.iter().step_by(2).copied().collect();
    let hi: Vec<u8> = offsets.iter().skip(1).step_by(2).copied().collect();
    huffman::encode_stream(&lo, &mut out);
    huffman::encode_stream(&hi, &mut out);
    out.extend_from_slice(&glz[ext_at..literals_at]);
    huffman::encode_stream(&glz[literals_at..], &mut out);
    out
}

/// The GLZ block that a GLZ-E block of a `uncomp_size`-byte chunk encodes.
/// Checks, in order: the header words (present, then counts within GLZ's
/// bounds and `lit_count <= uncomp_size`), each stream in block order, the
/// extension words, and that the block ends after the literal stream.
pub fn to_glz(src: &[u8], uncomp_size: usize) -> Result<Vec<u8>, GlzError> {
    let (Some(head), Some(ext_count), Some(lit_count)) = (word(src, 0), word(src, 4), word(src, 8))
    else {
        return Err(GlzError::Truncated);
    };
    let count = (head & !WIDE_BIT) as usize;
    let wide = head & WIDE_BIT != 0;
    let (ext_count, lit_count) = (ext_count as usize, lit_count as usize);
    if count == 0
        || count > uncomp_size / MIN_MATCH + 1
        || ext_count > 2 * count
        || lit_count > uncomp_size
    {
        return Err(GlzError::BadSequence);
    }
    let stream = |at: usize, n: usize, out: &mut Vec<u8>| {
        huffman::decode_stream(src, at, n, out).map_err(GlzError::Stream)
    };
    let (mut tokens, mut lo, mut hi, mut literals) = (
        Vec::with_capacity(count),
        Vec::with_capacity(count),
        Vec::with_capacity(count),
        Vec::with_capacity(lit_count),
    );
    let at = stream(12, count, &mut tokens)?;
    let at = stream(at, count, &mut lo)?;
    let at = stream(at, count, &mut hi)?;
    let ext_len = (ext_count * if wide { 4 } else { 2 }).next_multiple_of(4);
    let ext = src.get(at..at + ext_len).ok_or(GlzError::Truncated)?;
    let at = stream(at + ext_len, lit_count, &mut literals)?;
    if at != src.len() {
        return Err(GlzError::SizeMismatch);
    }
    let (_, offsets_at, ext_at, literals_at) = field_offsets(count, ext_count, wide);
    let mut glz = Vec::with_capacity(literals_at + lit_count);
    glz.extend_from_slice(&head.to_le_bytes());
    glz.extend_from_slice(&(ext_count as u32).to_le_bytes());
    glz.extend_from_slice(&tokens);
    glz.resize(offsets_at, 0);
    for (l, h) in lo.iter().zip(&hi) {
        glz.extend_from_slice(&[*l, *h]);
    }
    glz.resize(ext_at, 0);
    glz.extend_from_slice(ext);
    debug_assert_eq!(glz.len(), literals_at);
    glz.extend_from_slice(&literals);
    Ok(glz)
}

/// Decodes one GLZ-E block into `dst` (exactly the uncompressed size).
pub fn decode_block(src: &[u8], dst: &mut [u8]) -> Result<(), GlzError> {
    crate::glz::decode_block(&to_glz(src, dst.len())?, dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::huffman::StreamError;

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
        (0..n as u32)
            .flat_map(|i| format!("item {} of {}, ", (i * 7919) % 2003, i % 89).into_bytes())
            .take(n)
            .collect()
    }

    fn inputs() -> Vec<Vec<u8>> {
        vec![
            vec![],
            vec![42],
            text(100),
            text(65_536),
            vec![0; 65_536],
            random(10_000, 1),
            [
                text(20_000),
                random(5_000, 2),
                vec![9; 20_000],
                text(15_536),
            ]
            .concat(),
            vec![0; 1 << 20], // long matches: wide extension values
        ]
    }

    #[test]
    fn blocks_round_trip() {
        for input in inputs() {
            let block = encode_block(&input, &GlzParams::default());
            let mut out = vec![0; input.len()];
            decode_block(&block, &mut out).unwrap();
            assert!(out == input, "{} bytes", input.len());
            assert_eq!(block.len() % 4, 0);
        }
    }

    #[test]
    fn decoding_restores_the_exact_glz_block() {
        for input in inputs() {
            let glz = crate::glz::encode_block(&input, &GlzParams::default());
            assert!(to_glz(&transcode(&glz), input.len()).unwrap() == glz);
        }
    }

    #[test]
    fn text_shrinks_well_below_glz() {
        let input = text(65_536);
        let glz = crate::glz::encode_block(&input, &GlzParams::default());
        let glze = transcode(&glz);
        assert!(
            glze.len() * 5 < glz.len() * 4,
            "{} vs {}",
            glze.len(),
            glz.len()
        );
    }

    #[test]
    fn malformed_blocks_are_rejected() {
        let input = text(20_000);
        let block = encode_block(&input, &GlzParams::default());
        let decode = |b: &[u8]| decode_block(b, &mut vec![0; input.len()]);
        assert_eq!(decode(&block[..8]), Err(GlzError::Truncated));
        let mut bad = block.clone();
        bad[..4].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(decode(&bad), Err(GlzError::BadSequence), "no sequences");
        let mut bad = block.clone();
        bad[8..12].copy_from_slice(&(input.len() as u32 + 1).to_le_bytes());
        assert_eq!(
            decode(&bad),
            Err(GlzError::BadSequence),
            "too many literals"
        );
        let mut bad = block.clone();
        bad[12] = 9; // the token stream's mode
        assert_eq!(decode(&bad), Err(GlzError::Stream(StreamError::BadMode)));
        let mut bad = block.clone();
        bad.extend_from_slice(&[0; 4]);
        assert_eq!(decode(&bad), Err(GlzError::SizeMismatch), "trailing words");
    }

    #[test]
    fn sequence_errors_are_glz_errors() {
        // A GLZ block whose first match has offset 0.
        let input = text(5_000);
        let mut glz = crate::glz::encode_block(&input, &GlzParams::default());
        let count =
            (u32::from_le_bytes(glz[..4].try_into().unwrap()) & !crate::glz::WIDE_BIT) as usize;
        let offsets_at = 8 + count.next_multiple_of(4);
        glz[offsets_at..offsets_at + 2].copy_from_slice(&[0, 0]);
        let mut out = vec![0; input.len()];
        assert_eq!(
            decode_block(&transcode(&glz), &mut out),
            crate::glz::decode_block(&glz, &mut out)
        );
        assert_eq!(
            decode_block(&transcode(&glz), &mut out),
            Err(GlzError::ZeroOffset)
        );
    }
}
