//! Hand-written LZ4 block decoder: simple, readable and bounds-checked.
//! The GPU decoder (M2) mirrors this loop.

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DecodeError {
    #[error("compressed block ends mid-sequence")]
    Truncated,
    #[error("match offset is zero")]
    ZeroOffset,
    #[error("match offset points before the start of the block")]
    OffsetBeforeStart,
    #[error("block decodes to more bytes than expected")]
    OutputOverflow,
    #[error("block decodes to {actual} bytes, expected {expected}")]
    SizeMismatch { expected: usize, actual: usize },
}

/// Decodes one LZ4 block into `dst`, which must be exactly the uncompressed size.
pub fn decode_block(src: &[u8], dst: &mut [u8]) -> Result<(), DecodeError> {
    let mut ip = 0; // read position in src
    let mut op = 0; // write position in dst
    loop {
        let token = *src.get(ip).ok_or(DecodeError::Truncated)?;
        ip += 1;

        let lit_len = read_length(src, &mut ip, usize::from(token >> 4))?;
        let lit_end = ip.checked_add(lit_len).ok_or(DecodeError::Truncated)?;
        let literals = src.get(ip..lit_end).ok_or(DecodeError::Truncated)?;
        dst.get_mut(op..op + lit_len)
            .ok_or(DecodeError::OutputOverflow)?
            .copy_from_slice(literals);
        ip = lit_end;
        op += lit_len;

        // The final sequence is literals only.
        if ip == src.len() {
            break;
        }

        let offset_bytes = src.get(ip..ip + 2).ok_or(DecodeError::Truncated)?;
        let offset = usize::from(u16::from_le_bytes([offset_bytes[0], offset_bytes[1]]));
        ip += 2;
        if offset == 0 {
            return Err(DecodeError::ZeroOffset);
        }
        if offset > op {
            return Err(DecodeError::OffsetBeforeStart);
        }
        let match_len = read_length(src, &mut ip, usize::from(token & 0x0F))? + super::MIN_MATCH;
        if match_len > dst.len() - op {
            return Err(DecodeError::OutputOverflow);
        }
        // Byte by byte: when offset < match_len the source overlaps what we write.
        for i in op..op + match_len {
            dst[i] = dst[i - offset];
        }
        op += match_len;
    }
    if op != dst.len() {
        return Err(DecodeError::SizeMismatch {
            expected: dst.len(),
            actual: op,
        });
    }
    Ok(())
}

/// Completes a 4-bit length field: 15 means "add following bytes until one is < 255".
fn read_length(src: &[u8], ip: &mut usize, nibble: usize) -> Result<usize, DecodeError> {
    let mut len = nibble;
    if nibble == 15 {
        loop {
            let b = *src.get(*ip).ok_or(DecodeError::Truncated)?;
            *ip += 1;
            len += usize::from(b);
            if b != 255 {
                break;
            }
        }
    }
    Ok(len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(src: &[u8], n: usize) -> Result<Vec<u8>, DecodeError> {
        let mut out = vec![0xEE; n];
        decode_block(src, &mut out).map(|()| out)
    }

    #[test]
    fn literals_only_block() {
        assert_eq!(decode(b"\x50hello", 5).unwrap(), b"hello");
    }

    #[test]
    fn overlapping_match_repeats_recent_bytes() {
        // 'a', then match offset 1 length 9, then an empty final sequence.
        let src = [0x15, b'a', 1, 0, 0x00];
        assert_eq!(decode(&src, 10).unwrap(), b"aaaaaaaaaa");
    }

    #[test]
    fn match_with_offset_two_copies_a_pair() {
        let src = [0x22, b'a', b'b', 2, 0, 0x00];
        assert_eq!(decode(&src, 8).unwrap(), b"abababab");
    }

    #[test]
    fn long_literal_length_uses_255_continuation() {
        let lit_len = 15 + 255 + 5;
        let mut src = vec![0xF0, 255, 5];
        src.extend((0..lit_len).map(|i| i as u8));
        let expected: Vec<u8> = (0..lit_len).map(|i| i as u8).collect();
        assert_eq!(decode(&src, lit_len).unwrap(), expected);
    }

    #[test]
    fn long_match_length_uses_255_continuation() {
        // 1 literal, match offset 1 of length 4 + 15 + 255 + 1 = 275.
        let src = [0x1F, b'z', 1, 0, 255, 1, 0x00];
        assert_eq!(decode(&src, 276).unwrap(), vec![b'z'; 276]);
    }

    #[test]
    fn empty_block_is_a_single_zero_token() {
        assert_eq!(decode(&[0x00], 0).unwrap(), b"");
    }

    #[test]
    fn decodes_lz4_flex_output() {
        let text = b"the quick brown fox jumps over the lazy dog. ".repeat(50);
        let compressed = lz4_flex::block::compress(&text);
        assert_eq!(decode(&compressed, text.len()).unwrap(), text);
    }

    #[test]
    fn rejects_empty_input() {
        assert_eq!(decode(&[], 0), Err(DecodeError::Truncated));
    }

    #[test]
    fn rejects_truncated_literals() {
        assert_eq!(decode(b"\x50hel", 5), Err(DecodeError::Truncated));
    }

    #[test]
    fn rejects_truncated_offset() {
        assert_eq!(decode(&[0x14, b'a', 1], 9), Err(DecodeError::Truncated));
    }

    #[test]
    fn rejects_truncated_length_continuation() {
        assert_eq!(decode(&[0xF0, 255], 300), Err(DecodeError::Truncated));
    }

    #[test]
    fn rejects_zero_offset() {
        assert_eq!(
            decode(&[0x14, b'a', 0, 0, 0x00], 9),
            Err(DecodeError::ZeroOffset)
        );
    }

    #[test]
    fn rejects_offset_before_block_start() {
        assert_eq!(
            decode(&[0x14, b'a', 2, 0, 0x00], 9),
            Err(DecodeError::OffsetBeforeStart)
        );
    }

    #[test]
    fn rejects_literals_overflowing_output() {
        assert_eq!(decode(b"\x50hello", 4), Err(DecodeError::OutputOverflow));
    }

    #[test]
    fn rejects_match_overflowing_output() {
        assert_eq!(
            decode(&[0x15, b'a', 1, 0, 0x00], 5),
            Err(DecodeError::OutputOverflow)
        );
    }

    #[test]
    fn rejects_output_shorter_than_expected() {
        assert_eq!(
            decode(b"\x50hello", 6),
            Err(DecodeError::SizeMismatch {
                expected: 6,
                actual: 5
            })
        );
    }

    #[test]
    fn rejects_match_after_output_is_full() {
        assert_eq!(
            decode(b"\x20hi\x01\x00", 2),
            Err(DecodeError::OutputOverflow)
        );
    }
}
