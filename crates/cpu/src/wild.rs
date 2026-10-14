//! Wild copies for the LZ decoders: short copies as one 16-byte move when
//! both sides have room, the overshoot overwritten by what follows.

/// `dst[op..op + len] = src[..len]`, checked by the caller. Output is written
/// in order and matches read only bytes before `op`, so a short run is one
/// 16-byte copy when both sides have room: what follows overwrites the rest.
#[inline(always)]
pub(crate) fn copy_literals(dst: &mut [u8], op: usize, src: &[u8], len: usize) {
    if len <= 16 && op + 16 <= dst.len() && src.len() >= 16 {
        dst[op..op + 16].copy_from_slice(&src[..16]);
    } else {
        dst[op..op + len].copy_from_slice(&src[..len]);
    }
}

/// `dst[op + k] = dst[op + k - offset]` for k in 0..len, in order (an offset
/// below `len` repeats the pattern), checked by the caller. With an offset of
/// at least 16 and room for the overshoot, 16 bytes at a time: each step reads
/// only bytes already written.
#[inline(always)]
pub(crate) fn copy_match(dst: &mut [u8], op: usize, offset: usize, len: usize) {
    if offset >= 16 && op + len + 16 <= dst.len() {
        let mut k = 0;
        while k < len {
            let (done, rest) = dst.split_at_mut(op + k);
            rest[..16].copy_from_slice(&done[op + k - offset..op + k - offset + 16]);
            k += 16;
        }
    } else if offset >= 8 && op + len + 8 <= dst.len() {
        let mut k = 0;
        while k < len {
            let (done, rest) = dst.split_at_mut(op + k);
            rest[..8].copy_from_slice(&done[op + k - offset..op + k - offset + 8]);
            k += 8;
        }
    } else {
        for k in op..op + len {
            dst[k] = dst[k - offset];
        }
    }
}
