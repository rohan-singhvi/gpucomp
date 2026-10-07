//! LZ4 block format: a hand-written decoder and the GPU-twin greedy encoder.
//! Spec: <https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md>.

pub mod decode;
pub mod encode;

/// Minimum match length.
pub const MIN_MATCH: usize = 4;
/// A match may not start within the last `MFLIMIT` bytes of a block.
pub const MFLIMIT: usize = 12;
/// The last `LAST_LITERALS` bytes of a block are always literals.
pub const LAST_LITERALS: usize = 5;
pub const MAX_OFFSET: usize = 65_535;

/// Worst-case compressed size of an `n`-byte block.
pub fn max_compressed_size(n: usize) -> usize {
    n + n / 255 + 16
}
