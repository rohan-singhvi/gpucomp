//! Host-side planning for a GPU decode: which payload bytes to upload and a
//! descriptor per chunk, all relative to the uploaded span so shaders use u32.

use std::ops::Range;

use format::{pad4, Index, STORED_BIT};

/// One chunk as the shader sees it (matches `ChunkDesc` in the WGSL).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChunkDesc {
    /// Payload start in the uploaded source span (bytes, 4-aligned).
    pub src_offset: u32,
    /// Payload size, with [`STORED_BIT`] set for stored chunks.
    pub comp_size: u32,
    /// Start of this chunk's output in the output span (bytes, 4-aligned).
    pub dst_offset: u32,
    pub uncomp_size: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodePlan {
    /// Byte range of the data section to upload (start is 4-aligned).
    pub src: Range<u64>,
    /// Uncompressed bytes the chunks produce.
    pub dst_len: u64,
    pub chunks: Vec<ChunkDesc>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{what} of {bytes} bytes exceeds the {limit}-byte limit for one GPU batch")]
pub struct TooLarge {
    pub what: &'static str,
    pub bytes: u64,
    pub limit: u64,
}

/// Plans decoding `chunks` of a validated `index`. `limit` caps both spans
/// (the device's storage binding size; u32 addressing in shaders).
pub fn plan(index: &Index, chunks: Range<usize>, limit: u64) -> Result<DecodePlan, TooLarge> {
    let entries = &index.chunks[chunks.clone()];
    let (Some(first), Some(last)) = (entries.first(), entries.last()) else {
        return Ok(DecodePlan {
            src: 0..0,
            dst_len: 0,
            chunks: Vec::new(),
        });
    };
    let src = first.comp_offset..pad4(last.comp_offset + u64::from(last.comp_size));
    let dst_len: u64 = entries.iter().map(|e| u64::from(e.uncomp_size)).sum();
    for (what, bytes) in [
        ("compressed input", src.end - src.start),
        ("output", dst_len),
    ] {
        if bytes > limit {
            return Err(TooLarge { what, bytes, limit });
        }
    }
    let chunk_size = u64::from(index.header.chunk_size);
    let descs = entries
        .iter()
        .enumerate()
        .map(|(k, e)| ChunkDesc {
            src_offset: (e.comp_offset - src.start) as u32,
            comp_size: e.comp_size | if e.stored { STORED_BIT } else { 0 },
            dst_offset: (k as u64 * chunk_size) as u32,
            uncomp_size: e.uncomp_size,
        })
        .collect();
    Ok(DecodePlan {
        src,
        dst_len,
        chunks: descs,
    })
}

/// One GLZ-E chunk as the entropy-decode shader sees it (`GlzeDesc` in
/// glze_decode.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlzeDesc {
    pub src_offset: u32,
    /// Payload size, with [`STORED_BIT`] set for stored chunks.
    pub comp_size: u32,
    /// Where the chunk's rebuilt GLZ block goes in the image buffer (4-aligned).
    pub image_offset: u32,
    pub image_size: u32,
    pub uncomp_size: u32,
    pub _pad: u32,
}

/// Image layout for the GLZ-E chunks `descs` of the uploaded span `src`: each
/// chunk's GLZ block size from its header (a stored chunk's image is its
/// payload; a missing or invalid header gives 0, and the shader reports the
/// error). Returns the descriptors and the image buffer's size.
pub fn glze_layout(descs: &[ChunkDesc], src: &[u8]) -> (Vec<GlzeDesc>, u64) {
    const WIDE_BIT: u32 = 1 << 31;
    let word = |at: usize| u32::from_le_bytes(src[at..at + 4].try_into().unwrap());
    let mut total = 0u64;
    let out = descs
        .iter()
        .map(|d| {
            let comp = d.comp_size & !STORED_BIT;
            let n = d.uncomp_size;
            let image_size = if d.comp_size & STORED_BIT != 0 {
                comp
            } else if comp < 12 {
                0
            } else {
                // Same checks as the shader (and cpu::glze::to_glz).
                let at = d.src_offset as usize;
                let (head, ext, lit) = (word(at), word(at + 4), word(at + 8));
                let count = head & !WIDE_BIT;
                if count == 0 || count > n / 4 + 1 || ext > 2 * count || lit > n {
                    0
                } else {
                    let ext_bytes = ext * if head & WIDE_BIT != 0 { 4 } else { 2 };
                    8 + pad4(u64::from(count)) as u32
                        + pad4(2 * u64::from(count)) as u32
                        + pad4(u64::from(ext_bytes)) as u32
                        + lit
                }
            };
            let g = GlzeDesc {
                src_offset: d.src_offset,
                comp_size: d.comp_size,
                image_offset: total as u32,
                image_size,
                uncomp_size: n,
                _pad: 0,
            };
            total += pad4(u64::from(image_size));
            g
        })
        .collect();
    (out, total)
}

/// GPU bytes one batch of `entries` needs: the uploaded payloads, the output
/// three times over (output buffer, inverse-filter scratch, readback staging)
/// and 20 bytes per chunk (descriptor and status).
pub fn batch_bytes(entries: &[format::ChunkEntry]) -> u64 {
    entries
        .iter()
        .map(|e| pad4(u64::from(e.comp_size)) + 3 * u64::from(e.uncomp_size) + 20)
        .sum()
}

/// Splits `chunks` into consecutive batches of at most `budget` GPU bytes
/// ([`batch_bytes`]) and at most `limit` bytes of payload or output each
/// (the binding limit). A chunk that alone exceeds the budget gets a batch of
/// its own.
pub fn batches(index: &Index, chunks: Range<usize>, budget: u64, limit: u64) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = chunks.start;
    while start < chunks.end {
        let (mut bytes, mut src, mut dst) = (0u64, 0u64, 0u64);
        let mut end = start;
        while end < chunks.end {
            let e = &index.chunks[end];
            let (s, d) = (pad4(u64::from(e.comp_size)), u64::from(e.uncomp_size));
            // GLZ-E also holds each chunk's rebuilt GLZ block; budget it at
            // the uncompressed size (typical blocks are smaller; prepare
            // checks the real total against the binding limit).
            let image = if index.header.codec == format::Codec::GlzE {
                d
            } else {
                0
            };
            let next = bytes + batch_bytes(std::slice::from_ref(e)) + image;
            let fits = next <= budget && src + s <= limit && dst + d <= limit;
            if end > start && !fits {
                break;
            }
            (bytes, src, dst) = (next, src + s, dst + d);
            end += 1;
        }
        out.push(start..end);
        start = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use format::{ChunkEntry, Codec, Filter, Header};

    use super::*;

    fn index() -> Index {
        let entry = |comp_offset, comp_size, stored, uncomp_size| ChunkEntry {
            comp_offset,
            comp_size,
            stored,
            uncomp_size,
            checksum: 0,
            filter: Filter::None,
        };
        Index {
            header: Header {
                codec: Codec::Lz4,
                chunk_size: 4096,
                chunk_count: 3,
                total_size: 10_000,
                checksums: false,
                level: 1,
            },
            chunks: vec![
                entry(0, 101, false, 4096),
                entry(104, 4096, true, 4096),
                entry(4200, 50, false, 1808),
            ],
        }
    }

    fn desc(src_offset: u32, comp_size: u32, dst_offset: u32, uncomp_size: u32) -> ChunkDesc {
        ChunkDesc {
            src_offset,
            comp_size,
            dst_offset,
            uncomp_size,
        }
    }

    #[test]
    fn whole_file_plan_uploads_all_payloads_with_padding() {
        let p = plan(&index(), 0..3, u64::MAX).unwrap();
        assert_eq!(p.src, 0..4252);
        assert_eq!(p.dst_len, 10_000);
        assert_eq!(
            p.chunks,
            [
                desc(0, 101, 0, 4096),
                desc(104, 4096 | STORED_BIT, 4096, 4096),
                desc(4200, 50, 8192, 1808),
            ]
        );
    }

    #[test]
    fn sub_range_plan_is_relative_to_its_first_chunk() {
        let p = plan(&index(), 1..3, u64::MAX).unwrap();
        assert_eq!(p.src, 104..4252);
        assert_eq!(p.dst_len, 4096 + 1808);
        assert_eq!(
            p.chunks,
            [
                desc(0, 4096 | STORED_BIT, 0, 4096),
                desc(4096, 50, 4096, 1808)
            ]
        );
    }

    #[test]
    fn empty_range_plans_nothing() {
        let p = plan(&index(), 2..2, u64::MAX).unwrap();
        assert_eq!((p.src.is_empty(), p.dst_len, p.chunks.len()), (true, 0, 0));
    }

    #[test]
    fn spans_beyond_the_limit_are_rejected() {
        assert_eq!(
            plan(&index(), 0..3, 8192),
            Err(TooLarge {
                what: "output",
                bytes: 10_000,
                limit: 8192
            })
        );
        assert_eq!(
            plan(&index(), 0..2, 4199),
            Err(TooLarge {
                what: "compressed input",
                bytes: 4200,
                limit: 4199
            })
        );
    }

    #[test]
    fn batch_bytes_counts_payloads_three_outputs_and_descriptors() {
        let idx = index();
        // Payloads padded to 4: 104 + 4096 + 52; outputs 10_000.
        assert_eq!(batch_bytes(&idx.chunks), 4252 + 3 * 10_000 + 3 * 20);
        assert_eq!(batch_bytes(&idx.chunks[1..2]), 4096 + 3 * 4096 + 20);
    }

    fn glze_header(count: u32, ext: u32, lit: u32) -> Vec<u8> {
        [count, ext, lit]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }

    #[test]
    fn glze_images_are_sized_from_headers() {
        // Chunk 0: 10 sequences, 3 u16 extension values, 50 literals:
        // 8 + pad4(10) + pad4(20) + pad4(6) + 50 = 8 + 12 + 20 + 8 + 50 = 98.
        // Chunk 1: stored, 40 bytes. Chunk 2: wide (u32) extension values.
        // Chunk 3: invalid (no sequences). Chunk 4: shorter than a header.
        let mut src = glze_header(10, 3, 50);
        src.resize(64, 0);
        src.extend(vec![7u8; 40]);
        src.extend(glze_header(5 | 1 << 31, 2, 0));
        src.resize(src.len() + 4, 0);
        src.extend(glze_header(0, 0, 0));
        src.extend([1, 2, 3, 4]);
        let desc = |src_offset, comp_size, uncomp_size| ChunkDesc {
            src_offset,
            comp_size,
            dst_offset: 0,
            uncomp_size,
        };
        let descs = [
            desc(0, 64, 4096),
            desc(64, 40 | STORED_BIT, 40),
            desc(104, 16, 4096),
            desc(120, 12, 4096),
            desc(132, 4, 4096),
        ];
        let (g, total) = glze_layout(&descs, &src);
        let sizes: Vec<_> = g.iter().map(|d| d.image_size).collect();
        // Chunk 2: 8 + pad4(5) + pad4(10) + pad4(8) + 0 = 8 + 8 + 12 + 8 = 36.
        assert_eq!(sizes, [98, 40, 36, 0, 0]);
        let offsets: Vec<_> = g.iter().map(|d| d.image_offset).collect();
        assert_eq!(offsets, [0, 100, 140, 176, 176]);
        assert_eq!(total, 176);
        assert_eq!(g[1].comp_size, 40 | STORED_BIT);
        assert_eq!((g[0].src_offset, g[0].uncomp_size), (0, 4096));
    }

    #[test]
    fn glze_batches_reserve_room_for_the_images() {
        // GLZ-E chunks also need their rebuilt GLZ blocks (bounded here by
        // the uncompressed size), so the budget that fits all three LZ4
        // chunks doesn't fit them.
        let lz4 = index();
        let mut glze = index();
        glze.header.codec = Codec::GlzE;
        let all = batch_bytes(&lz4.chunks);
        assert_eq!(batches(&lz4, 0..3, all, u64::MAX).len(), 1);
        assert!(batches(&glze, 0..3, all, u64::MAX).len() > 1);
    }

    #[test]
    fn a_big_budget_keeps_everything_in_one_batch() {
        assert_eq!(batches(&index(), 0..3, u64::MAX, u64::MAX), vec![(0..3)]);
    }

    #[test]
    fn a_small_budget_splits_into_consecutive_batches() {
        let idx = index();
        let one = batch_bytes(&idx.chunks[0..1]);
        let two = batch_bytes(&idx.chunks[0..2]);
        assert_eq!(batches(&idx, 0..3, two, u64::MAX), [0..2, 2..3]);
        assert_eq!(batches(&idx, 0..3, one, u64::MAX), [0..1, 1..2, 2..3]);
    }

    #[test]
    fn chunks_bigger_than_the_budget_still_get_a_batch() {
        assert_eq!(batches(&index(), 0..3, 1, u64::MAX), [0..1, 1..2, 2..3]);
    }

    #[test]
    fn the_binding_limit_also_splits_batches() {
        // Outputs of chunks 0 and 1 together (8192) exceed a 5000-byte limit.
        assert_eq!(batches(&index(), 0..3, u64::MAX, 5000), [0..1, 1..2, 2..3]);
    }

    #[test]
    fn batches_cover_sub_ranges_and_empty_ranges() {
        assert_eq!(batches(&index(), 1..3, u64::MAX, u64::MAX), vec![(1..3)]);
        assert!(batches(&index(), 2..2, u64::MAX, u64::MAX).is_empty());
    }
}
