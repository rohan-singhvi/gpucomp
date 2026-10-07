//! The `.gpcz` container format (see `FORMAT.md`): header, chunk table and
//! validation. Payload encoding/decoding lives in the `cpu` and `gpu` crates.

use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;

pub const MAGIC: [u8; 4] = *b"GPCZ";
pub const VERSION: u16 = 1;
pub const HEADER_SIZE: usize = 32;
pub const ENTRY_SIZE: usize = 24;
pub const MIN_CHUNK_SIZE: u32 = 4 << 10;
pub const MAX_CHUNK_SIZE: u32 = 1 << 20;
pub const DEFAULT_CHUNK_SIZE: u32 = 64 << 10;
/// Set in the serialized `comp_size` when the chunk is stored raw.
pub const STORED_BIT: u32 = 1 << 31;
const FLAG_CHECKSUMS: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FormatError {
    #[error("input too short: need {need} bytes, have {have}")]
    Truncated { need: u64, have: u64 },
    #[error("not a gpcz file (bad magic)")]
    BadMagic,
    #[error("unsupported format version {0}")]
    UnsupportedVersion(u16),
    #[error("unknown codec id {0}")]
    UnknownCodec(u16),
    #[error("unknown header flags {0:#x}")]
    UnknownFlags(u32),
    #[error("unknown filter id {0}")]
    UnknownFilter(u8),
    #[error("invalid filter width {width} for filter id {id}")]
    BadFilterWidth { id: u8, width: u8 },
    #[error("reserved field is nonzero")]
    NonzeroReserved,
    #[error("chunk size {0} is not a power of two in 4 KiB..=1 MiB")]
    BadChunkSize(u32),
    #[error("chunk count {actual} doesn't match total size (expected {expected})")]
    ChunkCountMismatch { expected: u64, actual: u32 },
    #[error("chunk {index}: {reason}")]
    BadChunk { index: usize, reason: &'static str },
    #[error("range {offset}+{len} is outside the {total}-byte original")]
    RangeOutOfBounds { offset: u64, len: u64, total: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Stored = 0,
    Lz4 = 1,
}

/// A reversible per-chunk transform applied before compression (§4a of the plan).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    None,
    Shuffle { width: u8 },
    Delta { width: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub codec: Codec,
    pub chunk_size: u32,
    pub chunk_count: u32,
    pub total_size: u64,
    pub checksums: bool,
    pub level: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkEntry {
    /// Offset of the payload from the start of the data section (4-aligned).
    pub comp_offset: u64,
    /// Payload size in bytes, without the stored bit.
    pub comp_size: u32,
    pub stored: bool,
    pub uncomp_size: u32,
    /// Low 32 bits of xxh3-64 of the uncompressed chunk; 0 without checksums.
    pub checksum: u32,
    pub filter: Filter,
}

/// Header plus chunk table: everything needed to locate any chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Index {
    pub header: Header,
    pub chunks: Vec<ChunkEntry>,
}

/// Rounds `n` up to a multiple of 4 (payload padding).
pub fn pad4(n: u64) -> u64 {
    n.div_ceil(4) * 4
}

/// Number of chunks needed for `total_size` bytes.
pub fn chunk_count_for(total_size: u64, chunk_size: u32) -> u64 {
    total_size.div_ceil(u64::from(chunk_size))
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

fn need(bytes: &[u8], need: u64) -> Result<(), FormatError> {
    if (bytes.len() as u64) < need {
        return Err(FormatError::Truncated {
            need,
            have: bytes.len() as u64,
        });
    }
    Ok(())
}

impl Filter {
    fn to_raw(self) -> (u8, u8) {
        match self {
            Filter::None => (0, 0),
            Filter::Shuffle { width } => (1, width),
            Filter::Delta { width } => (2, width),
        }
    }

    fn from_raw(id: u8, width: u8) -> Result<Self, FormatError> {
        let elem = matches!(width, 1 | 2 | 4 | 8);
        match (id, width) {
            (0, 0) => Ok(Filter::None),
            (1, w) if elem => Ok(Filter::Shuffle { width: w }),
            (2, w) if elem => Ok(Filter::Delta { width: w }),
            (0..=2, _) => Err(FormatError::BadFilterWidth { id, width }),
            _ => Err(FormatError::UnknownFilter(id)),
        }
    }
}

impl Header {
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut b = [0; HEADER_SIZE];
        b[0..4].copy_from_slice(&MAGIC);
        b[4..6].copy_from_slice(&VERSION.to_le_bytes());
        b[6..8].copy_from_slice(&(self.codec as u16).to_le_bytes());
        b[8..12].copy_from_slice(&self.chunk_size.to_le_bytes());
        b[12..16].copy_from_slice(&self.chunk_count.to_le_bytes());
        b[16..24].copy_from_slice(&self.total_size.to_le_bytes());
        let flags = if self.checksums { FLAG_CHECKSUMS } else { 0 };
        b[24..28].copy_from_slice(&flags.to_le_bytes());
        b[28] = self.level;
        b
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, FormatError> {
        need(bytes, HEADER_SIZE as u64)?;
        if bytes[0..4] != MAGIC {
            return Err(FormatError::BadMagic);
        }
        let version = u16_at(bytes, 4);
        if version != VERSION {
            return Err(FormatError::UnsupportedVersion(version));
        }
        let codec = match u16_at(bytes, 6) {
            0 => Codec::Stored,
            1 => Codec::Lz4,
            id => return Err(FormatError::UnknownCodec(id)),
        };
        let flags = u32_at(bytes, 24);
        if flags & !FLAG_CHECKSUMS != 0 {
            return Err(FormatError::UnknownFlags(flags));
        }
        if bytes[29..32] != [0, 0, 0] {
            return Err(FormatError::NonzeroReserved);
        }
        Ok(Header {
            codec,
            chunk_size: u32_at(bytes, 8),
            chunk_count: u32_at(bytes, 12),
            total_size: u64_at(bytes, 16),
            checksums: flags & FLAG_CHECKSUMS != 0,
            level: bytes[28],
        })
    }

    /// Uncompressed size of chunk `index`.
    pub fn uncomp_size_of(&self, index: u32) -> u32 {
        let start = u64::from(index) * u64::from(self.chunk_size);
        (self.total_size - start).min(u64::from(self.chunk_size)) as u32
    }
}

impl ChunkEntry {
    pub fn to_bytes(&self) -> [u8; ENTRY_SIZE] {
        let mut b = [0; ENTRY_SIZE];
        b[0..8].copy_from_slice(&self.comp_offset.to_le_bytes());
        let size = self.comp_size | if self.stored { STORED_BIT } else { 0 };
        b[8..12].copy_from_slice(&size.to_le_bytes());
        b[12..16].copy_from_slice(&self.uncomp_size.to_le_bytes());
        b[16..20].copy_from_slice(&self.checksum.to_le_bytes());
        (b[20], b[21]) = self.filter.to_raw();
        b
    }

    pub fn parse(bytes: &[u8; ENTRY_SIZE]) -> Result<Self, FormatError> {
        let filter = Filter::from_raw(bytes[20], bytes[21])?;
        if bytes[22..24] != [0, 0] {
            return Err(FormatError::NonzeroReserved);
        }
        let size = u32_at(bytes, 8);
        Ok(ChunkEntry {
            comp_offset: u64_at(bytes, 0),
            comp_size: size & !STORED_BIT,
            stored: size & STORED_BIT != 0,
            uncomp_size: u32_at(bytes, 12),
            checksum: u32_at(bytes, 16),
            filter,
        })
    }
}

impl Index {
    /// Byte offset of the data section from the start of the file.
    pub fn data_offset(&self) -> u64 {
        HEADER_SIZE as u64 + ENTRY_SIZE as u64 * u64::from(self.header.chunk_count)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data_offset() as usize);
        out.extend_from_slice(&self.header.to_bytes());
        for chunk in &self.chunks {
            out.extend_from_slice(&chunk.to_bytes());
        }
        out
    }

    /// Parses the header and chunk table from the start of `bytes`.
    /// Doesn't validate chunk layout; call [`Index::validate`].
    pub fn parse(bytes: &[u8]) -> Result<Self, FormatError> {
        let header = Header::parse(bytes)?;
        let end = HEADER_SIZE as u64 + ENTRY_SIZE as u64 * u64::from(header.chunk_count);
        need(bytes, end)?;
        let chunks = bytes[HEADER_SIZE..end as usize]
            .chunks_exact(ENTRY_SIZE)
            .map(|e| ChunkEntry::parse(e.try_into().unwrap()))
            .collect::<Result<_, _>>()?;
        Ok(Index { header, chunks })
    }

    /// Checks the table against the header and a data section of `data_len`
    /// bytes: chunk size and count, per-chunk sizes, alignment, ordering, bounds.
    pub fn validate(&self, data_len: u64) -> Result<(), FormatError> {
        let h = &self.header;
        let size = h.chunk_size;
        if !size.is_power_of_two() || !(MIN_CHUNK_SIZE..=MAX_CHUNK_SIZE).contains(&size) {
            return Err(FormatError::BadChunkSize(size));
        }
        let expected = chunk_count_for(h.total_size, size);
        if expected != u64::from(h.chunk_count) || self.chunks.len() != h.chunk_count as usize {
            return Err(FormatError::ChunkCountMismatch {
                expected,
                actual: h.chunk_count,
            });
        }
        let mut next_free = 0u64;
        for (index, c) in self.chunks.iter().enumerate() {
            let bad = |reason| Err(FormatError::BadChunk { index, reason });
            if c.uncomp_size != h.uncomp_size_of(index as u32) {
                return bad("uncompressed size doesn't match the header");
            }
            if c.comp_offset % 4 != 0 {
                return bad("payload offset isn't 4-byte aligned");
            }
            if c.comp_offset < next_free {
                return bad("payload overlaps the previous chunk");
            }
            let end = c.comp_offset + u64::from(c.comp_size);
            // Each payload is padded to 4 bytes, so its padding must be present too.
            if pad4(end) > data_len {
                return bad("payload extends past the end of the data section");
            }
            if c.stored && c.comp_size != c.uncomp_size {
                return bad("stored chunk's size differs from its uncompressed size");
            }
            if !c.stored && h.codec == Codec::Stored {
                return bad("compressed chunk in a stored-codec file");
            }
            if !h.checksums && c.checksum != 0 {
                return bad("checksum present but the header has no checksum flag");
            }
            next_free = pad4(end);
        }
        Ok(())
    }

    /// Indices of the chunks that overlap `[offset, offset + len)` of the original.
    pub fn chunks_for_range(&self, offset: u64, len: u64) -> Result<Range<usize>, FormatError> {
        let total = self.header.total_size;
        if offset.checked_add(len).is_none_or(|end| end > total) {
            return Err(FormatError::RangeOutOfBounds { offset, len, total });
        }
        if len == 0 {
            return Ok(0..0);
        }
        let size = u64::from(self.header.chunk_size);
        let first = offset / size;
        let last = (offset + len - 1) / size;
        Ok(first as usize..last as usize + 1)
    }
}

/// Error from [`read_index`]: I/O or format.
#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Format(#[from] FormatError),
}

/// Low 32 bits of xxh3-64, the per-chunk checksum.
pub fn checksum(bytes: &[u8]) -> u32 {
    xxhash_rust::xxh3::xxh3_64(bytes) as u32
}

/// Reads and validates the header and chunk table, checking payload bounds
/// against the reader's total length.
pub fn read_index<R: Read + Seek>(reader: &mut R) -> Result<Index, ReadError> {
    let file_len = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(0))?;
    let mut header = [0u8; HEADER_SIZE];
    read_exact_or_truncated(reader, &mut header, file_len)?;
    let parsed = Header::parse(&header)?;
    let table_len = ENTRY_SIZE as u64 * u64::from(parsed.chunk_count);
    if HEADER_SIZE as u64 + table_len > file_len {
        return Err(FormatError::Truncated {
            need: HEADER_SIZE as u64 + table_len,
            have: file_len,
        }
        .into());
    }
    let mut bytes = header.to_vec();
    bytes.resize(HEADER_SIZE + table_len as usize, 0);
    reader.read_exact(&mut bytes[HEADER_SIZE..])?;
    let index = Index::parse(&bytes)?;
    index.validate(file_len - index.data_offset())?;
    Ok(index)
}

fn read_exact_or_truncated<R: Read>(
    reader: &mut R,
    buf: &mut [u8],
    have: u64,
) -> Result<(), ReadError> {
    if (buf.len() as u64) > have {
        return Err(FormatError::Truncated {
            need: buf.len() as u64,
            have,
        }
        .into());
    }
    reader.read_exact(buf)?;
    Ok(())
}

#[cfg(test)]
mod tests;
