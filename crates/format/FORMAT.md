# `.gpcz` container format, version 1

All integers are little-endian. A file is a header, a chunk table, and a data section:

```
+-----------------+----------------------------+--------------------------------+
| header (32 B)   | chunk table (24 B × count) | data section (padded payloads) |
+-----------------+----------------------------+--------------------------------+
```

The original data is split into **independent chunks**. Every chunk except possibly the last holds
exactly `chunk_size` uncompressed bytes. No back-reference crosses a chunk boundary, so chunks can be
compressed and decompressed in parallel, and any byte range can be decoded by touching only the
chunks it overlaps.

## Header (32 bytes)

| Offset | Size | Field         | Meaning |
|-------:|-----:|---------------|---------|
| 0      | 4    | `magic`       | `GPCZ` |
| 4      | 2    | `version`     | `1` |
| 6      | 2    | `codec`       | `0` = stored, `1` = LZ4 block |
| 8      | 4    | `chunk_size`  | power of two, 4 KiB ..= 1 MiB |
| 12     | 4    | `chunk_count` | `ceil(total_size / chunk_size)`, so `0` for an empty file |
| 16     | 8    | `total_size`  | uncompressed size in bytes |
| 24     | 4    | `flags`       | bit 0: per-chunk checksums present. Other bits must be 0 |
| 28     | 1    | `level`       | compression level used. Informational: decoders ignore it |
| 29     | 3    | reserved      | must be 0 |

## Chunk table entry (24 bytes)

| Offset | Size | Field          | Meaning |
|-------:|-----:|----------------|---------|
| 0      | 8    | `comp_offset`  | payload offset from the start of the data section. Multiple of 4 |
| 8      | 4    | `comp_size`    | payload size in bytes. **Bit 31 set = stored raw** (payload is the uncompressed bytes) |
| 12     | 4    | `uncomp_size`  | `chunk_size`, or the remainder for the last chunk |
| 16     | 4    | `checksum`     | low 32 bits of xxh3-64 of the uncompressed chunk. Must be 0 when flag bit 0 is clear |
| 20     | 1    | `filter`       | `0` = none, `1` = byte-shuffle, `2` = delta. Applied before compression |
| 21     | 1    | `filter_width` | element width for shuffle/delta: 1, 2, 4 or 8. Must be 0 for none |
| 22     | 2    | reserved       | must be 0 |

## Data section

Payloads appear in chunk order. Each is padded with zeros to a 4-byte boundary, including the last
one, so `comp_offset[i + 1] >= pad4(comp_offset[i] + comp_size[i])` and the data section is at least
`pad4(end of last payload)` bytes long. The 4-byte alignment keeps word addressing simple in WGSL.

Payload encodings:

- **Stored** (bit 31 of `comp_size`): the raw bytes. Encoders store a chunk whenever compression
  doesn't make it smaller, and the `stored` codec stores every chunk.
- **LZ4 block** (codec 1): one standard
  [LZ4 block](https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md) per chunk, with no size
  prefix. Encoders obey the end-of-block rules: the last 5 bytes are literals, and the last match
  starts at least 12 bytes before the end. Match offsets are 1..=65535 and never reach before the
  start of the chunk.

## Filters

Filters are reserved in v1 and land in milestone M7: byte-shuffle and delta over `filter_width`-byte
elements, with the trailing `n % width` bytes untouched. Until a decoder implements a filter, it
must reject chunks that use it.

## Validation

Readers must reject:
- a bad magic, version, codec, flags or nonzero reserved fields
- a chunk size that isn't a power of two in range, or a `chunk_count` that doesn't match `total_size`
- an `uncomp_size` that doesn't match its position
- a misaligned, overlapping or out-of-bounds payload (padding included)
- a stored chunk whose size differs from its `uncomp_size`
- a compressed chunk in a `stored`-codec file
- a nonzero checksum without the checksum flag

Decoders bound every read and write by the chunk's sizes and report malformed payloads as errors.

## Random access

To read `[offset, offset + len)`:
1. Read the header, then the `24 × chunk_count`-byte table.
2. The chunks needed are `offset / chunk_size ..= (offset + len - 1) / chunk_size`.
3. Read only their payloads. They're contiguous, from the first one's `comp_offset` to the last
   one's end.
4. Decode them, then trim the first and last chunk.
