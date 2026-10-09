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
| 6      | 2    | `codec`       | `0` = stored, `1` = LZ4 block, `2` = GLZ, `3` = GLZ-E |
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

## GLZ block (codec 2)

GLZ stores each sequence's fields in separate arrays, so a GPU can compute any
sequence's lengths, output position and literal source with prefix sums instead of a
serial token parse. All values are little-endian, and every array starts on a 4-byte
boundary (zero padding).

```
u32       seq_count | WIDE_BIT        WIDE_BIT = 1 << 31: extension values are u32, else u16
u32       ext_count                   number of extension values
tokens    [seq_count] u8              lit nibble << 4 | match nibble        (padded to 4)
offsets   [seq_count] u16             match offset; 0 in the last sequence  (padded to 4)
ext       [ext_count] u16 or u32      escaped length remainders, in order   (padded to 4)
literals  [sum of lit_len] u8         all literal bytes, in sequence order
```

- The literal nibble is `min(lit_len, 15)`, and the match nibble is `min(match_len - 4, 15)`.
  A nibble of 15 adds the next extension value. Within a sequence, the literal's
  extension comes before the match's. A sequence's first extension slot is the count of
  15-nibbles in all earlier tokens, which is an exclusive prefix sum.
- Every sequence has a match of at least 4 bytes, except the **last**, which is literals
  only: its match nibble must be 0 and its offset 0.
- Offsets are 1..=65535 and never reach before the start of the chunk. A match with
  offset < length repeats its source pattern, as in LZ4.
- The literal bytes must be consumed exactly, and the output must fill `uncomp_size`
  exactly.
- `seq_count` must be ≥ 1 and ≤ `uncomp_size / 4 + 1` (every sequence but the last
  outputs ≥ 4 bytes). `ext_count` must be ≤ `2 × seq_count` and must equal the number
  of 15-nibbles.

Decoders check the header first, then each sequence in order: literals available, literals
fit the output, final-sequence rules, offset nonzero, offset within the output so far,
match fits the output. Last come the totals. A parallel decoder reports the error of the
lowest failing sequence, which is what a serial decoder reports.

**Dependency elimination** is an optional encoder setting, invisible in the format: within
each group of G consecutive sequences, no match copies bytes from another match's output
in the same group. A decoder that resolves G sequences at a time can then copy all of a
group's matches at once.

## GLZ-E block (codec 3)

A GLZ block whose byte streams are entropy-coded. Decoding a GLZ-E block gives back an
exact GLZ block (codec 2), which then decodes, and is checked, as GLZ. All values are
little-endian, and every part is a whole number of 4-byte words.

```
u32     seq_count | WIDE_BIT          as in GLZ
u32     ext_count                     as in GLZ
u32     lit_count                     literal bytes (sum of lit_len)
stream  tokens    [seq_count]         the GLZ token bytes
stream  off_lo    [seq_count]         low byte of each GLZ offset
stream  off_hi    [seq_count]         high byte of each GLZ offset
ext     [ext_count] u16 or u32        exactly the GLZ extension array (padded to 4)
stream  literals  [lit_count]         the GLZ literal bytes
```

The block must end right after the literal stream.

**Streams.** A stream of `n` byte symbols (n comes from the header) starts with a mode
word:

- **Raw**, `0x00000000`: the `n` bytes, zero-padded to 4.
- **RLE**, `0x0000ss01`: all `n` symbols are `ss`. Nothing follows.
- **Huffman**, `0x00000002`:
  - **Code lengths:** 128 bytes. Byte `i` holds the lengths of symbols `2i` (low nibble)
    and `2i + 1` (high nibble). A length is 0 (symbol unused) or 1..=11. The lengths must
    form a complete prefix code: Σ 2^(11 − len) over used symbols = 2048.
  - **Lane sizes:** `L = clamp(ceil(n / 512), 1, 32)` lanes. Lane `k` codes symbols
    `[k·n / L, (k+1)·n / L)`. Then `L` u16 word counts, zero-padded to 4 bytes.
  - **Lane words:** lane 0's words, then lane 1's, and so on.
  - **Codes:** canonical. Shorter codes come first, and codes of equal length are in
    symbol order. Codes are bit-reversed and packed least significant bit first into u32
    words, as in deflate. A decoder reads 11 bits at a time, treating bits past the lane's
    last word as zero. A lane must not consume bits past its last word.

**Checks** (decoders report the first failure in this order):
1. **Header:** the 3 words are present, else `Truncated`. Then `seq_count` in
   `1..=uncomp_size / 4 + 1`, `ext_count ≤ 2 × seq_count` and `lit_count ≤ uncomp_size`,
   else `BadSequence`.
2. **Each stream, in order:**
   - The mode word is present, else `Truncated`. An unknown mode or nonzero reserved bits
     gives `BadMode`.
   - The raw bytes, code lengths, lane sizes and lane words are present, else
     `Truncated`.
   - The code lengths are complete and ≤ 11, else `BadTable`. (This is checked after the
     lengths are present and before the lane sizes.)
   - The lanes decode in order without reading past their words, else `LaneOverrun`.
3. **Extension words** are present, else `Truncated`, then the literal stream.
4. **Nothing follows** the literal stream, else `SizeMismatch`.
5. **The rebuilt GLZ block** decodes, with the GLZ checks.

**Encoder (the CPU twin and the GPU must match):**
- **Choosing a mode:** RLE if exactly one symbol value occurs. Otherwise Huffman if that's
  strictly smaller than raw, else raw. An empty stream is raw.
- **Code lengths:**
  1. Sort the used symbols by (count, symbol).
  2. Build a two-queue Huffman tree from that leaf queue and a queue of merged nodes,
     merging the two lightest each time. When a leaf and a node weigh the same, the leaf
     comes first.
  3. Fold depths over 11 back with JPEG Annex K.3. For each overlong length `i`, from the
     deepest: while codes remain at `i`, take the deepest `j < i − 1` that has codes; move
     two codes from `i` to `i − 1`, and replace one code at `j` with two at `j + 1`.
  4. Hand out the per-length counts, longest first, to the symbols in sorted order.

## Filters

A chunk's filter is a reversible transform of its uncompressed bytes. The encoder filters the
chunk, then compresses (or stores) the result. A decoder decodes the payload (stored or
compressed) to `uncomp_size` bytes, then applies the inverse filter. With `w = filter_width` and
`n = uncomp_size`, there are `m = n / w` whole elements, and the trailing `n % w` bytes are never
changed by either direction.

- **none (0):** identity; `filter_width` must be 0.
- **byte-shuffle (1):** byte `j` of element `i` moves to position `j * m + i` (all first bytes,
  then all second bytes, ...). The inverse moves it back.
- **delta (2):** elements are little-endian `w`-byte unsigned integers. Element `i > 0` becomes
  `elem[i] - elem[i - 1]` modulo `2^(8w)`; element 0 is unchanged. The inverse is a running sum
  modulo `2^(8w)`.

Checksums are always of the original, **unfiltered** chunk, so they also catch a wrong filter.
The format allows a filter on a stored chunk (its raw payload is then the filtered bytes), but
the encoders here never write one: a chunk that no candidate shrinks is stored unfiltered.

Encoders choose filters per chunk (M7, CPU encoder with `FilterMode::Auto`): the chunk is
compressed under each candidate of the level's set and the smallest block wins; ties go to the
earlier candidate in the order none < shuffle < delta, then smaller width. Level 1 tries
`{none, shuffle-4, delta-4}`; level 2 and above try `{none, shuffle-2, shuffle-4, shuffle-8,
delta-2, delta-4, delta-8}`. If the winning block is not smaller than the chunk, the chunk is
stored raw with filter none. The choice is an encoder setting; decoders only read the table.

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
