// Shared by the encoder's three kernels, which the host appends to this file
// one at a time (each kernel is its own shader module). Together they run the
// same algorithm as the CPU twins (cpu::lz4::encode, cpu::glz), so the output
// is byte-identical:
//   1. encode_matches.wgsl: one workgroup per chunk; WG_SIZE positions at a
//      time against a workgroup hash table holding only positions from earlier
//      blocks (latest wins). The only kernel with the big table.
//   2. encode_parse.wgsl: ONE INVOCATION PER CHUNK, many chunks per workgroup,
//      no workgroup memory. The parse is serial within a chunk, so it runs
//      chunks side by side in SIMD lanes instead of on 1 lane of a workgroup.
//   3. lz4_emit.wgsl / glz_emit.wgsl: one workgroup per chunk, prefix sums
//      give every sequence its output position.
// They're separate dispatches so each kernel's occupancy is set by its own
// needs: when fused, the 16 KiB hash table limited how many workgroups (and so
// how many serial parses) could run at once (see DECISIONS.md, "encoder split").

struct Params {
    chunk_size: u32, // input bytes per chunk (last chunk may be shorter)
    input_len: u32,  // input bytes in this batch
    slot_size: u32,  // output bytes reserved per chunk (multiple of 4)
    probe_len: u32,  // phase-1 match extension cap
    groups: u32,     // GLZ dependency-elimination group size; 0 = off
    lazy: u32,       // 1 = lazy parse (cpu::lz4::encode::defer_match)
    glze: u32,       // 1 = GLZ-E: emit every GLZ block (it gets transcoded)
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
// chunk_size words per chunk: phase-1 matches, then (in place) the sequences.
@group(0) @binding(1) var<storage, read_write> scratch: array<u32>;
// Zero-initialised; slot_size bytes per chunk. Neighbouring sequences share
// words at their edges, so bytes are merged with atomicOr.
@group(0) @binding(2) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sizes: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
// Per chunk, from the parse: (sequence count, encoded size, GLZ extension
// count, GLZ wide flag).
@group(0) @binding(5) var<storage, read_write> chunk_info: array<vec4<u32>>;
// Per chunk, MAX_GROUP entries: the parse's dependency-group match outputs.
@group(0) @binding(6) var<storage, read_write> group_buf: array<vec2<u32>>;
// Per chunk, PARSE_SEGMENTS entries: the index of the first sequence of each
// parse segment (non-decreasing). Segment k's sequences are stored from
// scratch word seg_len(n) * k of the chunk, 4 words each.
@group(0) @binding(7) var<storage, read_write> segs: array<u32>;
// With hash chains (DEPTH > 1): per position, its first candidate (position
// + 1; 0 = none), written by match finding and followed to older candidates.
@group(0) @binding(8) var<storage, read_write> chain: array<u32>;

const MIN_MATCH: u32 = 4u;
const MFLIMIT: u32 = 12u;
const LAST_LITERALS: u32 = 5u;
const MAX_OFFSET: u32 = 65535u;
const LONG_LITERALS: u32 = 64u;
const MAX_GROUP: u32 = 64u; // keep in sync with gpu::encode::MAX_GROUP
const PARSE_SEGMENTS: u32 = 32u; // keep in sync with gpu::encode::PARSE_SEGMENTS

fn chunk_count() -> u32 {
    return (params.input_len + params.chunk_size - 1u) / params.chunk_size;
}

// Uncompressed bytes in `chunk` (all chunk_size but the last).
fn chunk_len(chunk: u32) -> u32 {
    return min(params.chunk_size, params.input_len - chunk * params.chunk_size);
}

fn in_byte(i: u32) -> u32 {
    return (input[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu;
}

// Little-endian u32 at byte offset i (any alignment); needs i + 4 <= input bytes.
fn in_word(i: u32) -> u32 {
    let shift = (i & 3u) * 8u;
    let lo = input[i >> 2u];
    if (shift == 0u) {
        return lo;
    }
    return (lo >> shift) | (input[(i >> 2u) + 1u] << (32u - shift));
}

// Extends a match: the first position >= `end` (chunk-relative, up to `limit`)
// where input[start + e - offset] != input[start + e]. Compares a word at a
// time (XOR, then trailing zero bits / 8 = equal leading bytes), then bytes.
fn extend(start: u32, end_in: u32, offset: u32, limit: u32) -> u32 {
    var end = end_in;
    loop {
        if (end + 4u > limit) {
            break;
        }
        let diff = in_word(start + end - offset) ^ in_word(start + end);
        if (diff != 0u) {
            return end + countTrailingZeros(diff) / 8u;
        }
        end += 4u;
    }
    loop {
        if (end >= limit || in_byte(start + end - offset) != in_byte(start + end)) {
            break;
        }
        end++;
    }
    return end;
}

fn put_byte(pos: u32, b: u32) {
    atomicOr(&output[pos >> 2u], b << ((pos & 3u) * 8u));
}

fn pad4(n: u32) -> u32 {
    return (n + 3u) & ~3u;
}

// Positions per parse segment of an n-byte chunk: a multiple of 4, so a
// segment's sequences (at most one per 4 positions) fit in its own scratch
// words, and PARSE_SEGMENTS of them never exceed chunk_size.
fn seg_len(n: u32) -> u32 {
    return max(4u, pad4((n + PARSE_SEGMENTS - 1u) / PARSE_SEGMENTS));
}

// Continuation bytes of an LZ4 length whose 4-bit field is 15.
fn length_extra(len: u32) -> u32 {
    if (len < 15u) {
        return 0u;
    }
    return (len - 15u) / 255u + 1u;
}

// Encoded bytes of one LZ4 sequence (match_len 0 = final, literals only).
fn encoded_len(lit_len: u32, match_len: u32) -> u32 {
    var size = 1u + length_extra(lit_len) + lit_len;
    if (match_len > 0u) {
        size += 2u + length_extra(match_len - MIN_MATCH);
    }
    return size;
}

