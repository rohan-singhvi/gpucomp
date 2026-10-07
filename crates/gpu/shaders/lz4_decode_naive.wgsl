// M2: naive LZ4 block decoder. One invocation decodes one whole chunk,
// mirroring cpu::lz4::decode::decode_block step for step (same checks, same
// order), and writes a status code per chunk instead of failing.

struct ChunkDesc {
    src_offset: u32,  // payload start in `src`, bytes, 4-aligned
    comp_size: u32,   // payload bytes; STORED_BIT = stored raw
    dst_offset: u32,  // output start in `dst`, bytes, 4-aligned
    uncomp_size: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> chunks: array<ChunkDesc>;
// Zero-initialised. Each invocation owns the words of its own chunk (chunk
// outputs start 4-aligned and only the last chunk ends unaligned), so plain
// read-modify-write is race-free.
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read_write> status: array<u32>;

override WG_SIZE: u32 = 64u;

const STORED_BIT: u32 = 0x80000000u;
const MIN_MATCH: u32 = 4u;

// Status codes: keep in sync with gpu::decode::ChunkStatus.
const OK: u32 = 0u;
const TRUNCATED: u32 = 1u;
const ZERO_OFFSET: u32 = 2u;
const OFFSET_BEFORE_START: u32 = 3u;
const OUTPUT_OVERFLOW: u32 = 4u;
const SIZE_MISMATCH: u32 = 5u;

fn src_byte(i: u32) -> u32 {
    return (src[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu;
}

fn dst_byte(i: u32) -> u32 {
    return (dst[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu;
}

// Bytes are written once each, into zeroed words, so OR is enough.
fn put_byte(i: u32, b: u32) {
    dst[i >> 2u] = dst[i >> 2u] | (b << ((i & 3u) * 8u));
}

fn decode_chunk(c: ChunkDesc) -> u32 {
    let base = c.src_offset;
    let n_src = c.comp_size & ~STORED_BIT;
    let out = c.dst_offset;
    let n_dst = c.uncomp_size;

    // Defensive bounds (the host validated these already).
    if (base + n_src > arrayLength(&src) * 4u || out + n_dst > arrayLength(&dst) * 4u) {
        return TRUNCATED;
    }

    if ((c.comp_size & STORED_BIT) != 0u) {
        // Whole words: the payload is padded, and a partial last word only
        // occurs in the final chunk, past which nothing is read back.
        let words = (n_dst + 3u) / 4u;
        for (var w = 0u; w < words; w++) {
            dst[(out >> 2u) + w] = src[(base >> 2u) + w];
        }
        return OK;
    }

    var ip = 0u; // read position within the payload
    var op = 0u; // write position within the chunk's output
    // Every iteration consumes at least one payload byte, so this terminates.
    loop {
        if (ip >= n_src) {
            return TRUNCATED;
        }
        let token = src_byte(base + ip);
        ip++;

        var lit_len = token >> 4u;
        if (lit_len == 15u) {
            loop {
                if (ip >= n_src) {
                    return TRUNCATED;
                }
                let b = src_byte(base + ip);
                ip++;
                lit_len += b;
                if (b != 255u) {
                    break;
                }
            }
        }
        if (lit_len > n_src - ip) {
            return TRUNCATED;
        }
        if (lit_len > n_dst - op) {
            return OUTPUT_OVERFLOW;
        }
        for (var k = 0u; k < lit_len; k++) {
            put_byte(out + op + k, src_byte(base + ip + k));
        }
        ip += lit_len;
        op += lit_len;

        // The final sequence is literals only.
        if (ip == n_src) {
            break;
        }

        if (n_src - ip < 2u) {
            return TRUNCATED;
        }
        let offset = src_byte(base + ip) | (src_byte(base + ip + 1u) << 8u);
        ip += 2u;
        if (offset == 0u) {
            return ZERO_OFFSET;
        }
        if (offset > op) {
            return OFFSET_BEFORE_START;
        }

        var match_len = token & 15u;
        if (match_len == 15u) {
            loop {
                if (ip >= n_src) {
                    return TRUNCATED;
                }
                let b = src_byte(base + ip);
                ip++;
                match_len += b;
                if (b != 255u) {
                    break;
                }
            }
        }
        match_len += MIN_MATCH;
        if (match_len > n_dst - op) {
            return OUTPUT_OVERFLOW;
        }
        // Byte by byte: with offset < match_len the source overlaps the output
        // being written, and each byte must see the ones written before it.
        for (var k = 0u; k < match_len; k++) {
            put_byte(out + op + k, dst_byte(out + op + k - offset));
        }
        op += match_len;
    }
    if (op != n_dst) {
        return SIZE_MISMATCH;
    }
    return OK;
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = gid.x + gid.y * nwg.x * WG_SIZE;
    if (i >= arrayLength(&chunks)) {
        return;
    }
    status[i] = decode_chunk(chunks[i]);
}
