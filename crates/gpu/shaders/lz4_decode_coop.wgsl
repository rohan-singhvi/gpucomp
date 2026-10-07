// M5: cooperative LZ4 block decoder, one workgroup per chunk (hybrid).
//
// Invocation 0 walks the token stream exactly like the naive decoder (same
// checks, same order, so the same status codes) and copies short literal runs
// and matches itself. A copy of at least LONG_COPY bytes is handed to the whole
// workgroup instead: barrier, every invocation copies whole output words,
// barrier, and invocation 0 resumes. Short sequences cost what they cost in the
// naive kernel, and only long runs pay for barriers, where they're amortised.
//
// An overlapping match (offset < length) is periodic: output byte k equals
// dst[m - offset + k % offset], so it too can be copied in parallel, because
// everything before m is final when the copy starts.
//
// Cooperative copies assemble whole output words and merge each with one
// atomicOr into the zeroed output; edge words shared with neighbouring copies
// merge safely. (Plain stores for words a single copy owns benchmarked the
// same, within ±2%, so the simpler scheme stays; see DECISIONS.md.)

struct ChunkDesc {
    src_offset: u32,
    comp_size: u32,
    dst_offset: u32,
    uncomp_size: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> chunks: array<ChunkDesc>;
@group(0) @binding(2) var<storage, read_write> dst: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> status: array<u32>;

override WG_SIZE: u32 = 32u;
override LONG_COPY: u32 = 16u;

const STORED_BIT: u32 = 0x80000000u;
const MIN_MATCH: u32 = 4u;

const OK: u32 = 0u;
const TRUNCATED: u32 = 1u;
const ZERO_OFFSET: u32 = 2u;
const OFFSET_BEFORE_START: u32 = 3u;
const OUTPUT_OVERFLOW: u32 = 4u;
const SIZE_MISMATCH: u32 = 5u;

const JOB_NONE: u32 = 0u;
const JOB_LITERALS: u32 = 1u;
const JOB_MATCH: u32 = 2u;

// Cooperative copy requested by invocation 0.
var<workgroup> job_kind: u32;
var<workgroup> job_dst: u32;   // absolute dst byte
var<workgroup> job_arg: u32;   // literals: absolute src byte; match: offset
var<workgroup> job_len: u32;
var<workgroup> w_done: u32;
var<workgroup> w_status: u32;
var<workgroup> w_stored: u32;

fn src_byte(i: u32) -> u32 {
    return (src[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu;
}

fn dst_byte(i: u32) -> u32 {
    return (atomicLoad(&dst[i >> 2u]) >> ((i & 3u) * 8u)) & 0xFFu;
}

fn put_byte(i: u32, b: u32) {
    atomicOr(&dst[i >> 2u], b << ((i & 3u) * 8u));
}

// Cooperative: dst[d .. d + len) = src[s .. s + len), one output word per step.
fn copy_literals(d: u32, s: u32, len: u32, lid: u32) {
    let last = (d + len - 1u) >> 2u;
    for (var w = (d >> 2u) + lid; w <= last; w += WG_SIZE) {
        var value = 0u;
        for (var b = 0u; b < 4u; b++) {
            let p = w * 4u + b;
            if (p >= d && p < d + len) {
                value |= src_byte(s + (p - d)) << (b * 8u);
            }
        }
        atomicOr(&dst[w], value);
    }
}

// Cooperative: dst[m .. m + len) = periodic extension of dst[m - offset .. m).
fn copy_match(m: u32, offset: u32, len: u32, lid: u32) {
    let last = (m + len - 1u) >> 2u;
    for (var w = (m >> 2u) + lid; w <= last; w += WG_SIZE) {
        var value = 0u;
        for (var b = 0u; b < 4u; b++) {
            let p = w * 4u + b;
            if (p >= m && p < m + len) {
                value |= dst_byte(m - offset + (p - m) % offset) << (b * 8u);
            }
        }
        atomicOr(&dst[w], value);
    }
}

struct Parser {
    ip: u32,          // read position within the payload
    op: u32,          // write position within the chunk's output
    in_sequence: bool, // literals done; the match part of the sequence is next
    token: u32,
}

// Completes a 4-bit length; returns (length, ok).
fn read_length(ip: ptr<function, u32>, base: u32, n_src: u32, nibble: u32) -> vec2<u32> {
    var len = nibble;
    if (nibble == 15u) {
        loop {
            if (*ip >= n_src) {
                return vec2<u32>(0u, 0u);
            }
            let b = src_byte(base + *ip);
            *ip += 1u;
            len += b;
            if (b != 255u) {
                break;
            }
        }
    }
    return vec2<u32>(len, 1u);
}

fn fail(code: u32) {
    w_status = code;
    w_done = 1u;
}

// Invocation 0: decode serially until a long copy is due (posted as a job) or
// the chunk ends. Checks mirror cpu::lz4::decode::decode_block.
fn run_serial(p: ptr<function, Parser>, c: ChunkDesc) {
    let base = c.src_offset;
    let n_src = c.comp_size & ~STORED_BIT;
    let out = c.dst_offset;
    let n_dst = c.uncomp_size;
    job_kind = JOB_NONE;
    loop {
        if (!(*p).in_sequence) {
            if ((*p).ip >= n_src) {
                fail(TRUNCATED);
                return;
            }
            let token = src_byte(base + (*p).ip);
            (*p).ip += 1u;
            (*p).token = token;
            let lit = read_length(&(*p).ip, base, n_src, token >> 4u);
            if (lit.y == 0u) {
                fail(TRUNCATED);
                return;
            }
            let lit_len = lit.x;
            if (lit_len > n_src - (*p).ip) {
                fail(TRUNCATED);
                return;
            }
            if (lit_len > n_dst - (*p).op) {
                fail(OUTPUT_OVERFLOW);
                return;
            }
            let d = out + (*p).op;
            let s = base + (*p).ip;
            (*p).ip += lit_len;
            (*p).op += lit_len;
            (*p).in_sequence = true;
            if (lit_len >= LONG_COPY) {
                job_kind = JOB_LITERALS;
                job_dst = d;
                job_arg = s;
                job_len = lit_len;
                return;
            }
            for (var k = 0u; k < lit_len; k++) {
                put_byte(d + k, src_byte(s + k));
            }
        }

        // Match part of the current sequence.
        (*p).in_sequence = false;
        if ((*p).ip == n_src) {
            if ((*p).op != n_dst) {
                fail(SIZE_MISMATCH);
            } else {
                w_done = 1u;
            }
            return;
        }
        if (n_src - (*p).ip < 2u) {
            fail(TRUNCATED);
            return;
        }
        let offset = src_byte(base + (*p).ip) | (src_byte(base + (*p).ip + 1u) << 8u);
        (*p).ip += 2u;
        if (offset == 0u) {
            fail(ZERO_OFFSET);
            return;
        }
        if (offset > (*p).op) {
            fail(OFFSET_BEFORE_START);
            return;
        }
        let ml = read_length(&(*p).ip, base, n_src, (*p).token & 15u);
        if (ml.y == 0u) {
            fail(TRUNCATED);
            return;
        }
        let match_len = ml.x + MIN_MATCH;
        if (match_len > n_dst - (*p).op) {
            fail(OUTPUT_OVERFLOW);
            return;
        }
        let m = out + (*p).op;
        (*p).op += match_len;
        if (match_len >= LONG_COPY) {
            job_kind = JOB_MATCH;
            job_dst = m;
            job_arg = offset;
            job_len = match_len;
            return;
        }
        // Byte by byte: overlapping matches must see the bytes just written.
        for (var k = 0u; k < match_len; k++) {
            put_byte(m + k, dst_byte(m + k - offset));
        }
    }
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let chunk = wid.x + wid.y * nwg.x;
    if (chunk >= arrayLength(&chunks)) {
        return;
    }
    let c = chunks[chunk];
    let n_src = c.comp_size & ~STORED_BIT;

    if (lid == 0u) {
        w_done = 0u;
        w_status = OK;
        job_kind = JOB_NONE;
        w_stored = select(0u, 1u, (c.comp_size & STORED_BIT) != 0u);
        // Defensive bounds (the host validated these already).
        if (c.src_offset + n_src > arrayLength(&src) * 4u
            || c.dst_offset + c.uncomp_size > arrayLength(&dst) * 4u) {
            fail(TRUNCATED);
            w_stored = 0u;
        }
    }
    let stored = workgroupUniformLoad(&w_stored);
    if (stored != 0u) {
        // Whole words; see the naive decoder for why a padded tail is harmless.
        let words = (c.uncomp_size + 3u) / 4u;
        for (var w = lid; w < words; w += WG_SIZE) {
            atomicStore(&dst[(c.dst_offset >> 2u) + w], src[(c.src_offset >> 2u) + w]);
        }
        if (lid == 0u) {
            status[chunk] = OK;
        }
        return;
    }

    var parser = Parser(0u, 0u, false, 0u);
    loop {
        if (lid == 0u && w_done == 0u) {
            run_serial(&parser, c);
        }
        // Invocation 0's writes must be visible before anyone copies from them.
        storageBarrier();
        let kind = workgroupUniformLoad(&job_kind);
        if (kind == JOB_NONE) {
            break;
        }
        if (kind == JOB_LITERALS) {
            copy_literals(job_dst, job_arg, job_len, lid);
        } else {
            copy_match(job_dst, job_arg, job_len, lid);
        }
        // The copy must be complete before invocation 0 reads from it.
        storageBarrier();
    }
    if (lid == 0u) {
        status[chunk] = w_status;
    }
}
