// M3: LZ4 block encoder, one workgroup per chunk. Runs the same algorithm as
// cpu::lz4::encode (the "CPU twin"), so the output is byte-identical:
//   1. match finding, WG_SIZE positions at a time, against a workgroup hash
//      table that only holds positions from earlier blocks (latest wins);
//   2. greedy parse by invocation 0, extending the matches it takes;
//   3. parallel emit: a workgroup prefix sum over encoded sequence sizes gives
//      each sequence its output offset; each sequence's owner writes its
//      header bytes and short literal runs, and the whole workgroup copies
//      literal runs of at least LONG_LITERALS bytes.
// The parse also totals the encoded size: if the block wouldn't shrink the
// chunk, emit is skipped (the host stores such chunks raw).

struct Params {
    chunk_size: u32, // input bytes per chunk (last chunk may be shorter)
    input_len: u32,  // input bytes in this batch
    slot_size: u32,  // output bytes reserved per chunk (multiple of 4)
    probe_len: u32,  // phase-1 match extension cap
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
// chunk_size words per chunk: phase-1 matches, then (in place) the sequences.
@group(0) @binding(1) var<storage, read_write> scratch: array<u32>;
// Zero-initialised; slot_size bytes per chunk. Neighbouring sequences share
// words at their edges, so bytes are merged with atomicOr.
@group(0) @binding(2) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sizes: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

override WG_SIZE: u32 = 64u;
override HASH_LOG: u32 = 12u;

const MIN_MATCH: u32 = 4u;
const MFLIMIT: u32 = 12u;
const LAST_LITERALS: u32 = 5u;
const MAX_OFFSET: u32 = 65535u;
const LONG_LITERALS: u32 = 64u;

// Bucket value = position + 1; 0 = empty (workgroup memory starts zeroed).
var<workgroup> table: array<atomic<u32>, 1u << HASH_LOG>;
var<workgroup> scan: array<u32, WG_SIZE>;
var<workgroup> seq_count: u32;
var<workgroup> encoded_size: u32;

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

fn hash(word: u32) -> u32 {
    return (word * 2654435761u) >> (32u - HASH_LOG);
}

fn put_byte(pos: u32, b: u32) {
    atomicOr(&output[pos >> 2u], b << ((pos & 3u) * 8u));
}

// Continuation bytes of a length whose 4-bit field is 15.
fn length_extra(len: u32) -> u32 {
    if (len < 15u) {
        return 0u;
    }
    return (len - 15u) / 255u + 1u;
}

fn put_length(pos_in: u32, len: u32) -> u32 {
    var pos = pos_in;
    if (len < 15u) {
        return pos;
    }
    var rest = len - 15u;
    loop {
        if (rest < 255u) {
            break;
        }
        put_byte(pos, 255u);
        pos++;
        rest -= 255u;
    }
    put_byte(pos, rest);
    return pos + 1u;
}

// Encoded bytes of one sequence (match_len 0 = final, literals only).
fn encoded_len(lit_len: u32, match_len: u32) -> u32 {
    var size = 1u + length_extra(lit_len) + lit_len;
    if (match_len > 0u) {
        size += 2u + length_extra(match_len - MIN_MATCH);
    }
    return size;
}

// Sequence k of a chunk lives at scratch[q .. q + 4]:
// lit_start, lit_len, match_len (0 = final sequence), offset.
fn sequence_size(q: u32) -> u32 {
    return encoded_len(scratch[q + 1u], scratch[q + 2u]);
}

// Writes the sequence's header bytes, and its literals if the run is short
// (long runs are copied cooperatively; see copy_literals).
fn emit_sequence(q: u32, chunk_start: u32, pos_in: u32) {
    let lit_start = scratch[q];
    let lit_len = scratch[q + 1u];
    let match_len = scratch[q + 2u];
    let offset = scratch[q + 3u];
    var match_code = 0u;
    if (match_len > 0u) {
        match_code = match_len - MIN_MATCH;
    }
    var pos = pos_in;
    put_byte(pos, (min(lit_len, 15u) << 4u) | min(match_code, 15u));
    pos = put_length(pos + 1u, lit_len);
    if (lit_len < LONG_LITERALS) {
        for (var k = 0u; k < lit_len; k++) {
            put_byte(pos + k, in_byte(chunk_start + lit_start + k));
        }
    }
    pos += lit_len;
    if (match_len > 0u) {
        put_byte(pos, offset & 0xFFu);
        put_byte(pos + 1u, offset >> 8u);
        pos = put_length(pos + 2u, match_code);
    }
}

// Cooperative: output[d .. d + len) = input[s .. s + len), one output word
// per step, merged with atomicOr (edge words are shared with neighbours).
fn copy_literals(d: u32, s: u32, len: u32, lid: u32) {
    let last = (d + len - 1u) >> 2u;
    for (var w = (d >> 2u) + lid; w <= last; w += WG_SIZE) {
        var value = 0u;
        for (var b = 0u; b < 4u; b++) {
            let p = w * 4u + b;
            if (p >= d && p < d + len) {
                value |= in_byte(s + (p - d)) << (b * 8u);
            }
        }
        atomicOr(&output[w], value);
    }
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    // Everything up to the parse depends only on uniform values, so barriers
    // below are in uniform control flow.
    let chunk = wid.x + wid.y * nwg.x;
    let chunk_count = (params.input_len + params.chunk_size - 1u) / params.chunk_size;
    if (chunk >= chunk_count) {
        return;
    }
    let start = chunk * params.chunk_size;
    let n = min(params.chunk_size, params.input_len - start);
    let sbase = chunk * params.chunk_size;

    // ---- Phase 1: match finding ----
    if (n >= MFLIMIT) {
        let last_start = n - MFLIMIT;
        let match_limit = n - LAST_LITERALS;
        for (var block = 0u; block <= last_start; block += WG_SIZE) {
            let p = block + lid;
            let in_range = p <= last_start;
            var h = 0u;
            if (in_range) {
                h = hash(in_word(start + p));
                let entry = atomicLoad(&table[h]);
                var m = 0u;
                if (entry != 0u) {
                    let offset = p - (entry - 1u);
                    if (offset <= MAX_OFFSET) {
                        let limit = min(p + params.probe_len, match_limit);
                        var end = p;
                        loop {
                            if (end >= limit || in_byte(start + end - offset) != in_byte(start + end)) {
                                break;
                            }
                            end++;
                        }
                        if (end - p >= MIN_MATCH) {
                            m = ((end - p) << 16u) | offset;
                        }
                    }
                }
                scratch[sbase + p] = m;
            }
            // All lookups of this block see only earlier blocks' positions.
            workgroupBarrier();
            if (in_range) {
                atomicMax(&table[h], p + 1u);
            }
            workgroupBarrier();
        }
    }
    storageBarrier();
    workgroupBarrier();

    // ---- Phase 2: greedy parse (serial) ----
    // Sequence k is written over scratch[4k .. 4k + 4]. That's safe: the k
    // earlier matches each consumed >= 4 positions, so the current match starts
    // at p >= 4k and the next read is at >= p + 4.
    if (lid == 0u) {
        var count = 0u;
        var total = 0u; // encoded block size so far
        var anchor = 0u;
        var p = 0u;
        var match_limit = 0u;
        if (n >= LAST_LITERALS) {
            match_limit = n - LAST_LITERALS;
        }
        loop {
            if (p + MFLIMIT > n) {
                break;
            }
            let m = scratch[sbase + p];
            if ((m >> 16u) < MIN_MATCH) {
                p++;
                continue;
            }
            let offset = m & 0xFFFFu;
            var end = p + (m >> 16u);
            loop {
                if (end >= match_limit || in_byte(start + end - offset) != in_byte(start + end)) {
                    break;
                }
                end++;
            }
            let q = sbase + 4u * count;
            scratch[q] = anchor;
            scratch[q + 1u] = p - anchor;
            scratch[q + 2u] = end - p;
            scratch[q + 3u] = offset;
            total += encoded_len(p - anchor, end - p);
            count++;
            p = end;
            anchor = end;
        }
        let q = sbase + 4u * count;
        scratch[q] = anchor;
        scratch[q + 1u] = n - anchor;
        scratch[q + 2u] = 0u;
        scratch[q + 3u] = 0u;
        total += encoded_len(n - anchor, 0u);
        seq_count = count + 1u;
        encoded_size = total;
    }
    storageBarrier();
    let count = workgroupUniformLoad(&seq_count);
    let total = workgroupUniformLoad(&encoded_size);
    if (total >= n) {
        // Won't shrink the chunk: report the size and skip emit.
        if (lid == 0u) {
            sizes[chunk] = total;
        }
        return;
    }

    // ---- Phase 3: parallel emit ----
    let out_base = chunk * params.slot_size;
    var running = 0u;
    for (var block = 0u; block < count; block += WG_SIZE) {
        let i = block + lid;
        var size = 0u;
        if (i < count) {
            size = sequence_size(sbase + 4u * i);
        }
        scan[lid] = size;
        workgroupBarrier();
        // Inclusive Hillis–Steele scan over the block's sizes.
        for (var d = 1u; d < WG_SIZE; d <<= 1u) {
            var v = 0u;
            if (lid >= d) {
                v = scan[lid - d];
            }
            workgroupBarrier();
            scan[lid] += v;
            workgroupBarrier();
        }
        if (i < count) {
            emit_sequence(sbase + 4u * i, start, out_base + running + scan[lid] - size);
        }
        // Long literal runs of this block's sequences, by the whole workgroup.
        let in_block = min(WG_SIZE, count - block);
        for (var j = 0u; j < in_block; j++) {
            let q = sbase + 4u * (block + j);
            let lit_len = scratch[q + 1u];
            if (lit_len >= LONG_LITERALS) {
                let seq_pos = out_base + running + scan[j] - sequence_size(q);
                copy_literals(seq_pos + 1u + length_extra(lit_len), start + scratch[q], lit_len, lid);
            }
        }
        running += scan[WG_SIZE - 1u];
        workgroupBarrier();
    }
    if (lid == 0u) {
        sizes[chunk] = running;
    }
}
