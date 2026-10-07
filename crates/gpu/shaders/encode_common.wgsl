// Shared by the LZ4 (lz4_emit.wgsl) and GLZ (glz_emit.wgsl) encoders, which
// the host appends to this file. One workgroup per chunk. Runs the same
// algorithm as the CPU twins (cpu::lz4::encode, cpu::glz), so the output is
// byte-identical:
//   1. find_matches: WG_SIZE positions at a time against a workgroup hash
//      table that only holds positions from earlier blocks (latest wins);
//   2. parse: greedy, by invocation 0, extending the matches it takes and
//      (GLZ, params.groups > 0) refusing matches that would copy from another
//      match's output in the same group of params.groups sequences. It also
//      totals the encoded size, so emit can be skipped when the block wouldn't
//      shrink the chunk (the host stores those raw);
//   3. emit: codec-specific, in the appended file.

struct Params {
    chunk_size: u32, // input bytes per chunk (last chunk may be shorter)
    input_len: u32,  // input bytes in this batch
    slot_size: u32,  // output bytes reserved per chunk (multiple of 4)
    probe_len: u32,  // phase-1 match extension cap
    groups: u32,     // GLZ dependency-elimination group size; 0 = off
    _pad0: u32,
    _pad1: u32,
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

override WG_SIZE: u32 = 64u;
override HASH_LOG: u32 = 12u;
// 0 = LZ4, 1 = GLZ: selects the size accounting in parse().
override CODEC: u32 = 0u;

const MIN_MATCH: u32 = 4u;
const MFLIMIT: u32 = 12u;
const LAST_LITERALS: u32 = 5u;
const MAX_OFFSET: u32 = 65535u;
const LONG_LITERALS: u32 = 64u;
const MAX_GROUP: u32 = 64u; // keep in sync with gpu::encode::MAX_GROUP

// Bucket value = position + 1; 0 = empty (workgroup memory starts zeroed).
var<workgroup> table: array<atomic<u32>, 1u << HASH_LOG>;
var<workgroup> scan: array<u32, WG_SIZE>;
var<workgroup> scan2: array<vec2<u32>, WG_SIZE>;
var<workgroup> seq_count: u32;
var<workgroup> encoded_size: u32;
// GLZ block shape, from the parse.
var<workgroup> glz_ext_count: u32;
var<workgroup> glz_wide: u32;
// Parse (invocation 0): output ranges of the current dependency group's
// matches. In workgroup memory: a private array would be allocated for every
// invocation and spill registers in all phases.
var<workgroup> group_out: array<vec2<u32>, MAX_GROUP>;

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

fn pad4(n: u32) -> u32 {
    return (n + 3u) & ~3u;
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

// Phase 1. Must be called from uniform control flow by the whole workgroup.
fn find_matches(start: u32, n: u32, sbase: u32, lid: u32) {
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
}

// Bytes from `src` on that are free of the group's first `len` match outputs
// (ascending, disjoint): 0 if `src` is inside one, 0xFFFFFFFF if none follows.
fn source_room(src: u32, len: u32) -> u32 {
    // Binary search for the first output that ends after `src`.
    var lo = 0u;
    var hi = len;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) / 2u;
        if (group_out[mid].y <= src) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo == len) {
        return 0xFFFFFFFFu;
    }
    let o = group_out[lo];
    if (o.x <= src) {
        return 0u;
    }
    return o.x - src;
}

// Phase 2 (invocation 0 only). Sequence k is written over scratch[4k .. 4k + 4]
// (lit_start, lit_len, match_len, offset). That's safe: the k earlier matches
// each consumed >= 4 positions, so the current match starts at p >= 4k and the
// next read is at >= p + 4. Sets seq_count and encoded_size (and the GLZ shape).
fn parse(start: u32, n: u32, sbase: u32) {
    var count = 0u;
    var total = 0u;     // LZ4: encoded block size so far
    var lit_total = 0u; // GLZ: literal bytes
    var ext_count = 0u; // GLZ: extension values
    var wide = 0u;      // GLZ: some extension value exceeds u16
    var group_len = 0u;
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
        // With dependency elimination, the match's source pattern
        // [src, src + min(offset, len)) must avoid this group's match outputs,
        // so the match may be capped (or, below MIN_MATCH, dropped).
        var cap = 0xFFFFFFFFu;
        if (params.groups > 0u) {
            if (count % params.groups == 0u) {
                group_len = 0u;
            }
            let room = source_room(p - offset, group_len);
            if (room == 0u) {
                p++;
                continue;
            }
            if (offset > room) {
                cap = room;
            }
            if (cap < MIN_MATCH) {
                p++;
                continue;
            }
        }
        let limit = min(match_limit, p + min(cap, n));
        var end = p + min(m >> 16u, cap);
        loop {
            if (end >= limit || in_byte(start + end - offset) != in_byte(start + end)) {
                break;
            }
            end++;
        }
        if (params.groups > 0u) {
            group_out[group_len] = vec2<u32>(p, end);
            group_len++;
        }
        let q = sbase + 4u * count;
        let lit_len = p - anchor;
        let match_len = end - p;
        scratch[q] = anchor;
        scratch[q + 1u] = lit_len;
        scratch[q + 2u] = match_len;
        scratch[q + 3u] = offset;
        if (CODEC == 0u) {
            total += encoded_len(lit_len, match_len);
        } else {
            lit_total += lit_len;
            if (lit_len >= 15u) {
                ext_count++;
                wide = max(wide, select(0u, 1u, lit_len - 15u > 0xFFFFu));
            }
            if (match_len - MIN_MATCH >= 15u) {
                ext_count++;
                wide = max(wide, select(0u, 1u, match_len - MIN_MATCH - 15u > 0xFFFFu));
            }
        }
        count++;
        p = end;
        anchor = end;
    }
    let q = sbase + 4u * count;
    let lit_len = n - anchor;
    scratch[q] = anchor;
    scratch[q + 1u] = lit_len;
    scratch[q + 2u] = 0u;
    scratch[q + 3u] = 0u;
    count++;
    if (CODEC == 0u) {
        total += encoded_len(lit_len, 0u);
    } else {
        lit_total += lit_len;
        if (lit_len >= 15u) {
            ext_count++;
            wide = max(wide, select(0u, 1u, lit_len - 15u > 0xFFFFu));
        }
        total = 8u + pad4(count) + pad4(2u * count)
            + pad4(ext_count * select(2u, 4u, wide != 0u)) + lit_total;
    }
    seq_count = count;
    encoded_size = total;
    glz_ext_count = ext_count;
    glz_wide = wide;
}
