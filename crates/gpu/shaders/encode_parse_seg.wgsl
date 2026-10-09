// Encoder kernel 2: greedy parse, ONE WORKGROUP PER CHUNK, one lane per
// segment of seg_len(n) positions (appended to encode_common.wgsl). Without
// dependency elimination a greedy step depends only on its position
// (p -> p + 1 for a literal, p -> the extended match end), so the serial
// parse can be computed exactly in parallel:
//   A. Each lane walks from its segment's start until it leaves the segment,
//      marking the positions it visits (MARK in the match word). Its last
//      match is extended at most SPEC_EXTRA past the segment: in long runs
//      every lane's walk lands in the same match, and only the lane on the
//      true path should extend it to the end ("inexact" exits wait for that).
//   B. The true walk enters segment k where it leaves segment k - 1. In
//      rounds, each lane whose entry changed (and is known) re-walks from it
//      until it reaches one of its marks (from there it follows its first
//      walk, so it leaves where that walk did, extended now if inexact) or
//      leaves the segment. Until no exit changes; walks usually merge within
//      a few bytes (DECISIONS.md, segmented parse).
//   C. Each lane walks its part of the true path and writes its sequences from
//      its segment's first scratch word. The j-th match starts at least 4j past
//      the segment start, so writes stay behind the walk (which only reads
//      ahead), inside the segment, and away from other lanes. The first
//      sequence's literals start at the last match end before the segment,
//      a match leaving the segment ends at the lane's exit (no re-extension),
//      found by a scan; the last lane adds the final literals-only sequence.
// Output (sequences, chunk_info, sizes) equals encode_parse.wgsl's, except
// that sequences are stored per segment: see `segs`.

// 0 = LZ4, 1 = GLZ: selects the size accounting.
override CODEC: u32 = 0u;
const MARK: u32 = 0x80000000u;
const NONE: u32 = 0xFFFFFFFFu;
// How far past its segment phase A extends a lane's last match.
const SPEC_EXTRA: u32 = 256u;

var<workgroup> exits: array<u32, PARSE_SEGMENTS>;
var<workgroup> changed: atomic<u32>;
var<workgroup> flag: u32;
// Lanes whose confirmed exit needs its last match extended (bit k), with the
// match's extension so far and offset; extended by the whole workgroup.
var<workgroup> pending: atomic<u32>;
var<workgroup> pending_now: u32;
var<workgroup> fin_from: array<u32, PARSE_SEGMENTS>;
var<workgroup> fin_off: array<u32, PARSE_SEGMENTS>;
var<workgroup> coop_min: atomic<u32>;
var<workgroup> coop_res: u32;
// Per segment, from phase C: sequence count, last match end (NONE if none),
// and (LZ4 encoded bytes | GLZ literal bytes, GLZ ext count, GLZ wide).
var<workgroup> counts: array<u32, PARSE_SEGMENTS>;
var<workgroup> last_end: array<u32, PARSE_SEGMENTS>;
var<workgroup> stats: array<vec3<u32>, PARSE_SEGMENTS>;

struct Chunk {
    start: u32,       // first input byte
    n: u32,           // bytes
    sbase: u32,       // first scratch word
    match_limit: u32, // matches end at or before this
}

// The greedy step from `p` (with p + MFLIMIT <= n): returns (next position,
// match word or 0 for a literal). Extension stops at `limit` (and always at
// match_limit), so a match reaching `limit` may really be longer.
fn greedy_step(c: Chunk, p: u32, limit: u32) -> vec2<u32> {
    let m = scratch[c.sbase + p] & ~MARK;
    if ((m >> 16u) < MIN_MATCH) {
        return vec2<u32>(p + 1u, 0u);
    }
    let end = extend(c.start, p + (m >> 16u), m & 0xFFFFu, min(limit, c.match_limit));
    return vec2<u32>(end, m);
}

// Phase A: the walk from the segment start, marking visited positions.
// Returns (exit, offset of the last match, 1 if the exit is inexact: that
// match reached the SPEC_EXTRA cap). The exit is the first position >= seg_end,
// or seg_end if the parse ends (no match can start) before that.
fn speculative_walk(c: Chunk, seg_start: u32, seg_end: u32) -> vec3<u32> {
    let cap = seg_end + SPEC_EXTRA;
    var p = seg_start;
    var inexact = vec2<u32>(0u, 0u);
    loop {
        if (p >= seg_end) {
            break;
        }
        if (p + MFLIMIT > c.n) {
            p = seg_end;
            break;
        }
        scratch[c.sbase + p] |= MARK;
        let s = greedy_step(c, p, cap);
        if (s.y != 0u && s.x >= cap && cap < c.match_limit) {
            inexact = vec2<u32>(s.y & 0xFFFFu, 1u);
        }
        p = s.x;
    }
    return vec3<u32>(p, inexact);
}

// Phase B: the walk from entry `e`. Returns (exit, 0), or (_, 1) if it
// reaches a marked position, from where it follows the speculative walk.
fn corrected_walk(c: Chunk, e: u32, seg_end: u32) -> vec2<u32> {
    var p = e;
    loop {
        if (p >= seg_end) {
            break;
        }
        if (p + MFLIMIT > c.n) {
            p = seg_end;
            break;
        }
        if ((scratch[c.sbase + p] & MARK) != 0u) {
            return vec2<u32>(p, 1u);
        }
        p = greedy_step(c, p, c.match_limit).x;
    }
    return vec2<u32>(p, 0u);
}

fn lit_ext(lit_len: u32) -> vec2<u32> {
    // (extension values, wide) contributed by a literal length.
    if (lit_len >= 15u) {
        return vec2<u32>(1u, select(0u, 1u, lit_len - 15u > 0xFFFFu));
    }
    return vec2<u32>(0u, 0u);
}

fn match_ext(match_len: u32) -> vec2<u32> {
    if (match_len - MIN_MATCH >= 15u) {
        return vec2<u32>(1u, select(0u, 1u, match_len - MIN_MATCH - 15u > 0xFFFFu));
    }
    return vec2<u32>(0u, 0u);
}

// extend(start, begin, offset, limit) computed by the whole workgroup (call in
// uniform control flow): each lane checks 16 bytes of every 16 * PARSE_SEGMENTS
// and the first mismatch is the smallest any lane finds.
fn coop_extend(c: Chunk, begin: u32, offset: u32, limit: u32, lid: u32) -> u32 {
    var base = begin;
    loop {
        if (lid == 0u) {
            atomicStore(&coop_min, NONE);
        }
        workgroupBarrier();
        let lo = base + 16u * lid;
        var first = NONE;
        if (lo >= limit) {
            first = limit;
        } else {
            let hi = min(lo + 16u, limit);
            var e = lo;
            loop {
                if (e + 4u > hi) {
                    break;
                }
                let diff = in_word(c.start + e - offset) ^ in_word(c.start + e);
                if (diff != 0u) {
                    first = e + countTrailingZeros(diff) / 8u;
                    break;
                }
                e += 4u;
            }
            if (first == NONE) {
                loop {
                    if (e >= hi) {
                        break;
                    }
                    if (in_byte(c.start + e - offset) != in_byte(c.start + e)) {
                        first = e;
                        break;
                    }
                    e++;
                }
            }
            if (first == NONE && hi == limit) {
                first = limit;
            }
        }
        if (first != NONE) {
            atomicMin(&coop_min, first);
        }
        workgroupBarrier();
        if (lid == 0u) {
            coop_res = atomicLoad(&coop_min);
        }
        let r = workgroupUniformLoad(&coop_res);
        if (r != NONE) {
            return r;
        }
        base += 16u * PARSE_SEGMENTS;
    }
    return NONE;
}

@compute @workgroup_size(PARSE_SEGMENTS)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) k: u32,
) {
    // Depends only on uniform values, so the barriers below are in uniform
    // control flow.
    let chunk = wid.x + wid.y * nwg.x;
    if (chunk >= chunk_count()) {
        return;
    }
    let n = chunk_len(chunk);
    var c = Chunk(chunk * params.chunk_size, n, chunk * params.chunk_size, 0u);
    if (n >= LAST_LITERALS) {
        c.match_limit = n - LAST_LITERALS;
    }
    let len = seg_len(n);
    let seg_start = k * len;
    let seg_end = seg_start + len;

    // ---- A: speculative walks ----
    let spec = speculative_walk(c, seg_start, seg_end);
    var spec_end = spec.x;
    var spec_inexact = spec.z != 0u;
    // An inexact exit is unknown (NONE) until this lane's walk is confirmed.
    exits[k] = select(spec_end, NONE, spec_inexact);
    if (k == 0u) {
        atomicStore(&changed, 0u);
        atomicStore(&pending, 0u);
    }

    // ---- B: fix-up rounds ----
    var entry = NONE; // the entry this lane's exit assumes
    loop {
        workgroupBarrier();
        var e = 0u;
        if (k > 0u) {
            e = exits[k - 1u];
        }
        workgroupBarrier();
        if (e != NONE && e != entry) {
            entry = e;
            let w = corrected_walk(c, e, seg_end);
            if (w.y != 0u && spec_inexact) {
                // Confirmed, but the last match's end is unknown: extend it
                // below with the whole workgroup.
                fin_from[k] = spec_end;
                fin_off[k] = spec.y;
                atomicOr(&pending, 1u << k);
            } else {
                var x = w.x;
                if (w.y != 0u) {
                    x = spec_end;
                }
                if (x != exits[k]) {
                    exits[k] = x;
                    atomicStore(&changed, 1u);
                }
            }
        }
        workgroupBarrier();
        if (k == 0u) {
            pending_now = atomicLoad(&pending);
            atomicStore(&pending, 0u);
        }
        var todo = workgroupUniformLoad(&pending_now);
        loop {
            if (todo == 0u) {
                break;
            }
            let j = firstTrailingBit(todo);
            todo &= todo - 1u;
            let end = coop_extend(c, fin_from[j], fin_off[j], c.match_limit, k);
            if (k == j) {
                spec_end = end;
                spec_inexact = false;
                exits[k] = end;
                atomicStore(&changed, 1u);
            }
        }
        workgroupBarrier();
        if (k == 0u) {
            flag = atomicLoad(&changed);
            atomicStore(&changed, 0u);
        }
        if (workgroupUniformLoad(&flag) == 0u) {
            break;
        }
    }

    // ---- C: write this segment's part of the true path ----
    var count = 0u;
    var acc = vec3<u32>(0u, 0u, 0u);
    var anchor = 0u; // previous match end; unknown before the first match
    var first_p = 0u;
    var first_len = 0u;
    var last = NONE;
    let exit = exits[k];
    var p = entry;
    loop {
        if (p >= seg_end || p + MFLIMIT > n) {
            break;
        }
        var s = greedy_step(c, p, seg_end);
        if (s.y == 0u) {
            p = s.x;
            continue;
        }
        if (s.x >= seg_end) {
            // The match leaving the segment ends where the true walk does.
            s.x = exit;
        }
        let match_len = s.x - p;
        let q = c.sbase + seg_start + 4u * count;
        scratch[q + 2u] = match_len;
        scratch[q + 3u] = s.y & 0xFFFFu;
        if (count == 0u) {
            // Literals patched below, once the anchor is known.
            first_p = p;
            first_len = match_len;
        } else {
            let lit_len = p - anchor;
            scratch[q] = anchor;
            scratch[q + 1u] = lit_len;
            if (CODEC == 0u) {
                acc.x += encoded_len(lit_len, match_len);
            } else {
                acc.x += lit_len;
                let e = lit_ext(lit_len);
                acc.y += e.x;
                acc.z = max(acc.z, e.y);
            }
        }
        if (CODEC == 1u) {
            let e = match_ext(match_len);
            acc.y += e.x;
            acc.z = max(acc.z, e.y);
        }
        count++;
        anchor = s.x;
        last = s.x;
        p = s.x;
    }
    last_end[k] = last;
    workgroupBarrier();

    // The last match end before this segment (0 if none).
    var anchor_in = 0u;
    for (var j = k; j > 0u; j--) {
        if (last_end[j - 1u] != NONE) {
            anchor_in = last_end[j - 1u];
            break;
        }
    }
    if (count > 0u) {
        let q = c.sbase + seg_start;
        let lit_len = first_p - anchor_in;
        scratch[q] = anchor_in;
        scratch[q + 1u] = lit_len;
        if (CODEC == 0u) {
            acc.x += encoded_len(lit_len, first_len);
        } else {
            acc.x += lit_len;
            let e = lit_ext(lit_len);
            acc.y += e.x;
            acc.z = max(acc.z, e.y);
        }
    }
    if (k == PARSE_SEGMENTS - 1u) {
        // The final sequence: literals from the last match end to the end.
        var final_anchor = anchor_in;
        if (last != NONE) {
            final_anchor = last;
        }
        let q = c.sbase + seg_start + 4u * count;
        let lit_len = n - final_anchor;
        scratch[q] = final_anchor;
        scratch[q + 1u] = lit_len;
        scratch[q + 2u] = 0u;
        scratch[q + 3u] = 0u;
        count++;
        if (CODEC == 0u) {
            acc.x += encoded_len(lit_len, 0u);
        } else {
            acc.x += lit_len;
            let e = lit_ext(lit_len);
            acc.y += e.x;
            acc.z = max(acc.z, e.y);
        }
    }
    counts[k] = count;
    stats[k] = acc;
    workgroupBarrier();

    var first = 0u;
    for (var j = 0u; j < k; j++) {
        first += counts[j];
    }
    segs[chunk * PARSE_SEGMENTS + k] = first;
    if (k == PARSE_SEGMENTS - 1u) {
        let total_count = first + count;
        var sum = vec3<u32>(0u, 0u, 0u);
        for (var j = 0u; j < PARSE_SEGMENTS; j++) {
            let s = stats[j];
            sum = vec3<u32>(sum.x + s.x, sum.y + s.y, max(sum.z, s.z));
        }
        var total = sum.x;
        if (CODEC == 1u) {
            total = 8u + pad4(total_count) + pad4(2u * total_count)
                + pad4(sum.y * select(2u, 4u, sum.z != 0u)) + sum.x;
        }
        chunk_info[chunk] = vec4<u32>(total_count, total, sum.y, sum.z);
        if (total >= n) {
            sizes[chunk] = total;
        }
    }
}
