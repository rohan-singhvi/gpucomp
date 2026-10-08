// Encoder kernel 2: greedy parse, ONE INVOCATION PER CHUNK (appended to
// encode_common.wgsl). Extends the matches it takes and, for GLZ with
// params.groups > 0, caps matches whose source pattern would reach another
// match's output in the same group of params.groups sequences (dependency
// elimination). No workgroup memory, so occupancy is limited only by
// registers: every lane of every resident SIMD group parses its own chunk.

// 0 = LZ4, 1 = GLZ: selects the size accounting.
override CODEC: u32 = 0u;
const PARSE_WG: u32 = 64u;

// Bytes from `src` on that are free of the group's first `len` match outputs
// (ascending, disjoint): 0 if `src` is inside one, 0xFFFFFFFF if none follows.
fn source_room(gbase: u32, src: u32, len: u32) -> u32 {
    // Binary search for the first output that ends after `src`.
    var lo = 0u;
    var hi = len;
    loop {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) / 2u;
        if (group_buf[gbase + mid].y <= src) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo == len) {
        return 0xFFFFFFFFu;
    }
    let o = group_buf[gbase + lo];
    if (o.x <= src) {
        return 0u;
    }
    return o.x - src;
}

// Sequence k is written over scratch[4k .. 4k + 4] (lit_start, lit_len,
// match_len, offset). That's safe: the k earlier matches each consumed >= 4
// positions, so the current match starts at p >= 4k and the next read is at
// >= p + 4. Records (count, encoded size, ext count, wide) in chunk_info, and
// the size in `sizes` when the block won't shrink the chunk (emit skips it).
fn parse(chunk: u32) {
    let start = chunk * params.chunk_size;
    let n = chunk_len(chunk);
    let sbase = chunk * params.chunk_size;
    let gbase = chunk * MAX_GROUP;
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
            let room = source_room(gbase, p - offset, group_len);
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
        let end = extend(start, p + min(m >> 16u, cap), offset, limit);
        if (params.groups > 0u) {
            group_buf[gbase + group_len] = vec2<u32>(p, end);
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
    chunk_info[chunk] = vec4<u32>(count, total, ext_count, wide);
    if (total >= n) {
        sizes[chunk] = total;
    }
}

@compute @workgroup_size(PARSE_WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let chunk = gid.x + gid.y * nwg.x * PARSE_WG;
    if (chunk >= chunk_count()) {
        return;
    }
    parse(chunk);
}
