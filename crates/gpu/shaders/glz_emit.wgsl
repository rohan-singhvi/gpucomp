// Encoder kernel 3, GLZ (appended to encode_common.wgsl and encode_emit_shared.wgsl). Writes the layout of
// cpu::glz::emit: header words, then per sequence (one invocation each, WG_SIZE
// at a time) its token, offset and extension values at positions given by a
// prefix sum of escape counts, and its literals at a prefix sum of literal
// lengths. Long literal runs are copied by the whole workgroup.

fn put_u16(pos: u32, v: u32) {
    put_byte(pos, v & 0xFFu);
    put_byte(pos + 1u, (v >> 8u) & 0xFFu);
}

fn put_u32(pos: u32, v: u32) {
    put_u16(pos, v & 0xFFFFu);
    put_u16(pos + 2u, v >> 16u);
}

fn put_ext(ext_pos: u32, k: u32, v: u32, wide: u32) {
    if (wide != 0u) {
        put_u32(ext_pos + 4u * k, v);
    } else {
        put_u16(ext_pos + 2u * k, v);
    }
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let chunk = wid.x + wid.y * nwg.x;
    if (chunk >= chunk_count()) {
        return;
    }
    let start = chunk * params.chunk_size;
    let n = chunk_len(chunk);
    let sbase = chunk * params.chunk_size;
    let shape = load_info(chunk, lid);
    load_segs(chunk, lid);
    let count = shape.x;
    let total = shape.y;
    if (total >= n && params.glze == 0u) {
        // Won't shrink the chunk; the parse already reported its size. (GLZ-E
        // transcodes every block, so it needs them all.)
        return;
    }
    let ext_count = shape.z;
    let wide = shape.w;

    let out_base = chunk * params.slot_size;
    let offsets_at = out_base + 8u + pad4(count);
    let ext_at = offsets_at + pad4(2u * count);
    let literals_at = ext_at + pad4(ext_count * select(2u, 4u, wide != 0u));
    if (lid == 0u) {
        put_u32(out_base, count | (wide << 31u));
        put_u32(out_base + 4u, ext_count);
    }

    var ext_base = 0u;
    var lit_base = 0u;
    for (var block = 0u; block < count; block += WG_SIZE) {
        let i = block + lid;
        let in_range = i < count;
        var lit_start = 0u;
        var lit_len = 0u;
        var match_code = 0u;
        var offset = 0u;
        var esc = 0u;
        var q = 0u;
        if (in_range) {
            q = seq_addr(sbase, n, i);
            lit_start = scratch[q];
            lit_len = scratch[q + 1u];
            let match_len = scratch[q + 2u];
            offset = scratch[q + 3u];
            if (match_len > 0u) {
                match_code = match_len - MIN_MATCH;
            }
            esc = select(0u, 1u, lit_len >= 15u) + select(0u, 1u, match_code >= 15u);
        }
        qaddr[lid] = q;
        // Inclusive scan of (escapes, literal bytes).
        scan2[lid] = vec2<u32>(esc, lit_len);
        workgroupBarrier();
        for (var d = 1u; d < WG_SIZE; d <<= 1u) {
            var v = vec2<u32>(0u, 0u);
            if (lid >= d) {
                v = scan2[lid - d];
            }
            workgroupBarrier();
            scan2[lid] += v;
            workgroupBarrier();
        }
        var before = vec2<u32>(0u, 0u);
        if (lid > 0u) {
            before = scan2[lid - 1u];
        }
        let totals = scan2[WG_SIZE - 1u];

        if (in_range) {
            let lit_nibble = min(lit_len, 15u);
            let match_nibble = min(match_code, 15u);
            put_byte(out_base + 8u + i, (lit_nibble << 4u) | match_nibble);
            put_u16(offsets_at + 2u * i, offset);
            var k = ext_base + before.x;
            if (lit_nibble == 15u) {
                put_ext(ext_at, k, lit_len - 15u, wide);
                k++;
            }
            if (match_nibble == 15u) {
                put_ext(ext_at, k, match_code - 15u, wide);
            }
            if (lit_len < LONG_LITERALS) {
                let dst = literals_at + lit_base + before.y;
                for (var b = 0u; b < lit_len; b++) {
                    put_byte(dst + b, in_byte(start + lit_start + b));
                }
            }
        }
        // Long literal runs of this block, by the whole workgroup.
        let in_block = min(WG_SIZE, count - block);
        for (var j = 0u; j < in_block; j++) {
            let q = qaddr[j];
            let len = scratch[q + 1u];
            if (len >= LONG_LITERALS) {
                var lp = lit_base;
                if (j > 0u) {
                    lp += scan2[j - 1u].y;
                }
                copy_literals(literals_at + lp, start + scratch[q], len, lid);
            }
        }
        ext_base += totals.x;
        lit_base += totals.y;
        workgroupBarrier();
    }
    if (lid == 0u) {
        sizes[chunk] = total;
    }
}
