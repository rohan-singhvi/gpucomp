// LZ4 emit (appended to encode_common.wgsl): a workgroup prefix sum over the
// sequences' encoded sizes gives each its output offset; each sequence's owner
// writes its header bytes and short literal runs, and the whole workgroup
// copies literal runs of at least LONG_LITERALS bytes.

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

    find_matches(start, n, sbase, lid);
    if (lid == 0u) {
        parse(start, n, sbase);
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
