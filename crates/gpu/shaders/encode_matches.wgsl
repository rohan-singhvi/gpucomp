// Encoder kernel 1: match finding, one workgroup per chunk (appended to
// encode_common.wgsl). Each position's candidate is the latest position of an
// earlier block with the same hash (as cpu::lz4::encode::find_matches with
// depth 1). With DEPTH > 1 it also stores that candidate as the position's
// chain link; the parse computes deep matches (more candidates, along the
// links) only for the positions it visits (encode_deep.wgsl).

override WG_SIZE: u32 = 64u;
override HASH_LOG: u32 = 12u;
override DEPTH: u32 = 1u;

// Bucket value = position + 1; 0 = empty (workgroup memory starts zeroed).
var<workgroup> table: array<atomic<u32>, 1u << HASH_LOG>;

fn hash(word: u32) -> u32 {
    return (word * 2654435761u) >> (32u - HASH_LOG);
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    // Depends only on uniform values, so the barriers below are in uniform
    // control flow.
    let chunk = wid.x + wid.y * nwg.x;
    if (chunk >= chunk_count()) {
        return;
    }
    let start = chunk * params.chunk_size;
    let n = chunk_len(chunk);
    let sbase = chunk * params.chunk_size;
    if (n >= MFLIMIT) {
        let last_start = n - MFLIMIT;
        let match_limit = n - LAST_LITERALS;
        // Software-pipelined: each block's candidate word is loaded before the
        // first barrier and used after the insert, and the next block's input
        // word is loaded before the second barrier, so their latency overlaps
        // the barrier waits.
        var word = 0u;
        if (lid <= last_start) {
            word = in_word(start + lid);
        }
        for (var block = 0u; block <= last_start; block += WG_SIZE) {
            let p = block + lid;
            let in_range = p <= last_start;
            var h = 0u;
            var entry = 0u;
            var offset = 0u;
            var cword = 0u;
            if (in_range) {
                h = hash(word);
                entry = atomicLoad(&table[h]);
                if (DEPTH > 1u) {
                    chain[sbase + p] = entry;
                }
                if (entry != 0u) {
                    offset = p - (entry - 1u);
                    if (offset <= MAX_OFFSET) {
                        cword = in_word(start + p - offset);
                    }
                }
            }
            // All lookups of this block see only earlier blocks' positions.
            workgroupBarrier();
            if (in_range) {
                atomicMax(&table[h], p + 1u);
                var m = 0u;
                if (entry != 0u && offset <= MAX_OFFSET) {
                    let limit = min(p + params.probe_len, match_limit);
                    var end = p;
                    let diff = cword ^ word;
                    if (diff != 0u) {
                        end = p + min(countTrailingZeros(diff) / 8u, limit - p);
                    } else if (p + 4u <= limit) {
                        end = extend(start, p + 4u, offset, limit);
                    } else {
                        end = extend(start, p, offset, limit);
                    }
                    if (end - p >= MIN_MATCH) {
                        m = ((end - p) << 16u) | offset;
                    }
                }
                scratch[sbase + p] = m;
            }
            let next = p + WG_SIZE;
            if (next <= last_start) {
                word = in_word(start + next);
            }
            workgroupBarrier();
        }
    }
}
