// Encoder kernel 1: match finding, one workgroup per chunk (appended to
// encode_common.wgsl).

override WG_SIZE: u32 = 64u;
override HASH_LOG: u32 = 12u;

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
                        let end = extend(start, p, offset, limit);
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
}
