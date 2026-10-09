// Encoder kernel 1: match finding, one workgroup per chunk (appended to
// encode_common.wgsl). Each position's first candidate is the latest position
// of an earlier block with the same hash. With DEPTH > 1 it also records that
// candidate as its chain link and follows links to older candidates, keeping
// the longest match (ties: the nearest), as cpu::lz4::encode::find_matches.

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
        for (var block = 0u; block <= last_start; block += WG_SIZE) {
            let p = block + lid;
            let in_range = p <= last_start;
            var h = 0u;
            if (in_range) {
                h = hash(in_word(start + p));
                var entry = atomicLoad(&table[h]);
                let limit = min(p + params.probe_len, match_limit);
                var m = 0u;
                if (DEPTH == 1u) {
                    // Level 1: one candidate (kept separate: the chain loop
                    // costs ~6% even when it runs once).
                    if (entry != 0u) {
                        let offset = p - (entry - 1u);
                        if (offset <= MAX_OFFSET) {
                            let end = extend(start, p, offset, limit);
                            if (end - p >= MIN_MATCH) {
                                m = ((end - p) << 16u) | offset;
                            }
                        }
                    }
                } else {
                    chain[sbase + p] = entry;
                    for (var k = 0u; k < DEPTH; k++) {
                        if (entry == 0u) {
                            break;
                        }
                        let candidate = entry - 1u;
                        let offset = p - candidate;
                        if (offset > MAX_OFFSET) {
                            break;
                        }
                        // Load the next link before extending (it doesn't depend on it).
                        let next = chain[sbase + candidate];
                        // Only a longer match counts: if a best of length L
                        // exists, byte L must match too (skips most extensions).
                        let best = m >> 16u;
                        if (best == 0u || in_byte(start + p + best - offset) == in_byte(start + p + best)) {
                            let len = extend(start, p, offset, limit) - p;
                            if (len >= MIN_MATCH && len > best) {
                                m = (len << 16u) | offset;
                                if (p + len >= limit) {
                                    // Capped: nothing can be longer.
                                    break;
                                }
                            }
                        }
                        entry = next;
                    }
                }
                scratch[sbase + p] = m;
            }
            // All lookups of this block see only earlier blocks' positions;
            // its chain links are visible to later blocks' lookups.
            if (DEPTH > 1u) {
                storageBarrier();
            }
            workgroupBarrier();
            if (in_range) {
                atomicMax(&table[h], p + 1u);
            }
            workgroupBarrier();
        }
    }
}
