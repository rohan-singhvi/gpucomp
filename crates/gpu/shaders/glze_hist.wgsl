// M9e GLZ-E encode, kernel 1 of 3 (appended to encode_common.wgsl): one
// workgroup per chunk. Histograms of the chunk's four streams (read a word at
// a time from the GLZ block copied to the match scratch; the offset words
// feed both offset streams), then each stream's used symbols in (count,
// symbol) order by a bitonic sort of count << 8 | symbol keys, written to the
// chunk's build data for glze_build.

const HWG: u32 = 64u;

var<workgroup> hist: array<atomic<u32>, 1024>;
var<workgroup> keys: array<u32, 1024>;
var<workgroup> used: array<atomic<u32>, 4>;

// Counts the first min(valid, 4) bytes of `word` into stream `s`.
fn count_bytes(word: u32, valid: u32, s: u32) {
    for (var b = 0u; b < min(valid, 4u); b++) {
        atomicAdd(&hist[s * 256u + ((word >> (8u * b)) & 0xFFu)], 1u);
    }
}

@compute @workgroup_size(HWG)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let chunk = wid.x + wid.y * nwg.x;
    if (chunk >= chunk_count()) {
        return;
    }
    let info = chunk_info[chunk];
    let gword = chunk * params.slot_size / 4u;
    for (var e = lid; e < 1024u; e += HWG) {
        atomicStore(&hist[e], 0u);
    }
    if (lid < 4u) {
        atomicStore(&used[lid], 0u);
    }
    workgroupBarrier();
    let count = info.x;
    let tokens = glze_stream(info, 0u);
    let lits = glze_stream(info, 3u);
    // Tokens and literals start on words (GLZ arrays are 4-aligned).
    for (var w = lid; w < (count + 3u) / 4u; w += HWG) {
        count_bytes(scratch[gword + tokens.x / 4u + w], count - 4u * w, 0u);
    }
    let offsets = glze_stream(info, 1u).x / 4u;
    for (var w = lid; w < (count + 1u) / 2u; w += HWG) {
        let word = scratch[gword + offsets + w];
        atomicAdd(&hist[256u + (word & 0xFFu)], 1u);
        atomicAdd(&hist[512u + ((word >> 8u) & 0xFFu)], 1u);
        if (2u * w + 1u < count) {
            atomicAdd(&hist[256u + ((word >> 16u) & 0xFFu)], 1u);
            atomicAdd(&hist[512u + (word >> 24u)], 1u);
        }
    }
    for (var w = lid; w < (lits.z + 3u) / 4u; w += HWG) {
        count_bytes(scratch[gword + lits.x / 4u + w], lits.z - 4u * w, 3u);
    }
    workgroupBarrier();
    for (var e = lid; e < 1024u; e += HWG) {
        let c = atomicLoad(&hist[e]);
        keys[e] = (c << 8u) | (e & 255u);
        if (c > 0u) {
            atomicAdd(&used[e / 256u], 1u);
        }
    }
    workgroupBarrier();
    // Bitonic sort of each stream's 256 keys (ascending): counts below
    // 2^24, so keys are unique and order by (count, symbol).
    for (var k = 2u; k <= 256u; k <<= 1u) {
        for (var j = k >> 1u; j > 0u; j >>= 1u) {
            for (var i = lid; i < 1024u; i += HWG) {
                let l = i ^ j;
                if (l > i) {
                    let a = keys[i];
                    let b = keys[l];
                    if ((a > b) == (((i & 255u) & k) == 0u)) {
                        keys[i] = b;
                        keys[l] = a;
                    }
                }
            }
            workgroupBarrier();
        }
    }
    let bbase = chunk * 4u * GLZE_BUILD_WORDS;
    for (var e = lid; e < 1024u; e += HWG) {
        let s = e / 256u;
        let b = bbase + s * GLZE_BUILD_WORDS;
        bset(b + B_COUNTS + (e & 255u), atomicLoad(&hist[e]));
        // Zero-count keys sort first, so the used symbols are the stream's
        // last `used` keys, in rank order.
        let unused = 256u - atomicLoad(&used[s]);
        let i = e & 255u;
        if (i >= unused) {
            bset(b + B_SORTED + i - unused, keys[e] & 255u);
        }
    }
    if (lid < 4u) {
        bset(bbase + lid * GLZE_BUILD_WORDS + B_USED, atomicLoad(&used[lid]));
    }
}
