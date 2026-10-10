// M9e GLZ-E encode, kernel 2 of 3 (appended to encode_common.wgsl): ONE
// INVOCATION PER (CHUNK, STREAM). The Huffman tree build is serial, so it runs
// in thousands of independent invocations rather than on one lane of a
// chunk's workgroup (which cost ~30 ms on Silesia; DECISIONS.md). From the
// histogram and rank order of glze_hist: code lengths exactly as
// cpu::huffman::code_lengths (two-queue Huffman, the leaf first on equal
// weights; JPEG Annex K.3 folds depths over 11; lengths handed out
// longest-first in rank order), then canonical, bit-reversed codes.

const BWG: u32 = 64u;
const MAX_LEN: u32 = 11u;
const MAX_DEPTH: u32 = 32u;

@compute @workgroup_size(BWG)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let t = gid.x + gid.y * nwg.x * BWG;
    if (t >= 4u * chunk_count()) {
        return;
    }
    let b = t * GLZE_BUILD_WORDS;
    let m = bget(b + B_USED);
    if (m < 2u) {
        // Raw (no symbols) or RLE: no code.
        return;
    }
    // Two-queue Huffman over the sorted leaves and the created nodes.
    var li = 0u;
    var ni = 0u;
    for (var k = 0u; k + 1u < m; k++) {
        var sum = 0u;
        for (var pick = 0u; pick < 2u; pick++) {
            var leaf_weight = 0xFFFFFFFFu;
            if (li < m) {
                leaf_weight = bget(b + B_COUNTS + bget(b + B_SORTED + li));
            }
            if (li < m && (ni >= k || leaf_weight <= bget(b + B_WEIGHT + ni))) {
                sum += leaf_weight;
                bset(b + B_LEAF_PARENT + li, k);
                li++;
            } else {
                sum += bget(b + B_WEIGHT + ni);
                bset(b + B_NODE_PARENT + ni, k);
                ni++;
            }
        }
        bset(b + B_WEIGHT + k, sum);
    }
    // Parents are created after their children: depths go root-first.
    bset(b + B_DEPTH + m - 2u, 0u);
    for (var k = m - 2u; k > 0u; k--) {
        bset(b + B_DEPTH + k - 1u, bget(b + B_DEPTH + bget(b + B_NODE_PARENT + k - 1u)) + 1u);
    }
    for (var l = 0u; l <= MAX_DEPTH; l++) {
        bset(b + B_BL + l, 0u);
    }
    for (var i = 0u; i < m; i++) {
        let len = bget(b + B_DEPTH + bget(b + B_LEAF_PARENT + i)) + 1u;
        bset(b + B_BL + len, bget(b + B_BL + len) + 1u);
    }
    for (var i = MAX_DEPTH; i > MAX_LEN; i--) {
        loop {
            if (bget(b + B_BL + i) == 0u) {
                break;
            }
            var j = i - 2u;
            loop {
                if (bget(b + B_BL + j) != 0u) {
                    break;
                }
                j--;
            }
            bset(b + B_BL + i, bget(b + B_BL + i) - 2u);
            bset(b + B_BL + i - 1u, bget(b + B_BL + i - 1u) + 1u);
            bset(b + B_BL + j + 1u, bget(b + B_BL + j + 1u) + 2u);
            bset(b + B_BL + j, bget(b + B_BL + j) - 1u);
        }
    }
    for (var s = 0u; s < 256u; s++) {
        bset(b + B_LENGTHS + s, 0u);
    }
    var idx = 0u;
    for (var len = MAX_LEN; len > 0u; len--) {
        let n = bget(b + B_BL + len);
        for (var c = 0u; c < n; c++) {
            bset(b + B_LENGTHS + bget(b + B_SORTED + idx), len);
            idx++;
        }
    }
    // Canonical codes (shorter first, then by symbol), bit-reversed. bl[0]
    // is 0 and bl[1..=11] are the final per-length counts.
    var code = 0u;
    for (var len = 1u; len <= MAX_LEN; len++) {
        code = (code + bget(b + B_BL + len - 1u)) << 1u;
        bset(b + B_NEXT + len, code);
    }
    for (var s = 0u; s < 256u; s++) {
        let len = bget(b + B_LENGTHS + s);
        if (len > 0u) {
            let next = bget(b + B_NEXT + len);
            bset(b + B_CODES + s, reverseBits(next) >> (32u - len));
            bset(b + B_NEXT + len, next + 1u);
        }
    }
}
