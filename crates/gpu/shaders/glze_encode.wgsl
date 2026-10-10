// M9e: GLZ-E encode, one workgroup per chunk (appended to encode_common.wgsl).
// glz_emit.wgsl wrote every chunk's GLZ block into its output slot, and the
// host copied the slots into the (by then unused) match scratch. This kernel transcodes each block back
// into its slot exactly as cpu::glze::transcode and cpu::huffman::encode_stream
// do, and sets sizes[chunk] to the GLZ-E block size.
//
// Per stream (tokens, offset low bytes, offset high bytes, literals):
//   1. a histogram (workgroup atomics), and each used symbol's rank in
//      (count, symbol) order;
//   2. invocation 0: RLE if one symbol; otherwise code lengths (two-queue
//      Huffman, leaf first on ties; JPEG Annex K.3 folds depths over 11;
//      lengths handed out longest-first in rank order) and canonical codes;
//   3. each lane's bit count; Huffman if strictly smaller than raw;
//   4. output: every invocation writes whole words it owns.

const WG: u32 = 32u; // = the most lanes a stream has
const MAX_LEN: u32 = 11u;
const MAX_DEPTH: u32 = 32u;
const LANE_SYMBOLS: u32 = 512u;
const WIDE_BIT: u32 = 0x80000000u;

const RAW: u32 = 0u;
const RLE: u32 = 1u;
const HUFFMAN: u32 = 2u;

var<workgroup> counts: array<atomic<u32>, 256>;
var<workgroup> cnt: array<u32, 256>;
// Used symbols in (count, symbol) order.
var<workgroup> sorted: array<u32, 256>;
var<workgroup> lengths: array<u32, 256>;
var<workgroup> codes: array<u32, 256>;
// Two-queue Huffman: internal node weights and parents, leaves' parents.
var<workgroup> weight: array<u32, 256>;
var<workgroup> node_parent: array<u32, 256>;
var<workgroup> leaf_parent: array<u32, 256>;
var<workgroup> depth: array<u32, 256>;
var<workgroup> bl: array<u32, 33>; // MAX_DEPTH + 1
var<workgroup> next_code: array<u32, 12>;
var<workgroup> lane_words: array<u32, WG>;
var<workgroup> lane_first: array<u32, WG>; // lane's first output word (slot-relative)
// Broadcast slots, written by invocation 0.
var<workgroup> w_mode: u32;
var<workgroup> w_pos: u32; // output position (bytes, slot-relative)

// Byte `at` of the chunk's GLZ block.
fn gbyte(gbase: u32, at: u32) -> u32 {
    let b = gbase + at;
    return (scratch[b >> 2u] >> ((b & 3u) * 8u)) & 0xFFu;
}

fn put_word(obase: u32, at: u32, value: u32) {
    atomicStore(&output[(obase + at) >> 2u], value);
}

fn lanes(n: u32) -> u32 {
    return clamp((n + LANE_SYMBOLS - 1u) / LANE_SYMBOLS, 1u, WG);
}

// Invocation 0: code lengths for the `m` used symbols in `sorted`
// (cpu::huffman::code_lengths), then canonical, bit-reversed codes.
fn build_code(m: u32) {
    // Two-queue Huffman over sorted leaves and created nodes.
    var li = 0u;
    var ni = 0u;
    for (var k = 0u; k + 1u < m; k++) {
        var sum = 0u;
        for (var pick = 0u; pick < 2u; pick++) {
            if (li < m && (ni >= k || cnt[sorted[li]] <= weight[ni])) {
                sum += cnt[sorted[li]];
                leaf_parent[li] = k;
                li++;
            } else {
                sum += weight[ni];
                node_parent[ni] = k;
                ni++;
            }
        }
        weight[k] = sum;
    }
    // Parents are created after their children: depths go root-first.
    depth[m - 2u] = 0u;
    for (var k = m - 2u; k > 0u; k--) {
        depth[k - 1u] = depth[node_parent[k - 1u]] + 1u;
    }
    for (var l = 0u; l <= MAX_DEPTH; l++) {
        bl[l] = 0u;
    }
    for (var i = 0u; i < m; i++) {
        bl[depth[leaf_parent[i]] + 1u] += 1u;
    }
    for (var i = MAX_DEPTH; i > MAX_LEN; i--) {
        loop {
            if (bl[i] == 0u) {
                break;
            }
            var j = i - 2u;
            loop {
                if (bl[j] != 0u) {
                    break;
                }
                j--;
            }
            bl[i] -= 2u;
            bl[i - 1u] += 1u;
            bl[j + 1u] += 2u;
            bl[j] -= 1u;
        }
    }
    for (var s = 0u; s < 256u; s++) {
        lengths[s] = 0u;
    }
    var idx = 0u;
    for (var len = MAX_LEN; len > 0u; len--) {
        for (var c = 0u; c < bl[len]; c++) {
            lengths[sorted[idx]] = len;
            idx++;
        }
    }
    // Canonical codes (shorter first, then by symbol), bit-reversed.
    var code = 0u;
    for (var len = 1u; len <= MAX_LEN; len++) {
        code = (code + bl[len - 1u]) << 1u;
        next_code[len] = code;
    }
    // bl[0] is 0 and bl[1..=11] are the final per-length counts.
    for (var s = 0u; s < 256u; s++) {
        let len = lengths[s];
        if (len > 0u) {
            codes[s] = reverseBits(next_code[len]) >> (32u - len);
            next_code[len] += 1u;
        }
    }
}

// Encodes the n symbols at GLZ byte src + i * stride at output position
// w_pos, and advances w_pos. Call in uniform control flow.
fn stream(gbase: u32, obase: u32, src: u32, stride: u32, n: u32, lid: u32) {
    let pos = workgroupUniformLoad(&w_pos);
    for (var s = lid; s < 256u; s += WG) {
        atomicStore(&counts[s], 0u);
    }
    workgroupBarrier();
    for (var i = lid; i < n; i += WG) {
        atomicAdd(&counts[gbyte(gbase, src + i * stride)], 1u);
    }
    workgroupBarrier();
    for (var s = lid; s < 256u; s += WG) {
        cnt[s] = atomicLoad(&counts[s]);
    }
    workgroupBarrier();
    for (var s = lid; s < 256u; s += WG) {
        let c = cnt[s];
        if (c > 0u) {
            var rank = 0u;
            for (var t = 0u; t < 256u; t++) {
                let ct = cnt[t];
                if (ct > 0u && (ct < c || (ct == c && t < s))) {
                    rank++;
                }
            }
            sorted[rank] = s;
        }
    }
    workgroupBarrier();
    let raw_size = 4u + ((n + 3u) & ~3u);
    if (lid == 0u) {
        var m = 0u;
        for (var s = 0u; s < 256u; s++) {
            m += select(0u, 1u, cnt[s] > 0u);
        }
        if (m == 1u) {
            w_mode = RLE;
            put_word(obase, pos, RLE | (sorted[0] << 8u));
            w_pos = pos + 4u;
        } else if (m == 0u) {
            w_mode = RAW;
        } else {
            build_code(m);
            w_mode = HUFFMAN;
        }
    }
    var mode = workgroupUniformLoad(&w_mode);
    if (mode == RLE) {
        return;
    }
    let ln = lanes(n);
    if (mode == HUFFMAN) {
        if (lid < ln) {
            var bits = 0u;
            for (var i = lid * n / ln; i < (lid + 1u) * n / ln; i++) {
                bits += lengths[gbyte(gbase, src + i * stride)];
            }
            lane_words[lid] = (bits + 31u) / 32u;
        }
        workgroupBarrier();
        if (lid == 0u) {
            let data = pos + 132u + ((2u * ln + 3u) & ~3u);
            var total = 0u;
            for (var k = 0u; k < ln; k++) {
                lane_first[k] = data + 4u * total;
                total += lane_words[k];
            }
            let huffman_size = data + 4u * total - pos;
            if (huffman_size < raw_size) {
                w_pos = pos + huffman_size;
            } else {
                w_mode = RAW;
            }
        }
        mode = workgroupUniformLoad(&w_mode);
    }
    if (mode == RAW) {
        if (lid == 0u) {
            put_word(obase, pos, RAW);
            w_pos = pos + raw_size;
        }
        for (var w = lid; w < (n + 3u) / 4u; w += WG) {
            var value = 0u;
            for (var b = 0u; b < 4u; b++) {
                let i = 4u * w + b;
                if (i < n) {
                    value |= gbyte(gbase, src + i * stride) << (8u * b);
                }
            }
            put_word(obase, pos + 4u + 4u * w, value);
        }
        return;
    }
    // Huffman: mode word, 32 words of nibble lengths, lane sizes, lanes.
    if (lid == 0u) {
        put_word(obase, pos, HUFFMAN);
    }
    for (var w = lid; w < 32u; w += WG) {
        var value = 0u;
        for (var j = 0u; j < 8u; j++) {
            value |= lengths[8u * w + j] << (4u * j);
        }
        put_word(obase, pos + 4u + 4u * w, value);
    }
    for (var w = lid; w < (ln + 1u) / 2u; w += WG) {
        var value = lane_words[2u * w];
        if (2u * w + 1u < ln) {
            value |= lane_words[2u * w + 1u] << 16u;
        }
        put_word(obase, pos + 132u + 4u * w, value);
    }
    if (lid < ln) {
        var at = lane_first[lid];
        var acc = 0u;
        var fill = 0u;
        for (var i = lid * n / ln; i < (lid + 1u) * n / ln; i++) {
            let s = gbyte(gbase, src + i * stride);
            let len = lengths[s];
            let code = codes[s];
            acc |= code << fill;
            if (fill + len >= 32u) {
                put_word(obase, at, acc);
                at += 4u;
                acc = code >> (32u - fill);
                fill = fill + len - 32u;
            } else {
                fill += len;
            }
        }
        if (fill > 0u) {
            put_word(obase, at, acc);
        }
    }
}

@compute @workgroup_size(WG)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let chunk = wid.x + wid.y * nwg.x;
    if (chunk >= chunk_count()) {
        return;
    }
    let gbase = chunk * params.slot_size;
    let obase = gbase;
    let info = chunk_info[chunk];
    let count = info.x;
    let glz_size = info.y;
    let ext = info.z;
    let wide = info.w;
    let offsets_at = 8u + pad4(count);
    let ext_at = offsets_at + pad4(2u * count);
    let literals_at = ext_at + pad4(ext * select(2u, 4u, wide != 0u));
    if (lid == 0u) {
        put_word(obase, 0u, scratch[gbase / 4u]);
        put_word(obase, 4u, ext);
        put_word(obase, 8u, glz_size - literals_at);
        w_pos = 12u;
    }
    stream(gbase, obase, 8u, 1u, count, lid);
    stream(gbase, obase, offsets_at, 2u, count, lid);
    stream(gbase, obase, offsets_at + 1u, 2u, count, lid);
    // The extension words, as they are.
    let at = workgroupUniformLoad(&w_pos);
    for (var w = lid; w < (literals_at - ext_at) / 4u; w += WG) {
        put_word(obase, at + 4u * w, scratch[(gbase + ext_at) / 4u + w]);
    }
    workgroupBarrier();
    if (lid == 0u) {
        w_pos = at + (literals_at - ext_at);
    }
    stream(gbase, obase, literals_at, 1u, glz_size - literals_at, lid);
    let end = workgroupUniformLoad(&w_pos);
    if (lid == 0u) {
        sizes[chunk] = end;
    }
}
