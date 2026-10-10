// M9e GLZ-E encode, kernel 3 of 3 (appended to encode_common.wgsl): one
// workgroup per chunk, a group of 32 invocations per stream, all four streams
// at once. glz_emit.wgsl wrote every chunk's GLZ block into its output slot,
// the host copied the slots into the (by then unused) match scratch,
// glze_hist counted and ranked each stream's symbols and glze_build made its
// code. This kernel writes each block back into its slot as GLZ-E, exactly
// as cpu::glze::transcode and cpu::huffman::encode_stream, and sets
// sizes[chunk] to the GLZ-E block size:
//   1. each group counts its stream's bits per lane (if it may be Huffman);
//   2. invocation 0 picks each stream's mode (RLE if one symbol, raw if
//      none, else Huffman if strictly smaller than raw) and lays out the block;
//   3. each group writes its stream; every invocation writes whole words it
//      owns.

const WG: u32 = 128u;
const LANES: u32 = 32u; // per stream: the most lanes a stream has
const LANE_SYMBOLS: u32 = 512u;

const RAW: u32 = 0u;
const RLE: u32 = 1u;
const HUFFMAN: u32 = 2u;

var<workgroup> lengths: array<u32, 1024>; // [stream][symbol]
var<workgroup> codes: array<u32, 1024>;
var<workgroup> lane_words: array<u32, 128>; // [stream][lane]
var<workgroup> lane_first: array<u32, 128>; // lane's first output byte (slot-relative)
var<workgroup> mode: array<u32, 4>;
var<workgroup> stream_at: array<u32, 4>; // stream's output byte (slot-relative)
var<workgroup> ext_out: u32;

// Symbol byte at GLZ byte address `addr` (absolute, in the match scratch),
// through a one-word cache: consecutive symbols share a load.
fn cached_byte(addr: u32, wi: ptr<function, u32>, word: ptr<function, u32>) -> u32 {
    if ((addr >> 2u) != *wi) {
        *wi = addr >> 2u;
        *word = scratch[addr >> 2u];
    }
    return (*word >> ((addr & 3u) * 8u)) & 0xFFu;
}

fn put_word(obase: u32, at: u32, value: u32) {
    atomicStore(&output[(obase + at) >> 2u], value);
}

fn lanes(n: u32) -> u32 {
    return clamp((n + LANE_SYMBOLS - 1u) / LANE_SYMBOLS, 1u, LANES);
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
    let bbase = chunk * 4u * GLZE_BUILD_WORDS;
    let g = lid / LANES; // this invocation's stream
    let k = lid % LANES; // its lane
    let d = glze_stream(info, g);
    let n = d.z;
    let ln = lanes(n);
    let used = bget(bbase + g * GLZE_BUILD_WORDS + B_USED);

    // 1. Codes, and each lane's bits.
    for (var e = lid; e < 1024u; e += WG) {
        let b = bbase + (e / 256u) * GLZE_BUILD_WORDS;
        lengths[e] = bget(b + B_LENGTHS + (e & 255u));
        codes[e] = bget(b + B_CODES + (e & 255u));
    }
    workgroupBarrier();
    if (used >= 2u && k < ln) {
        var bits = 0u;
        var wi = 0xFFFFFFFFu;
        var word = 0u;
        for (var i = k * n / ln; i < (k + 1u) * n / ln; i++) {
            bits += lengths[g * 256u + cached_byte(gbase + d.x + i * d.y, &wi, &word)];
        }
        lane_words[g * LANES + k] = (bits + 31u) / 32u;
    }
    workgroupBarrier();

    // 2. Modes and layout: header, the three sequence streams, the raw
    // extension words, the literal stream.
    let ext_at = glze_stream(info, 1u).x + pad4(2u * info.x);
    let ext_len = glze_stream(info, 3u).x - ext_at;
    if (lid == 0u) {
        var pos = 12u;
        for (var s = 0u; s < 4u; s++) {
            if (s == 3u) {
                ext_out = pos;
                pos += ext_len;
            }
            let ns = glze_stream(info, s).z;
            let m = bget(bbase + s * GLZE_BUILD_WORDS + B_USED);
            let raw_size = 4u + pad4(ns);
            stream_at[s] = pos;
            if (m == 1u) {
                mode[s] = RLE;
                pos += 4u;
            } else {
                mode[s] = RAW;
                var size = raw_size;
                if (m >= 2u) {
                    let lns = lanes(ns);
                    let data = pos + 132u + pad4(2u * lns);
                    var total = 0u;
                    for (var j = 0u; j < lns; j++) {
                        lane_first[s * LANES + j] = data + 4u * total;
                        total += lane_words[s * LANES + j];
                    }
                    let huffman_size = data + 4u * total - pos;
                    if (huffman_size < raw_size) {
                        mode[s] = HUFFMAN;
                        size = huffman_size;
                    }
                }
                pos += size;
            }
        }
        put_word(obase, 0u, scratch[gbase / 4u]);
        put_word(obase, 4u, info.z);
        put_word(obase, 8u, glze_stream(info, 3u).z);
        sizes[chunk] = pos;
    }
    workgroupBarrier();

    // 3. Everyone copies the extension words; each group writes its stream.
    for (var w = lid; w < ext_len / 4u; w += WG) {
        put_word(obase, ext_out + 4u * w, scratch[(gbase + ext_at) / 4u + w]);
    }
    let pos = stream_at[g];
    let sm = mode[g];
    if (sm == RLE) {
        if (k == 0u) {
            put_word(obase, pos, RLE | (bget(bbase + g * GLZE_BUILD_WORDS + B_SORTED) << 8u));
        }
    } else if (sm == RAW) {
        if (k == 0u) {
            put_word(obase, pos, RAW);
        }
        for (var w = k; w < (n + 3u) / 4u; w += LANES) {
            // Zero past the last symbol (raw streams are zero-padded).
            let tail = min(n - 4u * w, 4u);
            let keep = select(0xFFFFFFFFu, (1u << (8u * tail)) - 1u, tail < 4u);
            var value = 0u;
            if (d.y == 1u) {
                // Tokens and literals start on words: copy.
                value = scratch[(gbase + d.x) / 4u + w];
            } else {
                // Offset low or high bytes: two offsets per source word.
                let src = gbase + d.x - (d.x & 1u);
                let a = scratch[src / 4u + 2u * w];
                let b = scratch[src / 4u + 2u * w + 1u];
                let sh = 8u * (d.x & 1u);
                value = ((a >> sh) & 0xFFu) | (((a >> (16u + sh)) & 0xFFu) << 8u)
                    | (((b >> sh) & 0xFFu) << 16u) | (((b >> (16u + sh)) & 0xFFu) << 24u);
            }
            put_word(obase, pos + 4u + 4u * w, value & keep);
        }
    } else {
        // Huffman: mode word, 32 words of nibble lengths, lane sizes, lanes.
        if (k == 0u) {
            put_word(obase, pos, HUFFMAN);
        }
        var table_word = 0u;
        for (var j = 0u; j < 8u; j++) {
            table_word |= lengths[g * 256u + 8u * k + j] << (4u * j);
        }
        put_word(obase, pos + 4u + 4u * k, table_word);
        if (k < (ln + 1u) / 2u) {
            var sizes_word = lane_words[g * LANES + 2u * k];
            if (2u * k + 1u < ln) {
                sizes_word |= lane_words[g * LANES + 2u * k + 1u] << 16u;
            }
            put_word(obase, pos + 132u + 4u * k, sizes_word);
        }
        if (k < ln) {
            var at = lane_first[g * LANES + k];
            var acc = 0u;
            var fill = 0u;
            var wi = 0xFFFFFFFFu;
            var word = 0u;
            for (var i = k * n / ln; i < (k + 1u) * n / ln; i++) {
                let s = g * 256u + cached_byte(gbase + d.x + i * d.y, &wi, &word);
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
}
