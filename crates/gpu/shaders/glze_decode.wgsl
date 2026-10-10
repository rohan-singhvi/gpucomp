// M9e: GLZ-E entropy decode, one workgroup per chunk. Rebuilds the chunk's
// GLZ block (cpu::glze::to_glz) in `image`, checking in FORMAT.md's order and
// stopping at the first error; glz_decode.wgsl then decodes the images (the
// host keeps this kernel's status when it reports an error). Stored chunks
// are copied as they are.
//
// Per stream, invocation 0 reads the mode word and makes every decision
// (sizes, table validity, lane layout) and broadcasts it, so all barriers are
// in uniform control flow. Raw streams are copied and RLE streams filled in
// parallel. For Huffman streams the workgroup fills a 2048-entry decode table,
// then invocation k decodes lane k. Output bytes are merged into the zeroed
// image with atomicOr.

struct GlzeDesc {
    src_offset: u32,   // payload start in src (bytes, 4-aligned)
    comp_size: u32,    // payload bytes, STORED_BIT for stored chunks
    image_offset: u32, // where its GLZ block goes in image (bytes, 4-aligned)
    image_size: u32,   // GLZ block bytes (host-computed from the header)
    uncomp_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> descs: array<GlzeDesc>;
@group(0) @binding(2) var<storage, read_write> image: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> status: array<u32>;

const WG: u32 = 32u; // = the most lanes a stream has
const STORED_BIT: u32 = 0x80000000u;
const WIDE_BIT: u32 = 0x80000000u;
const MAX_LEN: u32 = 11u;
const TABLE: u32 = 2048u; // 1 << MAX_LEN
const LANE_SYMBOLS: u32 = 512u;
const NONE: u32 = 0xFFFFFFFFu;

const OK: u32 = 0u;
const TRUNCATED: u32 = 1u;
const SIZE_MISMATCH: u32 = 5u;
const BAD_SEQUENCE: u32 = 6u;
const STREAM_TRUNCATED: u32 = 7u;
const STREAM_BAD_MODE: u32 = 8u;
const STREAM_BAD_TABLE: u32 = 9u;
const STREAM_LANE_OVERRUN: u32 = 10u;

const RAW: u32 = 0u;
const RLE: u32 = 1u;
const HUFFMAN: u32 = 2u;

var<workgroup> lengths: array<u32, 256>;
var<workgroup> codes: array<u32, 256>;
var<workgroup> table: array<u32, TABLE>;
var<workgroup> bl: array<u32, 12>;
var<workgroup> next_code: array<u32, 12>;
var<workgroup> lane_words: array<u32, WG>;
var<workgroup> lane_at: array<u32, WG>;
var<workgroup> overrun: atomic<u32>;
// Broadcast slots, written by invocation 0.
var<workgroup> w_err: u32;
var<workgroup> w_mode: u32;
var<workgroup> w_next: u32;
var<workgroup> w_flag: u32;

struct Chunk {
    p: u32,    // payload start in src (bytes)
    comp: u32, // payload bytes
    i: u32,    // image start (bytes)
}

fn pad4(n: u32) -> u32 {
    return (n + 3u) & ~3u;
}

fn sword(c: Chunk, at: u32) -> u32 {
    return src[(c.p + at) >> 2u];
}

fn sbyte(c: Chunk, at: u32) -> u32 {
    let b = c.p + at;
    return (src[b >> 2u] >> ((b & 3u) * 8u)) & 0xFFu;
}

fn put(c: Chunk, at: u32, value: u32) {
    let b = c.i + at;
    atomicOr(&image[b >> 2u], value << ((b & 3u) * 8u));
}

fn lanes(n: u32) -> u32 {
    return clamp((n + LANE_SYMBOLS - 1u) / LANE_SYMBOLS, 1u, WG);
}

// Decodes the n-symbol stream at payload byte `at`; symbol i goes to image
// byte dest + i * stride. Leaves w_err (OK or a status) and, if OK, w_next
// (the position after the stream). Call in uniform control flow.
fn stream(c: Chunk, at: u32, n: u32, dest: u32, stride: u32, lid: u32) {
    // Everyone has read the previous broadcast before invocation 0 rewrites it.
    workgroupBarrier();
    if (lid == 0u) {
        w_err = OK;
        w_mode = NONE;
        if (at + 4u > c.comp) {
            w_err = STREAM_TRUNCATED;
        } else {
            let head = sword(c, at);
            let mode = head & 0xFFu;
            if (mode == RAW && head >> 8u == 0u) {
                w_mode = RAW;
                let end = at + 4u + pad4(n);
                if (end > c.comp) {
                    w_err = STREAM_TRUNCATED;
                }
                w_next = end;
            } else if (mode == RLE && head >> 16u == 0u) {
                w_mode = RLE;
                w_next = at + 4u;
            } else if (mode == HUFFMAN && head >> 8u == 0u) {
                w_mode = HUFFMAN;
                if (at + 132u > c.comp) {
                    w_err = STREAM_TRUNCATED;
                }
            } else {
                w_err = STREAM_BAD_MODE;
            }
        }
    }
    if (workgroupUniformLoad(&w_err) != OK) {
        return;
    }
    let mode = workgroupUniformLoad(&w_mode);
    if (mode == RAW) {
        for (var i = lid; i < n; i += WG) {
            put(c, dest + i * stride, sbyte(c, at + 4u + i));
        }
        return;
    }
    if (mode == RLE) {
        let sym = (sword(c, at) >> 8u) & 0xFFu;
        if (sym != 0u) {
            for (var i = lid; i < n; i += WG) {
                put(c, dest + i * stride, sym);
            }
        }
        return;
    }

    // Huffman: the code lengths, then (invocation 0) checks, canonical codes
    // and the lane layout.
    for (var s = lid; s < 256u; s += WG) {
        lengths[s] = (sbyte(c, at + 4u + s / 2u) >> ((s & 1u) * 4u)) & 15u;
    }
    workgroupBarrier();
    if (lid == 0u) {
        var kraft = 0u;
        var bad = false;
        for (var l = 0u; l <= MAX_LEN; l++) {
            bl[l] = 0u;
        }
        for (var s = 0u; s < 256u; s++) {
            let len = lengths[s];
            if (len > MAX_LEN) {
                bad = true;
            } else if (len > 0u) {
                kraft += 1u << (MAX_LEN - len);
                bl[len] += 1u;
            }
        }
        if (bad || kraft != TABLE) {
            w_err = STREAM_BAD_TABLE;
        } else {
            var code = 0u;
            for (var l = 1u; l <= MAX_LEN; l++) {
                code = (code + bl[l - 1u]) << 1u;
                next_code[l] = code;
            }
            for (var s = 0u; s < 256u; s++) {
                let len = lengths[s];
                if (len > 0u) {
                    codes[s] = reverseBits(next_code[len]) >> (32u - len);
                    next_code[len] += 1u;
                }
            }
            let ln = lanes(n);
            let sizes_at = at + 132u;
            if (sizes_at + 2u * ln > c.comp) {
                w_err = STREAM_TRUNCATED;
            } else {
                let data_at = sizes_at + pad4(2u * ln);
                var total = 0u;
                for (var k = 0u; k < ln; k++) {
                    let words = sbyte(c, sizes_at + 2u * k) | (sbyte(c, sizes_at + 2u * k + 1u) << 8u);
                    lane_words[k] = words;
                    lane_at[k] = data_at + 4u * total;
                    total += words;
                }
                let end = data_at + 4u * total;
                if (end > c.comp) {
                    w_err = STREAM_TRUNCATED;
                }
                w_next = end;
            }
        }
        atomicStore(&overrun, NONE);
    }
    if (workgroupUniformLoad(&w_err) != OK) {
        return;
    }
    // Decode table: entry = symbol | length << 8, indexed by the next MAX_LEN
    // bits (least significant first).
    for (var s = lid; s < 256u; s += WG) {
        let len = lengths[s];
        if (len > 0u) {
            let entry = s | (len << 8u);
            for (var f = 0u; f < (1u << (MAX_LEN - len)); f++) {
                table[codes[s] | (f << len)] = entry;
            }
        }
    }
    workgroupBarrier();
    let ln = lanes(n);
    if (lid < ln) {
        let words = lane_words[lid];
        let base = lane_at[lid];
        let limit = 32u * words;
        var pos = 0u;
        for (var i = lid * n / ln; i < (lid + 1u) * n / ln; i++) {
            let w = pos >> 5u;
            let shift = pos & 31u;
            var peek = 0u;
            if (w < words) {
                peek = sword(c, base + 4u * w) >> shift;
            }
            if (shift > 0u && w + 1u < words) {
                peek |= sword(c, base + 4u * w + 4u) << (32u - shift);
            }
            let e = table[peek & (TABLE - 1u)];
            pos += e >> 8u;
            if (pos > limit) {
                atomicMin(&overrun, lid);
                break;
            }
            put(c, dest + i * stride, e & 0xFFu);
        }
    }
    workgroupBarrier();
    if (lid == 0u && atomicLoad(&overrun) != NONE) {
        w_err = STREAM_LANE_OVERRUN;
    }
}

// Stores the status and reports whether decoding stops (uniform).
fn failed(chunk: u32, lid: u32) -> bool {
    let e = workgroupUniformLoad(&w_err);
    if (e != OK && lid == 0u) {
        status[chunk] = e;
    }
    return e != OK;
}

@compute @workgroup_size(WG)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let chunk = wid.x + wid.y * nwg.x;
    if (chunk >= arrayLength(&descs)) {
        return;
    }
    let d = descs[chunk];
    let c = Chunk(d.src_offset, d.comp_size & ~STORED_BIT, d.image_offset);
    let n = d.uncomp_size;
    if (lid == 0u) {
        w_flag = select(0u, 1u, (d.comp_size & STORED_BIT) != 0u);
    }
    if (workgroupUniformLoad(&w_flag) != 0u) {
        for (var w = lid; w < pad4(c.comp) / 4u; w += WG) {
            atomicStore(&image[c.i / 4u + w], src[c.p / 4u + w]);
        }
        if (lid == 0u) {
            status[chunk] = OK;
        }
        return;
    }

    // Header: present, then counts within GLZ's bounds.
    if (lid == 0u) {
        w_err = OK;
        if (c.comp < 12u) {
            w_err = TRUNCATED;
        } else {
            let head = sword(c, 0u);
            let count = head & ~WIDE_BIT;
            let ext = sword(c, 4u);
            if (count == 0u || count > n / 4u + 1u || ext > 2u * count || sword(c, 8u) > n) {
                w_err = BAD_SEQUENCE;
            } else {
                atomicStore(&image[c.i / 4u], head);
                atomicStore(&image[c.i / 4u + 1u], ext);
            }
        }
    }
    if (failed(chunk, lid)) {
        return;
    }
    let head = sword(c, 0u);
    let count = head & ~WIDE_BIT;
    let ext = sword(c, 4u);
    let lit = sword(c, 8u);
    let offsets_at = 8u + pad4(count);
    let ext_at = offsets_at + pad4(2u * count);
    let ext_len = pad4(ext * select(2u, 4u, (head & WIDE_BIT) != 0u));
    let literals_at = ext_at + ext_len;

    stream(c, 12u, count, 8u, 1u, lid);
    if (failed(chunk, lid)) {
        return;
    }
    var at = workgroupUniformLoad(&w_next);
    stream(c, at, count, offsets_at, 2u, lid);
    if (failed(chunk, lid)) {
        return;
    }
    at = workgroupUniformLoad(&w_next);
    stream(c, at, count, offsets_at + 1u, 2u, lid);
    if (failed(chunk, lid)) {
        return;
    }
    at = workgroupUniformLoad(&w_next);
    if (lid == 0u && at + ext_len > c.comp) {
        w_err = TRUNCATED;
    }
    if (failed(chunk, lid)) {
        return;
    }
    for (var w = lid; w < ext_len / 4u; w += WG) {
        atomicStore(&image[(c.i + ext_at) / 4u + w], sword(c, at + 4u * w));
    }
    stream(c, at + ext_len, lit, literals_at, 1u, lid);
    if (failed(chunk, lid)) {
        return;
    }
    at = workgroupUniformLoad(&w_next);
    if (lid == 0u) {
        status[chunk] = select(OK, SIZE_MISMATCH, at != c.comp);
    }
}
