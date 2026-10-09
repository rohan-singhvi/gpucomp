// Shared by the emit kernels (lz4_emit.wgsl, glz_emit.wgsl); the host appends
// it to encode_common.wgsl, followed by one of them. One workgroup per chunk.

override WG_SIZE: u32 = 64u;

var<workgroup> scan: array<u32, WG_SIZE>;
var<workgroup> scan2: array<vec2<u32>, WG_SIZE>;
// Scratch word of each sequence in the current block (see seq_addr).
var<workgroup> qaddr: array<u32, WG_SIZE>;
// The parse's chunk_info for this chunk, made workgroup-uniform.
var<workgroup> info: vec4<u32>;
// The parse's segment table for this chunk (first sequence of each segment).
var<workgroup> seg_first: array<u32, PARSE_SEGMENTS>;

// Loads the chunk's segment table. Call in uniform control flow.
fn load_segs(chunk: u32, lid: u32) {
    if (lid < PARSE_SEGMENTS) {
        seg_first[lid] = segs[chunk * PARSE_SEGMENTS + lid];
    }
    workgroupBarrier();
}

// Scratch word of sequence i: in the last segment whose first sequence is
// <= i (binary search; seg_first is non-decreasing and seg_first[0] = 0).
fn seq_addr(sbase: u32, n: u32, i: u32) -> u32 {
    var k = 0u;
    for (var step = PARSE_SEGMENTS / 2u; step > 0u; step >>= 1u) {
        if (seg_first[k + step] <= i) {
            k += step;
        }
    }
    return sbase + k * seg_len(n) + 4u * (i - seg_first[k]);
}

// Loads chunk_info[chunk] uniformly: (count, encoded size, ext count, wide).
fn load_info(chunk: u32, lid: u32) -> vec4<u32> {
    if (lid == 0u) {
        info = chunk_info[chunk];
    }
    return workgroupUniformLoad(&info);
}

// Cooperative: output[d .. d + len) = input[s .. s + len), one output word
// per step, merged with atomicOr (edge words are shared with neighbours).
fn copy_literals(d: u32, s: u32, len: u32, lid: u32) {
    let last = (d + len - 1u) >> 2u;
    for (var w = (d >> 2u) + lid; w <= last; w += WG_SIZE) {
        var value = 0u;
        for (var b = 0u; b < 4u; b++) {
            let p = w * 4u + b;
            if (p >= d && p < d + len) {
                value |= in_byte(s + (p - d)) << (b * 8u);
            }
        }
        atomicOr(&output[w], value);
    }
}

