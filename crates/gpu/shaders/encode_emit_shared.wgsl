// Shared by the emit kernels (lz4_emit.wgsl, glz_emit.wgsl); the host appends
// it to encode_common.wgsl, followed by one of them. One workgroup per chunk.

override WG_SIZE: u32 = 64u;

var<workgroup> scan: array<u32, WG_SIZE>;
var<workgroup> scan2: array<vec2<u32>, WG_SIZE>;
// The parse's chunk_info for this chunk, made workgroup-uniform.
var<workgroup> info: vec4<u32>;

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

