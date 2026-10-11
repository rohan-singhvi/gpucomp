// Shared by the emit kernels (lz4_emit.wgsl, glz_emit.wgsl); the host appends
// it to encode_common.wgsl, followed by one of them. One workgroup per chunk.

override WG_SIZE: u32 = 64u;

var<workgroup> scan: array<u32, WG_SIZE>;
var<workgroup> scan2: array<vec2<u32>, WG_SIZE>;
// The parse's chunk_info for this chunk, made workgroup-uniform.
var<workgroup> info: vec4<u32>;
// The parse's segment table for this chunk (first sequence of each segment).
var<workgroup> seg_first: array<u32, PARSE_SEGMENTS>;
// Per lane of the current block: its sequence's literal run if it's long
// (output position, input position, length), else length 0. The whole
// workgroup copies these (copy_long_runs).
var<workgroup> long_run: array<vec3<u32>, WG_SIZE>;

// Loads the chunk's segment table. Call in uniform control flow.
fn load_segs(chunk: u32, lid: u32) {
    for (var k = lid; k < PARSE_SEGMENTS; k += WG_SIZE) {
        seg_first[k] = segs[chunk * PARSE_SEGMENTS + k];
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

// output[d .. d + len) = input[s .. s + len) by one invocation, a word at a
// time: atomicOr of each output word's bytes (masked at the run's edges).
fn put_run(d: u32, s: u32, len: u32) {
    if (len == 0u) {
        return;
    }
    let end = d + len;
    for (var w = d >> 2u; w <= (end - 1u) >> 2u; w++) {
        let lo = max(4u * w, d);
        let bytes = min(4u * w + 4u, end) - lo;
        var value = in_word(s + (lo - d));
        if (bytes < 4u) {
            value &= (1u << (bytes * 8u)) - 1u;
        }
        atomicOr(&output[w], value << ((lo & 3u) * 8u));
    }
}

// Copies the block's long literal runs (long_run, filled for `lanes` lanes),
// each by the whole workgroup. Call after a barrier that follows the writes.
fn copy_long_runs(lanes: u32, lid: u32) {
    for (var j = 0u; j < lanes; j++) {
        let r = long_run[j];
        if (r.z != 0u) {
            copy_literals(r.x, r.y, r.z, lid);
        }
    }
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

