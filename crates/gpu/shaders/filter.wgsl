// M7: byte-shuffle and delta filters and their combinations (plan §4a;
// crates/format/FORMAT.md), forward and inverse.
// One workgroup per job. A job reads `len` bytes at `src_offset` of `src` and
// writes `len` bytes at `dst_offset` of `dst` (both 4-aligned). With width w
// there are m = len / w whole elements; the trailing len % w bytes are copied.
// Each output word belongs to exactly one invocation, which writes it whole;
// the only word a job shares with bytes outside it is its last one, where the
// bytes past `len` are preserved (read-modify-write by the owner).
//
// Entry points: shuffle_forward, shuffle_inverse, delta_forward,
// delta_shuffle_forward and delta_zigzag_shuffle_forward (per word,
// independent), delta_bitshuffle_forward (per 32-element tile and byte), and
// delta_inverse, delta_shuffle_inverse, delta_zigzag_shuffle_inverse and
// delta_bitshuffle_inverse (a prefix sum: per-invocation segment sums, a
// workgroup scan, then each segment re-summed from its prefix).

struct Job {
    src_offset: u32,
    dst_offset: u32,
    len: u32,
    width: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> jobs: array<Job>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

override WG_SIZE: u32 = 256u;

var<workgroup> w_scan: array<vec2<u32>, WG_SIZE>;

fn src_byte(i: u32) -> u32 {
    return (src[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu;
}

// Element e of the job's input as a little-endian (lo, hi) pair.
fn elem(job: Job, e: u32) -> vec2<u32> {
    if (job.width == 8u) {
        let at = (job.src_offset >> 2u) + 2u * e;
        return vec2<u32>(src[at], src[at + 1u]);
    }
    let byte = job.src_offset + e * job.width;
    let word = src[byte >> 2u] >> ((byte & 3u) * 8u);
    return vec2<u32>(word & width_mask(job.width), 0u);
}

// Low `8 * w` bits for w < 4, all of them otherwise.
fn width_mask(w: u32) -> u32 {
    return select(0xFFFFFFFFu, (1u << (8u * w)) - 1u, w < 4u);
}

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}

fn sub64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}

// Fills the bytes of output word k at or past m * w (the unfiltered tail)
// from the input; `v` must have zeros there.
fn with_tail(job: Job, k: u32, v: u32) -> u32 {
    let tail = (job.len / job.width) * job.width;
    var out = v;
    for (var b = 0u; b < 4u; b++) {
        let p = 4u * k + b;
        if (p >= tail && p < job.len) {
            out |= src_byte(job.src_offset + p) << (8u * b);
        }
    }
    return out;
}

// Writes output word k, keeping bytes past `len` in a final partial word.
fn put_word(job: Job, k: u32, v: u32) {
    let at = (job.dst_offset >> 2u) + k;
    let valid = job.len - 4u * k;
    if (valid >= 4u) {
        dst[at] = v;
    } else {
        let mask = (1u << (8u * valid)) - 1u;
        dst[at] = (dst[at] & ~mask) | (v & mask);
    }
}

fn job_index(wg: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wg.x + wg.y * nwg.x;
}

// Shuffle: byte j of element i sits at j * m + i of the shuffled form.
// The source byte of output byte p (< m * w).
fn shuffle_source(job: Job, p: u32, forward: bool) -> u32 {
    let w = job.width;
    let m = job.len / w;
    if (p >= m * w) {
        return p;
    }
    if (forward) {
        return (p % m) * w + p / m;
    }
    return (p % w) * m + p / w;
}

fn shuffle(wg: vec3<u32>, nwg: vec3<u32>, lid: u32, forward: bool) {
    let j = job_index(wg, nwg);
    if (j >= arrayLength(&jobs)) {
        return;
    }
    let job = jobs[j];
    let words = (job.len + 3u) / 4u;
    for (var k = lid; k < words; k += WG_SIZE) {
        var v = 0u;
        for (var b = 0u; b < 4u; b++) {
            let p = 4u * k + b;
            if (p < job.len) {
                v |= src_byte(job.src_offset + shuffle_source(job, p, forward)) << (8u * b);
            }
        }
        put_word(job, k, v);
    }
}

@compute @workgroup_size(WG_SIZE)
fn shuffle_forward(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    shuffle(wg, nwg, lid, true);
}

@compute @workgroup_size(WG_SIZE)
fn shuffle_inverse(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    shuffle(wg, nwg, lid, false);
}

// Element e's delta, elem[e] - elem[e - 1] (wrapping; elem[0] itself). For
// w < 8 only the low 8 * w bits are meaningful.
fn delta_of(job: Job, e: u32) -> vec2<u32> {
    var prev = vec2<u32>(0u, 0u);
    if (e > 0u) {
        prev = elem(job, e - 1u);
    }
    return sub64(elem(job, e), prev);
}

// Output word k of delta for the elements from byte 0, without the tail.
fn delta_word(job: Job, k: u32) -> u32 {
    let w = job.width;
    let m = job.len / w;
    var v = 0u;
    if (w == 8u) {
        let e = k / 2u;
        if (e < m) {
            let d = delta_of(job, e);
            v = select(d.x, d.y, (k & 1u) == 1u);
        }
    } else {
        let per = 4u / w;
        for (var t = 0u; t < per; t++) {
            let e = k * per + t;
            if (e < m) {
                v |= (delta_of(job, e).x & width_mask(w)) << (8u * w * t);
            }
        }
    }
    return v;
}

// Delta: element i becomes elem[i] - elem[i - 1] (wrapping), elem[0] kept.
@compute @workgroup_size(WG_SIZE)
fn delta_forward(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let j = job_index(wg, nwg);
    if (j >= arrayLength(&jobs)) {
        return;
    }
    let job = jobs[j];
    let words = (job.len + 3u) / 4u;
    for (var k = lid; k < words; k += WG_SIZE) {
        put_word(job, k, with_tail(job, k, delta_word(job, k)));
    }
}

// Zigzag of a w-byte two's-complement delta: 0, -1, 1, -2, ... become 0, 1,
// 2, 3, ... (bits past 8 * w of the result are zero for w < 8).
fn zigzag(d: vec2<u32>, w: u32) -> vec2<u32> {
    if (w == 8u) {
        let neg = 0u - (d.y >> 31u);
        return vec2<u32>((d.x << 1u) ^ neg, ((d.y << 1u) | (d.x >> 31u)) ^ neg);
    }
    let neg = 0u - ((d.x >> (8u * w - 1u)) & 1u);
    return vec2<u32>(((d.x << 1u) ^ neg) & width_mask(w), 0u);
}

fn unzigzag(z: vec2<u32>) -> vec2<u32> {
    let neg = 0u - (z.x & 1u);
    return vec2<u32>(((z.x >> 1u) | (z.y << 31u)) ^ neg, (z.y >> 1u) ^ neg);
}

// Delta (zigzagged when `zz`), then shuffle: output byte p < m * w is byte
// p / m of element (p % m)'s delta.
fn delta_shuffled(wg: vec3<u32>, nwg: vec3<u32>, lid: u32, zz: bool) {
    let j = job_index(wg, nwg);
    if (j >= arrayLength(&jobs)) {
        return;
    }
    let job = jobs[j];
    let w = job.width;
    let m = job.len / w;
    let words = (job.len + 3u) / 4u;
    for (var k = lid; k < words; k += WG_SIZE) {
        var v = 0u;
        for (var b = 0u; b < 4u; b++) {
            let p = 4u * k + b;
            if (p < m * w) {
                let byte = p / m;
                var d = delta_of(job, p % m);
                if (zz) {
                    d = zigzag(d, w);
                }
                let half = select(d.x, d.y, byte >= 4u);
                v |= ((half >> (8u * (byte & 3u))) & 0xFFu) << (8u * b);
            }
        }
        put_word(job, k, with_tail(job, k, v));
    }
}

@compute @workgroup_size(WG_SIZE)
fn delta_shuffle_forward(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    delta_shuffled(wg, nwg, lid, false);
}

@compute @workgroup_size(WG_SIZE)
fn delta_zigzag_shuffle_forward(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    delta_shuffled(wg, nwg, lid, true);
}

// Words per bit plane of delta-bitshuffle: the first m / 32 * 32 elements
// are spread over 8 * w planes, one bit per element, element e at bit e % 32
// of word e / 32; later elements keep their deltas, in place.
fn plane_words(job: Job) -> u32 {
    return job.len / job.width / 32u;
}

// Delta, then bit-shuffle. One item per 32-element tile t and byte j of the
// element: the tile's word of planes 8j..8j + 8, all eight at once.
@compute @workgroup_size(WG_SIZE)
fn delta_bitshuffle_forward(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let j = job_index(wg, nwg);
    if (j >= arrayLength(&jobs)) {
        return;
    }
    let job = jobs[j];
    let w = job.width;
    let pw = plane_words(job);
    let low = vec4<u32>(0u, 1u, 2u, 3u);
    for (var item = lid; item < pw * w; item += WG_SIZE) {
        let t = item / w;
        let byte = item % w;
        var lo = vec4<u32>(0u);
        var hi = vec4<u32>(0u);
        for (var i = 0u; i < 32u; i++) {
            let d = delta_of(job, 32u * t + i);
            let half = select(d.x, d.y, byte >= 4u);
            let x = vec4<u32>((half >> (8u * (byte & 3u))) & 0xFFu);
            lo |= ((x >> low) & vec4<u32>(1u)) << vec4<u32>(i);
            hi |= ((x >> (low + 4u)) & vec4<u32>(1u)) << vec4<u32>(i);
        }
        for (var b = 0u; b < 4u; b++) {
            put_word(job, (8u * byte + b) * pw + t, lo[b]);
            put_word(job, (8u * byte + b + 4u) * pw + t, hi[b]);
        }
    }
    // Elements past the planes keep their deltas; then the tail.
    let words = (job.len + 3u) / 4u;
    for (var k = 8u * w * pw + lid; k < words; k += WG_SIZE) {
        put_word(job, k, with_tail(job, k, delta_word(job, k)));
    }
}

// How the inverse kernels find element e's delta in the job's input.
const PLAIN: u32 = 0u; // delta: contiguous
const SHUFFLED: u32 = 1u; // delta-shuffle: byte j at j * m + e
const ZIGZAG: u32 = 2u; // delta-zigzag-shuffle: shuffled, then unzigzagged
const BIT_PLANES: u32 = 3u; // delta-bitshuffle: bit k from plane k

fn input_elem(job: Job, e: u32, stored_as: u32) -> vec2<u32> {
    if (stored_as == PLAIN) {
        return elem(job, e);
    }
    var v = vec2<u32>(0u, 0u);
    if (stored_as == BIT_PLANES) {
        let pw = plane_words(job);
        if (e >= 32u * pw) {
            return elem(job, e);
        }
        let at = (job.src_offset >> 2u) + e / 32u;
        let shift = e % 32u;
        for (var k = 0u; k < 8u * job.width; k++) {
            let bit = (src[at + k * pw] >> shift) & 1u;
            if (k < 32u) {
                v.x |= bit << k;
            } else {
                v.y |= bit << (k - 32u);
            }
        }
        return v;
    }
    let m = job.len / job.width;
    for (var byte = 0u; byte < job.width; byte++) {
        let b = src_byte(job.src_offset + byte * m + e);
        if (byte < 4u) {
            v.x |= b << (8u * byte);
        } else {
            v.y |= b << (8u * (byte - 4u));
        }
    }
    if (stored_as == ZIGZAG) {
        return unzigzag(v);
    }
    return v;
}

@compute @workgroup_size(WG_SIZE)
fn delta_inverse(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    running_sum(wg, nwg, lid, PLAIN);
}

@compute @workgroup_size(WG_SIZE)
fn delta_shuffle_inverse(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    running_sum(wg, nwg, lid, SHUFFLED);
}

@compute @workgroup_size(WG_SIZE)
fn delta_zigzag_shuffle_inverse(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    running_sum(wg, nwg, lid, ZIGZAG);
}

@compute @workgroup_size(WG_SIZE)
fn delta_bitshuffle_inverse(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    running_sum(wg, nwg, lid, BIT_PLANES);
}

// `count << k` as a 64-bit (lo, hi) pair, for k < 64.
fn shl64(count: u32, k: u32) -> vec2<u32> {
    if (k >= 32u) {
        return vec2<u32>(0u, count << (k - 32u));
    }
    return vec2<u32>(count << k, select(count >> (32u - k), 0u, k == 0u));
}

// Bits lo..hi of a word (0 <= lo < hi <= 32).
fn bit_range(lo: u32, hi: u32) -> u32 {
    let below_hi = select((1u << hi) - 1u, 0xFFFFFFFFu, hi == 32u);
    return below_hi & ~((1u << lo) - 1u);
}

// The sum of deltas first..end. In bit planes, that's the sum over planes k
// of 2^k times the number of set bits: a popcount per word, not a gather per
// element.
fn segment_sum(job: Job, first: u32, end: u32, stored_as: u32) -> vec2<u32> {
    var sum = vec2<u32>(0u, 0u);
    var e = first;
    if (stored_as == BIT_PLANES) {
        let pw = plane_words(job);
        let planes_end = min(end, 32u * pw);
        if (e < planes_end) {
            let base = job.src_offset >> 2u;
            for (var k = 0u; k < 8u * job.width; k++) {
                var count = 0u;
                for (var q = e / 32u; 32u * q < planes_end; q++) {
                    let lo = max(e, 32u * q) - 32u * q;
                    let hi = min(planes_end, 32u * q + 32u) - 32u * q;
                    count += countOneBits(src[base + k * pw + q] & bit_range(lo, hi));
                }
                sum = add64(sum, shl64(count, k));
            }
            e = planes_end;
        }
    }
    for (; e < end; e++) {
        sum = add64(sum, input_elem(job, e, stored_as));
    }
    return sum;
}

// The inverse of every delta filter: a running sum of the job's deltas, read
// through `stored_as`.
fn running_sum(wg: vec3<u32>, nwg: vec3<u32>, lid: u32, stored_as: u32) {
    let j = job_index(wg, nwg);
    if (j >= arrayLength(&jobs)) {
        return;
    }
    let job = jobs[j];
    let w = job.width;
    let m = job.len / w;
    // Elements per invocation, a whole number of words (w < 4: a multiple of 4 / w).
    let per = max(1u, 4u / w);
    let seg = ((m + WG_SIZE - 1u) / WG_SIZE + per - 1u) / per * per;
    let first = min(lid * seg, m);
    let end = min(first + seg, m);

    // 1. Each invocation sums its segment.
    let sum = segment_sum(job, first, end, stored_as);

    // 2. Inclusive scan of the segment sums (Hillis–Steele).
    w_scan[lid] = sum;
    workgroupBarrier();
    for (var step = 1u; step < WG_SIZE; step <<= 1u) {
        var other = vec2<u32>(0u, 0u);
        if (lid >= step) {
            other = w_scan[lid - step];
        }
        workgroupBarrier();
        w_scan[lid] = add64(w_scan[lid], other);
        workgroupBarrier();
    }
    var running = vec2<u32>(0u, 0u);
    if (lid > 0u) {
        running = w_scan[lid - 1u];
    }

    // 3. Re-sum the segment from its prefix, writing whole words.
    if (w == 8u) {
        for (var e = first; e < end; e++) {
            running = add64(running, input_elem(job, e, stored_as));
            put_word(job, 2u * e, running.x);
            put_word(job, 2u * e + 1u, running.y);
        }
    } else if (first < end) {
        let mask = width_mask(w);
        let last_word = (end * w + 3u) / 4u;
        for (var k = first * w / 4u; k < last_word; k++) {
            var v = 0u;
            for (var t = 0u; t < per; t++) {
                let e = k * per + t;
                if (e < end) {
                    running = add64(running, input_elem(job, e, stored_as));
                    v |= (running.x & mask) << (8u * w * t);
                }
            }
            put_word(job, k, with_tail(job, k, v));
        }
    }

    // 4. Words entirely in the tail (none of them holds an element byte).
    let elem_words = (m * w + 3u) / 4u;
    let words = (job.len + 3u) / 4u;
    for (var k = elem_words + lid; k < words; k += WG_SIZE) {
        put_word(job, k, with_tail(job, k, 0u));
    }
}
