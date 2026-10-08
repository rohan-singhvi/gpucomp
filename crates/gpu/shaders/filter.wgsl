// M7: byte-shuffle and delta filters (plan §4a), forward and inverse.
// One workgroup per job. A job reads `len` bytes at `src_offset` of `src` and
// writes `len` bytes at `dst_offset` of `dst` (both 4-aligned). With width w
// there are m = len / w whole elements; the trailing len % w bytes are copied.
// Each output word belongs to exactly one invocation, which writes it whole;
// the only word a job shares with bytes outside it is its last one, where the
// bytes past `len` are preserved (read-modify-write by the owner).
//
// Entry points: shuffle_forward, shuffle_inverse, delta_forward (per word,
// independent) and delta_inverse (a prefix sum: per-invocation segment sums,
// a workgroup scan, then each segment re-summed from its prefix).

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
    let w = job.width;
    let m = job.len / w;
    let words = (job.len + 3u) / 4u;
    for (var k = lid; k < words; k += WG_SIZE) {
        var v = 0u;
        if (w == 8u) {
            let e = k / 2u;
            if (e < m) {
                var prev = vec2<u32>(0u, 0u);
                if (e > 0u) {
                    prev = elem(job, e - 1u);
                }
                let d = sub64(elem(job, e), prev);
                v = select(d.x, d.y, (k & 1u) == 1u);
            }
        } else {
            let per = 4u / w;
            for (var t = 0u; t < per; t++) {
                let e = k * per + t;
                if (e < m) {
                    var prev = 0u;
                    if (e > 0u) {
                        prev = elem(job, e - 1u).x;
                    }
                    let d = (elem(job, e).x - prev) & width_mask(w);
                    v |= d << (8u * w * t);
                }
            }
        }
        put_word(job, k, with_tail(job, k, v));
    }
}

@compute @workgroup_size(WG_SIZE)
fn delta_inverse(
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
    let m = job.len / w;
    // Elements per invocation, a whole number of words (w < 4: a multiple of 4 / w).
    let per = max(1u, 4u / w);
    let seg = ((m + WG_SIZE - 1u) / WG_SIZE + per - 1u) / per * per;
    let first = min(lid * seg, m);
    let end = min(first + seg, m);

    // 1. Each invocation sums its segment.
    var sum = vec2<u32>(0u, 0u);
    for (var e = first; e < end; e++) {
        sum = add64(sum, elem(job, e));
    }

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
            running = add64(running, elem(job, e));
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
                    running = add64(running, elem(job, e));
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
