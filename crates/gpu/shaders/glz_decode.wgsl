// M6: GLZ block decoder, one workgroup per chunk. No serial parse: sequences
// are processed WG_SIZE at a time, one per invocation.
//   1. A scan over per-sequence escape counts gives each sequence its slots in
//      the extension array, hence its literal and match lengths.
//   2. Saturating scans over lengths give each sequence's literal source and
//      output position, and every invocation validates its own sequence. The
//      lowest failing sequence index wins (atomicMin), which is the error the
//      serial CPU decoder (cpu::glz::decode_block) reports.
//   3. All literal runs are copied in parallel.
//   4. Matches resolve in rounds: a match copies once no unfinished earlier
//      match of the block overlaps its source pattern. The earliest pending
//      match is always ready, so every round makes progress. With an encoder
//      that eliminated dependencies in groups of WG_SIZE, one round suffices.
// Each sequence's invocation copies its own short runs; runs of at least
// LONG_COPY bytes are copied by the whole workgroup, word by word (as in the
// M5 LZ4 decoder), so one long run doesn't stall the block.

struct ChunkDesc {
    src_offset: u32,
    comp_size: u32,
    dst_offset: u32,
    uncomp_size: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> chunks: array<ChunkDesc>;
// Zeroed; neighbouring sequences share words at their edges, hence atomicOr.
@group(0) @binding(2) var<storage, read_write> dst: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> status: array<u32>;

override WG_SIZE: u32 = 64u;
override LONG_COPY: u32 = 32u;

const STORED_BIT: u32 = 0x80000000u;
const WIDE_BIT: u32 = 0x80000000u;
const MIN_MATCH: u32 = 4u;
// Lengths and positions saturate here, so malformed input can't wrap u32;
// it's far above any valid chunk size (<= 1 MiB).
const CAP: u32 = 0x40000000u;
const NO_ERROR: u32 = 0xFFFFFFFFu;

const OK: u32 = 0u;
const TRUNCATED: u32 = 1u;
const ZERO_OFFSET: u32 = 2u;
const OFFSET_BEFORE_START: u32 = 3u;
const OUTPUT_OVERFLOW: u32 = 4u;
const SIZE_MISMATCH: u32 = 5u;
const BAD_SEQUENCE: u32 = 6u;

// Header fields, set by invocation 0.
var<workgroup> w_header: u32;
var<workgroup> w_stored: u32;
var<workgroup> w_count: u32;
var<workgroup> w_ext_count: u32;
var<workgroup> w_wide: u32;
var<workgroup> w_offsets_at: u32;
var<workgroup> w_ext_at: u32;
var<workgroup> w_literals_at: u32;

var<workgroup> w_escapes: atomic<u32>;
var<workgroup> w_err: atomic<u32>;
var<workgroup> w_value: u32; // broadcast slot for uniform loads

var<workgroup> scan_esc: array<u32, WG_SIZE>;
var<workgroup> scan_len: array<vec2<u32>, WG_SIZE>; // (literal bytes, output bytes)
var<workgroup> pending: array<u32, WG_SIZE>;
// Bit j set = sequence slot j of the block still has a match to copy.
// (Two words: WG_SIZE must be <= 64.)
var<workgroup> pending_mask: array<atomic<u32>, 2>;
var<workgroup> m_start: array<u32, WG_SIZE>;
var<workgroup> m_end: array<u32, WG_SIZE>;
var<workgroup> m_offset: array<u32, WG_SIZE>;
// Compact list of this step's long copies (sequence slots), built with
// atomicAdd; order is irrelevant because the copies are disjoint.
var<workgroup> long_list: array<u32, WG_SIZE>;
var<workgroup> long_count: atomic<u32>;
var<workgroup> l_src: array<u32, WG_SIZE>; // absolute src byte of the literals
var<workgroup> l_len: array<u32, WG_SIZE>;

fn sbyte(i: u32) -> u32 {
    return (src[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu;
}

fn su16(i: u32) -> u32 {
    return sbyte(i) | (sbyte(i + 1u) << 8u);
}

fn su32(i: u32) -> u32 {
    return su16(i) | (su16(i + 2u) << 16u);
}

fn dbyte(i: u32) -> u32 {
    return (atomicLoad(&dst[i >> 2u]) >> ((i & 3u) * 8u)) & 0xFFu;
}

fn put_byte(i: u32, b: u32) {
    atomicOr(&dst[i >> 2u], b << ((i & 3u) * 8u));
}

// Cooperative: dst[d .. d + len) = src[s .. s + len), one output word per step.
fn copy_literals(d: u32, s: u32, len: u32, lid: u32) {
    let last = (d + len - 1u) >> 2u;
    for (var w = (d >> 2u) + lid; w <= last; w += WG_SIZE) {
        var value = 0u;
        for (var b = 0u; b < 4u; b++) {
            let p = w * 4u + b;
            if (p >= d && p < d + len) {
                value |= sbyte(s + (p - d)) << (b * 8u);
            }
        }
        atomicOr(&dst[w], value);
    }
}

// Cooperative: dst[m .. m + len) = periodic extension of dst[m - offset .. m).
fn copy_match(m: u32, offset: u32, len: u32, lid: u32) {
    let last = (m + len - 1u) >> 2u;
    for (var w = (m >> 2u) + lid; w <= last; w += WG_SIZE) {
        var value = 0u;
        for (var b = 0u; b < 4u; b++) {
            let p = w * 4u + b;
            if (p >= m && p < m + len) {
                value |= dbyte(m - offset + (p - m) % offset) << (b * 8u);
            }
        }
        atomicOr(&dst[w], value);
    }
}

fn sat(a: u32, b: u32) -> u32 {
    return min(a + b, CAP);
}

fn pad4(n: u32) -> u32 {
    return (n + 3u) & ~3u;
}

fn escapes(token: u32) -> u32 {
    return select(0u, 1u, (token >> 4u) == 15u) + select(0u, 1u, (token & 15u) == 15u);
}

// Extension value k (base = byte offset of the chunk's payload).
fn ext(base: u32, k: u32) -> u32 {
    if (w_wide != 0u) {
        return min(su32(base + w_ext_at + 4u * k), CAP);
    }
    return su16(base + w_ext_at + 2u * k);
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let chunk = wid.x + wid.y * nwg.x;
    if (chunk >= arrayLength(&chunks)) {
        return;
    }
    let c = chunks[chunk];
    let base = c.src_offset;
    let n_src = c.comp_size & ~STORED_BIT;
    let out = c.dst_offset;
    let n_dst = c.uncomp_size;

    // ---- Header (invocation 0), same checks and order as the CPU decoder ----
    if (lid == 0u) {
        atomicStore(&w_err, NO_ERROR);
        atomicStore(&w_escapes, 0u);
        w_stored = select(0u, 1u, (c.comp_size & STORED_BIT) != 0u);
        w_count = 0u;
        var hdr = OK;
        if (base + n_src > arrayLength(&src) * 4u || out + n_dst > arrayLength(&dst) * 4u) {
            hdr = TRUNCATED;
            w_stored = 0u;
        } else if (w_stored == 0u) {
            if (n_src < 8u) {
                hdr = TRUNCATED;
            } else {
                let head = su32(base);
                let count = head & ~WIDE_BIT;
                let ext_count = su32(base + 4u);
                if (count == 0u || count > n_dst / MIN_MATCH + 1u || ext_count > 2u * count) {
                    hdr = BAD_SEQUENCE;
                } else {
                    let wide = select(0u, 1u, (head & WIDE_BIT) != 0u);
                    let offsets_at = 8u + pad4(count);
                    let ext_at = offsets_at + pad4(2u * count);
                    let literals_at = ext_at + pad4(ext_count * select(2u, 4u, wide != 0u));
                    if (literals_at > n_src) {
                        hdr = TRUNCATED;
                    } else {
                        w_count = count;
                        w_ext_count = ext_count;
                        w_wide = wide;
                        w_offsets_at = offsets_at;
                        w_ext_at = ext_at;
                        w_literals_at = literals_at;
                    }
                }
            }
        }
        w_header = hdr;
    }
    let stored = workgroupUniformLoad(&w_stored);
    if (stored != 0u) {
        let words = (n_dst + 3u) / 4u;
        for (var w = lid; w < words; w += WG_SIZE) {
            atomicStore(&dst[(out >> 2u) + w], src[(base >> 2u) + w]);
        }
        if (lid == 0u) {
            status[chunk] = OK;
        }
        return;
    }
    let hdr = workgroupUniformLoad(&w_header);
    if (hdr != OK) {
        if (lid == 0u) {
            status[chunk] = hdr;
        }
        return;
    }
    let count = workgroupUniformLoad(&w_count);

    // ---- Escape count must equal the extension count ----
    for (var i = lid; i < count; i += WG_SIZE) {
        let e = escapes(sbyte(base + 8u + i));
        if (e > 0u) {
            atomicAdd(&w_escapes, e);
        }
    }
    workgroupBarrier();
    if (lid == 0u) {
        w_value = select(0u, 1u, atomicLoad(&w_escapes) != w_ext_count);
    }
    if (workgroupUniformLoad(&w_value) != 0u) {
        if (lid == 0u) {
            status[chunk] = BAD_SEQUENCE;
        }
        return;
    }

    let n_lit = n_src - w_literals_at;
    var ext_base = 0u;
    var lit_base = 0u;
    var out_base = 0u;
    for (var block = 0u; block < count; block += WG_SIZE) {
        let i = block + lid;
        let in_range = i < count;

        // 1. Extension slots and lengths.
        var token = 0u;
        var esc = 0u;
        if (in_range) {
            token = sbyte(base + 8u + i);
            esc = escapes(token);
        }
        scan_esc[lid] = esc;
        workgroupBarrier();
        for (var d = 1u; d < WG_SIZE; d <<= 1u) {
            var v = 0u;
            if (lid >= d) {
                v = scan_esc[lid - d];
            }
            workgroupBarrier();
            scan_esc[lid] += v;
            workgroupBarrier();
        }
        let ext_idx = ext_base + scan_esc[lid] - esc;
        let esc_total = scan_esc[WG_SIZE - 1u];

        var lit_len = 0u;
        var match_code = 0u;
        var match_len = 0u;
        var offset = 0u;
        let is_final = i + 1u == count;
        if (in_range) {
            lit_len = token >> 4u;
            var k = ext_idx;
            if (lit_len == 15u) {
                lit_len = sat(15u, ext(base, k));
                k++;
            }
            match_code = token & 15u;
            if (match_code == 15u) {
                match_code = sat(15u, ext(base, k));
            }
            offset = su16(base + w_offsets_at + 2u * i);
            if (!is_final) {
                match_len = sat(match_code, MIN_MATCH);
            }
        }

        // 2. Positions (saturating inclusive scan, then shift to exclusive).
        scan_len[lid] = vec2<u32>(lit_len, sat(lit_len, match_len));
        workgroupBarrier();
        for (var d = 1u; d < WG_SIZE; d <<= 1u) {
            var v = vec2<u32>(0u, 0u);
            if (lid >= d) {
                v = scan_len[lid - d];
            }
            workgroupBarrier();
            scan_len[lid] = min(scan_len[lid] + v, vec2<u32>(CAP, CAP));
            workgroupBarrier();
        }
        var before = vec2<u32>(0u, 0u);
        if (lid > 0u) {
            before = scan_len[lid - 1u];
        }
        let block_total = scan_len[WG_SIZE - 1u];
        let lp = sat(lit_base, before.x);
        let op = sat(out_base, before.y);

        // Validate this sequence (CPU check order).
        var err = NO_ERROR;
        if (in_range) {
            if (lp + lit_len > n_lit) {
                err = TRUNCATED;
            } else if (op + lit_len > n_dst) {
                err = OUTPUT_OVERFLOW;
            } else if (is_final) {
                if (match_code != 0u || offset != 0u) {
                    err = BAD_SEQUENCE;
                }
            } else if (offset == 0u) {
                err = ZERO_OFFSET;
            } else if (offset > op + lit_len) {
                err = OFFSET_BEFORE_START;
            } else if (op + lit_len + match_len > n_dst) {
                err = OUTPUT_OVERFLOW;
            }
            if (err != NO_ERROR) {
                atomicMin(&w_err, (i << 3u) | err);
            }
        }
        let ok = in_range && err == NO_ERROR;

        // 3. Literals (every valid sequence's copy is in bounds): short runs
        // by their owner, long ones by the whole workgroup.
        let lit_src = base + w_literals_at + lp;
        let m = out + op + lit_len;
        let long_lit = ok && lit_len >= LONG_COPY;
        if (ok && !long_lit) {
            for (var k = 0u; k < lit_len; k++) {
                put_byte(out + op + k, sbyte(lit_src + k));
            }
        }
        l_src[lid] = lit_src;
        l_len[lid] = lit_len;
        let has_match = ok && match_len > 0u;
        pending[lid] = select(0u, 1u, has_match);
        m_start[lid] = m;
        m_end[lid] = m + match_len;
        m_offset[lid] = offset;
        if (lid == 0u) {
            atomicStore(&long_count, 0u);
        }
        workgroupBarrier();
        if (long_lit) {
            long_list[atomicAdd(&long_count, 1u)] = lid;
        }
        workgroupBarrier();
        if (lid == 0u) {
            w_value = atomicLoad(&long_count);
        }
        let long_lits = workgroupUniformLoad(&w_value);
        for (var k = 0u; k < long_lits; k++) {
            let j = long_list[k];
            copy_literals(m_start[j] - l_len[j], l_src[j], l_len[j], lid);
        }
        storageBarrier();
        workgroupBarrier();

        // 4. Matches, in dependency rounds. Each match records (once) which
        // earlier matches of the block overlap its source pattern, as a
        // 64-bit mask, and copies once none of them is pending.
        let src_start = m - offset;
        let src_end = src_start + min(offset, match_len);
        var deps = vec2<u32>(0u, 0u);
        if (has_match) {
            for (var j = 0u; j < lid; j++) {
                if (pending[j] == 1u && m_start[j] < src_end && src_start < m_end[j]) {
                    deps[j >> 5u] |= 1u << (j & 31u);
                }
            }
        }
        loop {
            if (lid == 0u) {
                atomicStore(&pending_mask[0], 0u);
                atomicStore(&pending_mask[1], 0u);
                atomicStore(&long_count, 0u);
            }
            workgroupBarrier();
            if (pending[lid] == 1u) {
                atomicOr(&pending_mask[lid >> 5u], 1u << (lid & 31u));
            }
            workgroupBarrier();
            let busy = vec2<u32>(atomicLoad(&pending_mask[0]), atomicLoad(&pending_mask[1]));
            if (lid == 0u) {
                w_value = busy.x | busy.y;
            }
            if (workgroupUniformLoad(&w_value) == 0u) {
                break;
            }
            let ready = pending[lid] == 1u && all((deps & busy) == vec2<u32>(0u, 0u));
            if (ready && match_len >= LONG_COPY) {
                long_list[atomicAdd(&long_count, 1u)] = lid;
            }
            if (ready && match_len < LONG_COPY) {
                // Serial copy: an overlapping match reads bytes it just wrote.
                for (var k = 0u; k < match_len; k++) {
                    put_byte(m + k, dbyte(m + k - offset));
                }
            }
            workgroupBarrier();
            if (lid == 0u) {
                w_value = atomicLoad(&long_count);
            }
            let long_matches = workgroupUniformLoad(&w_value);
            for (var k = 0u; k < long_matches; k++) {
                let j = long_list[k];
                copy_match(m_start[j], m_offset[j], m_end[j] - m_start[j], lid);
            }
            storageBarrier();
            workgroupBarrier();
            if (ready) {
                pending[lid] = 0u;
            }
            workgroupBarrier();
        }

        ext_base += esc_total;
        lit_base = sat(lit_base, block_total.x);
        out_base = sat(out_base, block_total.y);
        // Stop at the first failing block. (The barrier keeps invocation 0
        // from overwriting w_value before everyone has read the last value.)
        workgroupBarrier();
        if (lid == 0u) {
            w_value = atomicLoad(&w_err);
        }
        if (workgroupUniformLoad(&w_value) != NO_ERROR) {
            break;
        }
    }

    if (lid == 0u) {
        let e = atomicLoad(&w_err);
        if (e != NO_ERROR) {
            status[chunk] = e & 7u;
        } else if (out_base != n_dst || lit_base != n_lit) {
            status[chunk] = SIZE_MISMATCH;
        } else {
            status[chunk] = OK;
        }
    }
}
