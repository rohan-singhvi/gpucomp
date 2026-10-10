// Deep matches on demand, for the parse kernels (appended after
// encode_common.wgsl). With DEPTH > 1, match finding stores each position's
// first-candidate match and its chain link (chain[sbase + p] = the latest
// earlier-block position with its hash, + 1; 0 = none). A position's deep
// match follows links to up to DEPTH candidates, exactly as
// cpu::lz4::encode::find_matches: the longest wins, ties go to the nearest, a
// candidate beyond MAX_OFFSET ends the walk. The parse asks only for the
// positions it visits (~40% of them), and the result is memoised in the match
// word (DEEP) so later phases and the lazy look-ahead reuse it.

override DEPTH: u32 = 1u;
// Match word flags: DEEP = deep match already computed, VISIT = the
// segmented parse's speculative-walk mark (MARK there).
const DEEP: u32 = 0x40000000u;
const VISIT: u32 = 0x80000000u;

// The match at chunk position p (input bytes from `start`, match words from
// `sbase`); `memo` stores it back (only for positions this invocation owns).
fn deep_match(start: u32, sbase: u32, p: u32, match_limit: u32, memo: bool) -> u32 {
    let word = scratch[sbase + p];
    let first = word & ~(VISIT | DEEP);
    if (DEPTH == 1u || (word & DEEP) != 0u) {
        return first;
    }
    let limit = min(p + params.probe_len, match_limit);
    var m = first;
    let head = chain[sbase + p];
    // `first` is the match with the head (candidate 0); stop if there's none,
    // it's out of reach, or `first` already reaches the probe cap.
    if (head != 0u && p - (head - 1u) <= MAX_OFFSET && p + (first >> 16u) < limit) {
        // The 16 bytes at p, compared against every candidate (probe_len <=
        // 16, the default; else plain extend).
        let short_probe = params.probe_len <= 16u;
        var wp = vec4<u32>(0u);
        if (short_probe) {
            wp = vec4<u32>(
                in_word(start + p),
                in_word(start + p + 4u),
                in_word(start + p + 8u),
                in_word(start + p + 12u),
            );
        }
        var candidate = head - 1u;
        for (var k = 1u; k < DEPTH; k++) {
            let entry = chain[sbase + candidate];
            if (entry == 0u) {
                break;
            }
            candidate = entry - 1u;
            let offset = p - candidate;
            if (offset > MAX_OFFSET) {
                break;
            }
            // Only a longer match counts: if a best of length L exists, byte
            // L must match too (skips most extensions).
            let best = m >> 16u;
            var p_byte = 0u;
            if (short_probe) {
                p_byte = (wp[best / 4u] >> ((best % 4u) * 8u)) & 0xFFu;
            } else {
                p_byte = in_byte(start + p + best);
            }
            if (best == 0u || in_byte(start + p + best - offset) == p_byte) {
                var end = 0u;
                if (short_probe) {
                    end = extend16(start, p, offset, limit, wp);
                } else {
                    end = extend(start, p, offset, limit);
                }
                let len = end - p;
                if (len >= MIN_MATCH && len > best) {
                    m = (len << 16u) | offset;
                    if (p + len >= limit) {
                        // Capped: nothing can be longer.
                        break;
                    }
                }
            }
        }
    }
    if (memo) {
        scratch[sbase + p] = m | DEEP | (word & VISIT);
    }
    return m;
}
