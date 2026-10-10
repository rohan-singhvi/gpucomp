// Encoder kernel 1b (DEPTH > 1 only; appended to encode_common.wgsl and
// encode_deep.wgsl): ONE INVOCATION PER POSITION, no workgroup memory. Each
// position's deep match (chain candidates beyond the first), memoised in its
// match word for the parse. Running it apart from match finding (whose 16 KB
// hash table limits occupancy) lets many more invocations hide the latency
// of the link and candidate loads.

const DWG: u32 = 256u;

@compute @workgroup_size(DWG)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = gid.x + gid.y * nwg.x * DWG;
    if (i >= params.input_len) {
        return;
    }
    let chunk = i / params.chunk_size;
    let p = i % params.chunk_size;
    let n = chunk_len(chunk);
    if (n < MFLIMIT || p + MFLIMIT > n) {
        return;
    }
    let start = chunk * params.chunk_size;
    deep_match(start, start, p, n - LAST_LITERALS, true);
}
