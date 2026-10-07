// M0 smoke test: data[i] ^= key.

struct Params {
    key: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

override WG_SIZE: u32 = 64u;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    // 2D grid (see dispatch_grid): linearise, then bounds-check.
    let i = gid.x + gid.y * nwg.x * WG_SIZE;
    if (i >= arrayLength(&data)) {
        return;
    }
    data[i] = data[i] ^ params.key;
}
