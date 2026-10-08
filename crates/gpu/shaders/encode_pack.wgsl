// Encoder kernel 4: packing, one workgroup per chunk (standalone, NOT
// prefixed with encode_common.wgsl). After the host has read the per-chunk
// sizes and laid out the batch's data section, copies each chunk's payload
// into one contiguous buffer, so only compressed bytes are read back:
//   - a compressed chunk: its block, from the start of its output slot;
//   - a stored chunk (block no smaller than the chunk): the raw input bytes.
// Payloads start on 4-byte boundaries, so every copy is whole words; the
// bytes past a payload's end in its last word are zeroed (format padding).

struct Params { // same layout as encode_common.wgsl
    chunk_size: u32,
    input_len: u32,
    slot_size: u32,
    probe_len: u32,
    groups: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// The batch's input, zero-padded to whole words.
@group(0) @binding(0) var<storage, read> input: array<u32>;
// The encode output: slot_size bytes per chunk.
@group(0) @binding(1) var<storage, read> slots: array<u32>;
// Per chunk: (byte offset in the batch's data section, len | STORED_BIT), or
// with SKIP_BIT set: not packed by this pass (with filter selection, each
// candidate's pass packs only the chunks that candidate won).
@group(0) @binding(2) var<storage, read> entries: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> packed: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

const STORED_BIT: u32 = 0x80000000u; // format::STORED_BIT
const SKIP_BIT: u32 = 0x40000000u;   // gpu::encode::SKIP_BIT
const PACK_WG: u32 = 256u;

@compute @workgroup_size(PACK_WG)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let chunk = wid.x + wid.y * nwg.x;
    if (chunk >= (params.input_len + params.chunk_size - 1u) / params.chunk_size) {
        return;
    }
    let entry = entries[chunk];
    if ((entry.y & SKIP_BIT) != 0u) {
        return;
    }
    let stored = (entry.y & STORED_BIT) != 0u;
    let len = entry.y & ~STORED_BIT;
    let dst = entry.x / 4u;
    let words = (len + 3u) / 4u;
    // Mask for the last word: keep only the payload's bytes.
    let tail = len & 3u;
    let last_mask = select(0xFFFFFFFFu, (1u << (tail * 8u)) - 1u, tail != 0u);
    let src = select(chunk * params.slot_size, chunk * params.chunk_size, stored) / 4u;
    for (var w = lid; w < words; w += PACK_WG) {
        var v: u32;
        if (stored) {
            v = input[src + w];
        } else {
            v = slots[src + w];
        }
        if (w + 1u == words) {
            v &= last_mask;
        }
        packed[dst + w] = v;
    }
}
