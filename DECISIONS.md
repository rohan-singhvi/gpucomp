# Decisions

Deviations from `plan.md` and non-obvious choices, newest last.

## M0 — wgpu 30.0.1 API notes
`wgpu::InstanceDescriptor` no longer implements `Default`; we build it from
`InstanceDescriptor::new_without_display_handle_from_env()` so the standard `WGPU_*`
environment variables still work, overriding `backends` (default: `Backends::PRIMARY`,
i.e. Metal/D3D12/Vulkan). Device limits start from WebGPU defaults (downlevel defaults
if the adapter can't meet them) and raise only `max_storage_buffer_binding_size`,
`max_buffer_size` and `max_compute_workgroup_storage_size` to the adapter's values, so
we never request a limit the adapter lacks. On Apple M4 Pro / Metal wgpu reports
`max_buffer_size` = 4 GiB − 1, which M7's batching must respect.

## M0 — License
The repository ships under the "MIT NON-AI" license already committed upstream; this
answers open question 4 in the plan.

## M0 — GPU timestamps are probed, not trusted
On Apple M4 Pro / Metal with wgpu 30.0.1, the adapter advertises `TIMESTAMP_QUERY`,
but compute-pass timestamps (and encoder-level ones) resolve to all zeros, even
around a 17 ms dispatch. `GpuTimer::new` therefore runs an empty timed pass and
returns `None` unless the samples are nonzero and increasing. Benchmarks then time
kernels with wall clock around submit + wait (`wall-gpu`), which includes submission
overhead. Kernel-only numbers on this Mac are an upper bound on kernel time until
wgpu's Metal timestamps work. Windows (D3D12/Vulkan) should take the timestamp path.

## M0 — 2D dispatch grid
`max_compute_workgroups_per_dimension` is 65535, which caps a 1D dispatch of
64-invocation workgroups at about 16 MiB of `u32`s. Kernels use `dispatch_grid`, which
wraps the workgroups into an (x, y) grid, and each shader linearises
`gid.x + gid.y * num_workgroups.x * WG_SIZE` and bounds-checks.

## M0 — Benchmark log design
Each `gpucomp bench --record` run is one JSON file in `bench/results/`. That file is
the raw data, and it's never edited except to add a missing `note`.
`gpucomp bench --report` regenerates `BENCHMARKS.md`: per adapter, the latest run in
full, a history matrix (measurement × last 8 runs), and the "what changed" notes.
The first baseline (Apple M4 Pro) shows a trivial kernel at ~112 GB/s against
6–9 GB/s for host↔GPU copies. End-to-end paths on this machine are transfer-bound,
and per-call pipeline/buffer creation measurably hurts end to end (XOR e2e 2.4 GB/s
vs ~3.6 GB/s predicted from its parts), so codec pipelines must be cached.

## M1 — Greedy encoder: probe-capped match finding, extension in the parse
The plan's phase 1 extends a match at every position, which is O(n × match length):
about 2·10⁹ byte compares for a 64 KiB chunk of zeros, on the CPU and the GPU alike.
Phase 1 now extends each candidate only up to `probe_len` bytes (default 16). The
serial parse phase extends the matches it actually takes, which is O(n) in total
because it consumes the bytes it extends over. Lengths, offsets and the greedy choice
come out the same as with unlimited probing, except where a capped probe hides which of
two candidates is longer. With one candidate per position there's no choice to hide.
The CPU twin (`cpu::lz4::encode`) and the M3 GPU encoder run this same algorithm
(`block`, `hash_log`, `probe_len` parameters, latest-position-wins buckets), so their
outputs should be byte-identical, which M3 tests.
Silesia at 64 KiB chunks: greedy twin 1.97× vs `lz4_flex` 2.04× (3.5% behind;
target ≤ 10%).

## M1 — Final payload padding is mandatory
Every payload, including the last, is padded to 4 bytes, and validation checks that the
padding is present. Without this, a file truncated by 1–3 bytes inside the final
padding still decoded, so truncation went unnoticed. Since GPU uploads work in whole
words anyway, requiring the padding costs nothing.

## M1 — Hand-written decoder favours clarity
`cpu::lz4::decode` copies matches byte by byte, which makes overlap semantics obvious
and mirrors the M2 shader. It decodes at ~8 GB/s multi-threaded on Silesia vs
`lz4_flex`'s ~13 GB/s. It's the reference, and `lz4_flex` stays the speed baseline.

## M1 — Benchmark report shows totals, not per-file rows
With Silesia, the report was 237 lines. `BENCHMARKS.md` now shows one row per
measurement and one column per input. Per-file corpus inputs (`silesia/<file>`) stay in
the JSON, and `silesia/all` is shown. History tracks one headline input per measurement,
preferring `silesia/all`, then `text`, then `random`.

## M2 — Naive GPU decoder writes bytes with OR into owned words
The plan suggested accumulating output bytes in a register and flushing whole words.
Matches read back bytes written moments earlier, possibly still in that register,
which makes the bookkeeping fiddly. M2 instead ORs each byte into its word: the
output buffer is zeroed, every byte is written exactly once, and each invocation owns
all the words of its chunk (chunk outputs start 4-aligned, and only the last chunk ends
unaligned). That makes the read-modify-write race-free. It's the correctness baseline,
and M5 replaces it.

## M2 — Shader mirrors the CPU decoder's checks, in order
`lz4_decode_naive.wgsl` follows `cpu::lz4::decode::decode_block` step for step and
writes a per-chunk status (`ChunkStatus` = CPU `DecodeError`). A test feeds both
decoders the same malformed payloads and requires the same error. Every loop
consumes input or is bounded by the chunk's output size, so malformed input can't
hang the GPU. A property test flips random bytes and runs the decode.

## M2 — Shared helpers moved to `format`
`read_index` and `checksum` are part of the container spec, so they moved from `cpu`
to `format` (re-exported by `cpu`). `gpu` depends on `format` only, and on `cpu` only
as a dev-dependency for tests.

## M2 — One batch per decode until M8
A decode uploads the whole needed payload span and output in one binding each. Spans
larger than `max_storage_buffer_binding_size` (capped to u32 addressing) are rejected
with `TooLarge`. M8 adds batching.

## M2 — Results (Apple M4 Pro, 256 MiB inputs, 64 KiB chunks)
Naive GPU decode, kernel only: Silesia 2.0 GB/s, text 7.3 GB/s, vs `lz4_flex`
multi-threaded at 12–13 GB/s. Throughput scales with chunk count, because each chunk
is one serial thread (4096 threads for 256 MiB), so most of the GPU sits idle. End to
end it's 1.3–2.7 GB/s, bounded by the 6–10 GB/s transfers plus readback. M5
parallelises within each chunk.

## M3 — GPU encoder is byte-identical to the CPU twin
`lz4_encode.wgsl` runs `cpu::lz4::encode`'s algorithm phase for phase: workgroup =
match-finding block, one `atomicMax` insert per position after the block's lookups
(latest position wins), `probe_len`-capped probes, and a serial greedy parse that
extends the matches it takes. Tests require identical blocks, and identical whole
containers (`gpu compress` == `cpu compress --encoder greedy`), for several
`(block, hash_log, probe_len)` settings and for random inputs. This is stronger than
the plan's "valid and deterministic", and makes any GPU bug show up as a diff
against a CPU reference that's easy to debug.

## M3 — In-place sequences, workgroup scan for output offsets
The match scratch buffer (one u32 per position: `len << 16 | offset`) is overwritten
in place by the parse's sequences (4 u32 each). Sequence k lands at words 4k..4k+3,
and the k earlier matches each consumed ≥ 4 positions, so it never overwrites an
unread match. The emit phase scans encoded sequence sizes with a Hillis–Steele scan
in workgroup memory, `WG_SIZE` sequences at a time, then writes in parallel with
`atomicOr` into the zeroed output slot (the plan's first option). `HASH_LOG` and
`WG_SIZE` are pipeline overrides. Override-sized workgroup arrays work in wgpu 30,
and `new()` rejects parameters that exceed the device's workgroup memory or size.
Output packing (stored fallback, checksums, layout) happens on the host through
`format::assemble`, now shared with the CPU encoder.

## M3 — Encoder profile: the serial parse dominates
Apple M4 Pro, Silesia, 64 KiB chunks, timed by stopping the shader after each phase
(throwaway experiment, reverted):
phase 1 (match finding) ≈ 30%, phase 2 (one-thread parse) ≈ 45–60%, phase 3
(emit) ≈ 15–25%. Kernel throughput is ~0.7 GB/s, vs `lz4_flex` at 5.1 GB/s
multi-threaded. The parse runs on one invocation per workgroup, reading every
position's match from storage memory, so 63 of 64 lanes idle for most of the
kernel. Incompressible chunks are also slow in emit, because one invocation copies
an entire 64 KiB literal run with byte-wise atomics. Planned next steps, in order:
(1) parallel parse (pointer jumping over "next position" links, as the plan
suggests), (2) cooperative literal copies in emit (M5's word-ownership scheme),
(3) skipping emit for chunks that will be stored anyway. Ratio is on target:
greedy twin = GPU = 1.97× vs `lz4_flex` 2.04× on Silesia (3.4% gap; target ≤ 10%).

## M4 — Validation matrix is the full 3 × 3 grid, plus ranges and chunk sizes
`crates/gpu/tests/matrix.rs` runs every encoder (CPU `lz4_flex`, CPU greedy, GPU) ×
every decoder (CPU hand-written, CPU `lz4_flex`, GPU) on every backend that has an
adapter. It's a superset of the plan's table, since the CPU-encoded files also go
through `lz4_flex` and GPU-encoded ones through all three decoders. It covers chunk
sizes of 4 KiB, 64 KiB and 1 MiB (the 1 MiB chunks exercise the 65 535-byte offset
cap, with repeats at 40 KB and 70 KB) and four range reads per combination. Inputs
are the M1 fixtures, a 3 MiB text, and the Canterbury corpus. CI fetches Canterbury
(`scripts/fetch_corpus.* canterbury`), and local runs skip it with a note if it's
missing. A second test requires GPU files to equal CPU-twin files on every backend.
The malformed-input suite (every truncation, every single-byte corruption of a
3-chunk file) runs against the GPU decoder on each backend.
Apple M4 Pro / Metal: 621 combinations exact. M4 changes no code paths, so it has
no benchmark run.

## Toolchain pinned to 1.99.0
`rust-toolchain.toml` said `stable`, which meant 1.95 locally and 1.99 on CI. A
newer clippy lint (`chunks_exact` with a constant size) broke CI on the M2 and M3
pushes while local checks passed. The toolchain is now pinned to `1.99.0` (CI
installs it from the file), and the two flagged sites use `as_chunks`. Bump the pin
deliberately.

## M5 — Cooperative decoder: windowed design rejected, hybrid adopted
First design (rejected): invocation 0 parsed windows of 64 sequence headers into
workgroup memory, then all invocations copied literals, then matches in dependency
groups (a match joins the group if its source pattern lies before the group's first
match; overlapping matches copied in parallel as periodic patterns). Correct, but on
short-sequence data every invocation walked every sequence, and text needs a group
barrier every few sequences: synthetic text fell from 7.4 GB/s (naive) to 0.8–1.3.

Adopted (`lz4_decode_coop.wgsl`): invocation 0 decodes exactly like the naive kernel
(same checks, order and statuses) and hands only copies of ≥ `long_copy` bytes to the
whole workgroup (barrier, word-wise copy, barrier). Overlapping matches still copy in
parallel via the periodic formula `dst[m − off + k mod off]`.

Kernel GB/s, Apple M4 Pro, 256 MiB inputs, 64 KiB chunks:

| kernel | zeros | random | text | mixed | Silesia |
|---|---:|---:|---:|---:|---:|
| naive (M2) | 31.8 | 42.7 | 7.4 | 8.3 | 1.95 |
| hybrid wg16, long_copy 16 | 20.7 | 49.5 | 1.9 | 4.6 | 3.13 |
| **hybrid wg32, long_copy 16** | 33.2 | 52.5 | 1.9 | 4.9 | **3.23** |
| hybrid wg64, long_copy 16 | 34.0 | 54.5 | 1.1 | 2.9 | 1.93 |

Default: hybrid, workgroup 32 (one Apple SIMD group), `long_copy` 16. On Silesia,
the real-data benchmark, it's 1.66× the naive kernel (kernel) and 1.35× end to end
(1.74 vs 1.29 GB/s), which meets M5's "clear speedup over M2". **Regression:**
uniform short-sequence data (synthetic text) is ~4× slower. The naive kernel runs
32 *different chunks* in the 32 lanes of a SIMD group, in lockstep, while one
workgroup per chunk runs the serial parse on 1 lane of 32. Even with no cooperative
copies at all (`long_copy` = 1 MiB) the hybrid gets 1.9 GB/s on text. Naive stays
available (`DecodeKernel::Naive`). The structural fix is M6's GLZ, whose separate
streams remove the serial token parse.

## M5 — atomicOr vs word ownership: a tie; atomicOr kept
Both write schemes assemble whole output words. Option 1 merges every word with
`atomicOr`. Option 2 used `atomicStore` for words entirely inside one copy, and
`atomicOr` only at shared edges. Across workgroup sizes 16–64 and thresholds
16–128, they were within ±2% of each other on every input. The simpler option 1
stays (option 2 removed), and it's what M3's emit already uses.

## M5 — Encoder fixes folded in from the M3 profile
(1) The parse now totals the block's encoded size, from register values (an earlier
version re-read storage and cost text ~15%). If the block won't shrink the chunk,
the workgroup skips emit: random input 0.49 → 0.68 GB/s. `encode_blocks` reports
such chunks as `EncodedBlock::Incompressible { size }`. (2) Literal runs ≥ 64 bytes
are copied by the whole workgroup, word by word, and owners write only headers and
short runs. Text 0.79 → 0.95 GB/s. Silesia is unchanged (~0.7 GB/s), because the
serial parse dominates there. The parallel parse is still the next encoder step.
Output remains byte-identical to the CPU twin.

## M6 — GLZ v1 (fixed-width fields) rejected; v2 uses tokens + an extension array
The plan suggested fixed-width per-sequence fields. With u16 literal length, match
length and offset (6 bytes per sequence, vs LZ4's ~3), Silesia compressed to **1.35×**
vs LZ4's 1.97× with the same parse: short matches cost more in fields than they saved.
GLZ v2 keeps LZ4's 4-bit nibbles in a token byte and moves 15-escaped remainders to a
separate extension array (u16, or u32 when any value needs it). A sequence's first
extension slot is an exclusive prefix sum of escape counts, so decoding stays fully
parallel, and a typical sequence costs 3 bytes. Silesia: **1.93×** (LZ4 greedy
twin 1.97×, `lz4_flex` 2.04×). Zeros: 565× (LZ4 224×), because long runs need one
extension value instead of 255-byte continuation chains. Spec: `format/FORMAT.md`.

## M6 — Dependency elimination: capped matches, not rejected ones
With a single candidate per position, Gompresso-style rejection of matches whose source
overlaps a group mate's output cost far more ratio than Gompresso's ≤ 10%: Silesia 1.75×
(G = 8), 1.58× (G = 32), 1.45× (G = 128) vs 1.93×. Rejecting also made the serial parse
quadratic in repeated regions, because each rejected candidate was fully extended first.
The parse now **caps** such a match at the room before the first conflicting output
(binary search over the group's sorted, disjoint outputs), and drops it only if the
cap is below 4. Extension is bounded by the cap. G = 64: Silesia 1.55×. GPU parse
throughput went 0.06 → 0.40 GB/s on Silesia (no-group parse: 0.56). The group list
lives in workgroup memory, and groups are capped at 64 (the GLZ decode block). CPU and
GPU parses stay byte-identical, which the tests check for G ∈ {none, 1, 4, 64} and in
the matrix.

## M6 — GLZ GPU decoder
One workgroup per chunk, one invocation per sequence, 64 sequences per step: scan of
escape counts, then lengths, then saturating scans for literal sources and output
positions, then per-sequence validation (lowest failing index wins via `atomicMin`, so the
status equals the serial CPU decoder's), then literals in parallel, then matches in
dependency rounds. Each match computes once a 64-bit mask of the earlier matches
in the block that overlap its source pattern, and copies when `mask & pending == 0`.
(An earlier attempt waited only for the *latest* overlapping match. That's wrong,
because an earlier overlapping match can still be pending behind a longer chain, and
proptest caught it.) Runs ≥ 32 bytes are copied by the whole workgroup through a
compact `atomicAdd` list, the M5 lesson. Before these two changes GLZ decode ran at
1.84 GB/s on Silesia, slower than LZ4 cooperative.

## M6 — Results (Apple M4 Pro, 256 MiB synthetic, Silesia 202 MiB, 64 KiB chunks)

| | ratio | GPU decode kernel | GPU decode e2e | GPU encode kernel | CPU decode mt |
|---|---:|---:|---:|---:|---:|
| LZ4 (`lz4_flex` file) | 2.04× | 3.21 (coop) | 1.77 | — | 13.2 (`lz4_flex`) |
| LZ4 (greedy twin / GPU) | 1.97× | — | — | 0.72 | — |
| **GLZ** | 1.93× | **4.34** | **2.01** | 0.71 | 7.1 (reference) |
| **GLZ g64** | 1.55× | **6.55** | 2.30 | 0.39 | — |

GLZ beats LZ4 on GPU decode for every input except the uniform synthetic text, where
LZ4's naive kernel still leads: text 3.0 vs LZ4 coop 1.9 and naive 7.3; mixed 8.2 vs
4.8. That's at 2% less ratio than the GPU's own LZ4. Dependency elimination adds +51%
decode speed for −20% ratio on Silesia, so it stays an option, not the default. End
to end everything is still bound by 6–10 GB/s transfers. GPU compression (~0.7 GB/s)
remains the weak spot, and the serial parse is still next.

## Encoder split into three kernels (occupancy fix)
Profiling the fused encoder (one workgroup per chunk doing match finding, parse and
emit) showed that occupancy was the problem, not the amount of work. On Silesia, shrinking only
the hash table 16 KiB → 4 KiB made the kernel 2.6× faster (0.73 → 1.91 GB/s), and
the *unchanged* serial parse phase alone 2.8× faster (0.84 → 0.30 s/GB; 60% of kernel
time). The parse runs on one lane of each workgroup, so the number of parses in flight
equals the number of resident workgroups, and the 16 KiB table capped that at a few
per core. Bigger chunks were also *faster* (1 MiB: 1.16 GB/s), because of fixed
per-chunk costs (table zeroing, per-block barriers).

The encoder is now three kernels in one compute pass, sharing one explicit bind-group
layout (seven bindings), so each kernel's occupancy depends on its own needs:
1. `encode_matches.wgsl`: one workgroup per chunk, the only kernel with the table.
2. `encode_parse.wgsl`: **one invocation per chunk, 64 chunks per workgroup, no
   workgroup memory**, so every SIMD lane parses its own chunk (the M5 lesson). GLZ's
   dependency-group lists moved to a per-chunk slice of global memory
   (`chunk × MAX_GROUP`): a private array would be allocated per invocation.
3. `lz4_emit.wgsl` / `glz_emit.wgsl` (+ `encode_emit_shared.wgsl`): one workgroup per
   chunk; the parse's results come through a per-chunk `chunk_info` buffer.
Each kernel is its own shader module (common prelude + kernel), which also sidesteps a
naga 30 assertion seen with pipeline constants for overrides an entry point doesn't use.
Output is unchanged, and it's byte-identical to the CPU twins in every existing test.

Kernel GB/s, before → after: Silesia LZ4 0.72 → 1.25, GLZ g64 0.39 → 1.07, text
0.75 → 1.69, zeros 0.59 → 2.33, random 0.72 → 4.89. Match finding (~0.44 s/GB when
measured fused) is now roughly half the kernel. End to end rose only 0.48 → 0.69 GB/s.
Host work is now as large as the kernel: allocating a 4×-input scratch buffer per call,
reading back whole output slots (≈ 1.0× input, though the result is ~0.5×), and packing
on the CPU. Next: GPU-side packing so only compressed bytes are read back, reusable
buffers, then faster match finding (word-wise compares, larger workgroups).

## M7 — Filters: CPU selection, GPU inverse, forward GPU kernels for the encoder
Built by a parallel agent in a separate copy of the repo, then merged and re-verified
(249 tests, clippy and fmt clean).
- **Selection** (`FilterMode::Auto`): each chunk is compressed under the level's
  candidates (level ≤ 1: none, shuffle-4, delta-4; level ≥ 2 adds widths 2 and 8), and
  the strictly smallest block wins, so ties go to the lower (filter id, width). If no
  candidate's block is smaller than the chunk, the chunk is stored raw **unfiltered**.
  Checksums cover the unfiltered bytes. Decoders apply the inverse after any payload,
  stored ones included: a filter on a stored chunk is legal but never written. With
  `Auto`, `CompressOptions.level` now chooses the candidate set (it used to be only
  informational).
- **Format plumbing**: `format::assemble_filtered(header, payloads, filters)`; `assemble`
  wraps it with all `None`. `ChunkPayload` is unchanged, so the GPU encoder kept
  compiling during the parallel work. It switches to `assemble_filtered` when GPU
  selection lands.
- **GPU**: `gpu::filter::FilterKernels` runs forward or inverse shuffle and delta over
  (src range → dst range) jobs, one 256-invocation workgroup per job, each output word
  written by one owner. Delta's inverse is a workgroup prefix scan. The decoder decodes
  as before, copies filtered chunks into a compact scratch buffer (adjacent chunks
  merged into one copy), and runs the inverse back into the output. Shuffle can't be
  done in place.
- **Results** (agent's run, noisy GPU): sorted u32 1.00 → 2.66×, u64 timestamps
  1.58 → 3.04×, f32 points 1.00 → 1.44×, i16 audio 1.00 → 1.06×, Silesia 1.97 → 2.03×
  (+3%). Shuffle-4 wins every numeric chunk and ~10% of Silesia's; delta-4 wins ~1%.
  Alone, delta leaves random low bytes between zero runs too short for LZ to match;
  delta *then* shuffle is the obvious M11 follow-up. Trying 3 candidates cuts CPU
  compression throughput by about 2.7–4.6× (Silesia 0.95 → 0.35 GB/s), so the plan's
  cheap estimator is worth trying at level 1. The GPU inverse runs at 21–67 GB/s alone
  and doesn't measurably change Silesia's decode time.

## GPU-side packing and reusable encoder buffers
Built by a parallel agent in a separate copy, then merged and re-verified. End-to-end
GPU compression (0.69 GB/s on Silesia) was half the kernel speed (1.25). Every call
allocated all buffers (including 4 bytes of scratch per input byte), read back every
output slot (≈ 1.0× the input, although the result is ≈ 0.5×), and packed on the CPU.
`compress` now reads back only the per-chunk sizes after encoding. The host lays out
the payloads with the CPU rule (stored when the block doesn't shrink the chunk, every
payload padded to 4 bytes). A fourth kernel, `encode_pack.wgsl`, copies each block from
its slot (or the raw input for stored chunks) into one contiguous, zero-padded data
section, and only that is read back. Offsets are u32 within a batch on the GPU and
u64 across batches on the host. Checksums are computed while the GPU encodes. Buffers
are cached in the encoder and grow on demand. Bindings use exact sizes, so shaders
see the same array lengths as with fresh buffers. Output slots are cleared each encode,
and every scratch, info and size entry is written before it's read. The input goes
through a reused mapped staging buffer, because `Queue::write_buffer` allocates a fresh
one per call and filling it page-faults (about 20 ms → 3.5 ms for 212 MB).
`GpuEncoder::with_max_batch_chunks` caps batch size (the test seam for multi-batch
files, and a memory bound). Output is byte-identical. Agent's numbers (shared GPU):
Silesia end to end 0.68 → about 1.2 GB/s for LZ4 and GLZ, kernel unchanged. What
remains is about 4 ms upload, 2 ms pack and 6 ms copying into the result `Vec`, so the
kernels are the bottleneck again. Open: the encoder keeps its largest batch's buffers
(about 7.5× the batch) for its lifetime, and batches don't yet overlap (M8).

## M7 — GPU filter selection (integrated after both parallel tasks)
`GpuCompressOptions { filters: FilterMode::Auto, level }` uses `filter_candidates(level)`,
which is identical to `cpu::filter::candidates` (a test checks it). Per batch:
1. Upload once and encode the unfiltered candidate in the cached buffers.
2. For each other candidate, run `FilterKernels` forward from the input buffer into a
   filtered copy, then run the three encode kernels on it into that candidate's own
   output slots (one submission per candidate). Read back only each candidate's sizes.
3. On the host, choose per chunk the strictly smallest block, earlier candidate on ties.
   A chunk whose best block doesn't shrink it is stored raw and unfiltered.
4. Pack one pass per candidate that won any chunk (stored chunks go with the unfiltered
   pass). The pack shader skips entries with `SKIP_BIT`. All passes write one data
   section, which is read back once.

Batches shrink by the number of candidates, since each keeps its own slots. Files are
byte-identical to the CPU's `FilterMode::Auto` files: LZ4 and GLZ, levels 1 and 2,
numeric and M1 inputs, multi-batch, in the matrix (now 13 encoders, 2,592 exact
combinations on Metal), and in the CLI (`compress --gpu --filters auto`, previously
refused). The CLI still always uses level 1; `--level` belongs to M9. A possible
optimisation: chunks where the unfiltered block already wins clearly could skip the
other candidates, which needs the estimator the plan mentions.

## Faster match finding: word compares, block 128, fixed emit workgroup
Per-kernel profile of the split encoder (Silesia, 212 MB, block 64): match finding
76 ms (45%), parse 80 ms (47%), emit 13 ms (8%).
1. **Word-wise match extension** (`extend` in `encode_common.wgsl`, used by match finding
   and the parse): XOR two 4-byte words and count equal leading bytes with
   `countTrailingZeros / 8`, then finish byte-wise near the limit. Same lengths,
   byte-identical output. Match finding 76 → 58 ms, parse 80 → 61 ms; kernel 1.25 →
   1.60 GB/s.
2. **The emit kernel's workgroup is fixed at `EMIT_WG = 64`** instead of following
   `block`. Its scans got slower with bigger workgroups and masked the gain.
3. **Level-1 match-finding block 64 → 128**, for both the GPU encoder and its CPU twin
   (`EncodeParams::default`, `lz4::encode::Params::default`). Silesia kernel time and ratio:

   | block | kernel | GB/s | ratio |
   |---:|---:|---:|---:|
   | 64 | 134 ms | 1.59 | 1.966× |
   | **128** | **108 ms** | **1.97** | **1.943×** |
   | 256 | 101 ms | 2.09 | 1.904× |

   Bigger blocks hide in-block repeats from each other (positions only see earlier
   blocks), so ratio drops. 128 trades 1.2% ratio for +24% throughput; 256 adds only 6%
   more for another 2% of ratio. Block 64 stays available for higher levels (M9). Tests
   that hand-compute block-64 behaviour now pin `block: 64`. The agent-written filter
   test's "≥ 20% smaller" threshold became "≥ 15%" (GLZ at block 128 is 19.9% on its
   small f32 input).

Recorded run (M7-matchfind): Silesia GPU compress end to end 1.16 → 1.67 GB/s (LZ4),
1.16 → 1.72 (GLZ); kernel 1.25 → 1.95; ratio 1.97× → 1.94×. **The parse (~60 ms)
is now the largest kernel.** It runs one lane per chunk, and 64 KiB chunks give only
~3,200 lanes, too few to hide memory latency. The idea that fits next is a speculative
segmented parse: several lanes per chunk, each starting at a segment boundary. The
greedy parse re-synchronises with the true path within a few sequences, so the
results can be stitched exactly. Not done yet.

## Filter estimator: sampled trial encoding (`FilterMode::Auto`)
Exhaustive filter selection costs one full encode per candidate (3× at level 1).
Instead of a new heuristic, `Auto` trial-encodes only each chunk's leading
**sample**: `sample_len = max(4 KiB, chunk_size / 8)`, filtered as a buffer of its
own. Candidates compete on sample block size (strictly smaller wins, ties to the
earlier candidate). The winner encodes the whole chunk, which is stored raw and
unfiltered if that doesn't shrink it. It reuses the real encoders, so nothing needs
tuning, and it's deterministic and identical on CPU and GPU (`cpu::filter::sample_len`
== `gpu::encode::sample_len`). With chunks ≤ 4 KiB the sample is the whole chunk, so
`Auto` equals `Exhaustive` (tested). The old behaviour is kept as
`FilterMode::Exhaustive` / `--filters exhaustive`. On the GPU, the samples are
gathered on the host (1/8 of the batch) and encoded as chunks of `sample_len` under
each candidate. The batch is then copied once into a filtered buffer, the winning
filters are applied per chunk (unfiltered chunks keep the copy), and it's encoded and
packed once.

Recorded (M7-estimator), Silesia LZ4: Auto 1.99× at 1.07 GB/s GPU end to end vs
Exhaustive 2.00× at 0.57 (unfiltered: 1.94× at 1.72). CPU: 0.68 vs 0.39 GB/s. GLZ is
similar. Numeric inputs: Auto's ratio equals Exhaustive's to two decimals. Sampling
can be misled when a chunk's start isn't representative (a test builds such a chunk).
On real data that cost Silesia 0.5%.

## M8 — Large files and streaming
**Chunk table reserved up front, not a footer.** The input length is known before
compressing (`compress_stream(reader, len, …)`), so `chunk_count` is too. The
streaming encoder writes a placeholder header and table, streams each batch's data
after them, then seeks back and writes the real table. Format v1 is unchanged and the
table stays O(1) to locate for range reads. A footer table was rejected: it would need
a format change, and either a trailing pointer or a read from the end of the file. The
cost is that output must be seekable (a file, not a pipe).

**Memory model.** Batches are sized by a GPU memory budget (`with_memory_budget`,
`--gpu-memory`, default 3 GiB, see below) as well as by the binding limit.
- Encoder: `encoder_bytes_per_chunk` = 8 × chunk (input, match scratch, filtered
  copy, staging) + one output slot + 32, plus the group table for GLZ g-modes, plus
  per-candidate sample buffers (`Auto`) or full candidate copies (`Exhaustive`).
  Batch = min(binding limit, budget / bytes per chunk).
- Decoder: `batch_bytes` = padded payloads + 3 × output (output, inverse-filter
  scratch, readback staging) + 20 B per chunk. Batches are planned at **budget / 2**
  because two are in flight.

**Decoder pipelining is 2-deep.** Batch i+1 is uploaded and dispatched before batch i
is read back, so upload, kernel and readback overlap across batches. `decompress`,
`decompress_range` and `decompress_stream` share one loop. **The encoder is not
pipelined yet.** Each batch waits for its sizes, then for its packed data, before the
next starts. That's a candidate follow-up.

**4 GiB round trip** (`cargo test --release -p gpu --test stream -- --ignored`):
passed in 13.8 s, process peak RSS 483 MB, with a 512 MiB GPU budget.

**Finding: small budgets hurt compression, not decompression.** Silesia (212 MB) via
the CLI: compress 0.26 s at 1 GiB vs 0.74 s at 128 MiB. Decompress 0.18 s vs 0.24 s.
The parse kernel runs one lane per chunk, so a 128 MiB budget's batches (~217 chunks)
leave it badly under-occupied.
A segmented parse (several lanes per chunk) would also make small budgets cheaper.

**The same effect shows at the default budget.** The recorded M8 run reads lower than
M7-estimator on GPU compress end to end (Silesia LZ4 1.72 → 1.19 GB/s, Auto 1.07 →
0.64), with kernels unchanged. At 1 GiB, Silesia (212 MB) splits into 2 batches and the
256 MiB synthetics into 3, where M7 ran one. In-process A/B on Silesia, same binary,
interleaved runs:

| Budget | Batches | LZ4 none | LZ4 auto |
|---|---|---|---|
| 4 GiB | 1 | 1.74 GB/s (122 ms) | 1.10 |
| 2 GiB | 1 | 1.72 (123 ms) | — |
| 1 GiB | 2 | 1.20 (177 ms) | 0.64 |
| 512 MiB | 4 | 0.78 (273 ms) | 0.47 |

Each extra batch costs ~55 ms, about one whole parse kernel. The parse is
latency-bound at one lane per chunk, so a half-size batch takes nearly as long as a
full one. Streaming bounds memory, but throughput then depends on batch count until
the parse gets more parallelism.

**Decision: default budget 3 GiB** (was 1 GiB while developing). That's enough to
encode 256 MiB in one batch, with or without `Auto` filters (~590 KB / ~687 KB of GPU
memory per 64 KiB chunk), so benchmark-sized inputs run as fast as M7. A unit test
pins this. `Exhaustive` (~850 KB per chunk) still splits 256 MiB in two. 2 GiB was
considered: it fits plain Silesia only. Larger files still pay ~55 ms per extra
batch. The real fix is more parse parallelism (segmented parse), not a bigger budget.
`--gpu-memory` lowers the cap on smaller GPUs.
Recorded M8 run (3 GiB): every Silesia GPU path is within ±1% of M7-estimator or above
it. The exception is `Exhaustive` on the 256 MiB numeric synthetics, which split into 2
batches and run 20–26% slower than M7. That's accepted: `Exhaustive` is the slow,
opt-in mode.

**Decoder: no extra copy of the output.** With 3 GiB, GPU decompress end to end was
still 9–16% below M7 (Silesia LZ4 coop 1.73 → 1.57 GB/s, GLZ 2.06 → 1.74), with
kernels unchanged and Silesia in one batch. The shared batch loop handed each batch to
a sink that copied it into a fresh `Vec`: 212 MB and ~22 ms, with first-touch page
faults. M7 returned the readback `Vec` as is. The sink now takes each batch by value,
and `append_batch` takes over the first batch without copying. Silesia
`decompress`: 138.6 → 122.3 ms (1.53 → 1.73 GB/s, equal to M7).

## Ratio and throughput research (2026-10-08)
Four research passes (GPU LZ literature, GPU entropy coding, parallel parsing and
match finding, the CPU ratio/throughput frontier), plus local measurements on Silesia
(211.9 MB) with **64 KiB independent chunks** unless noted. The measurement tools were
the lz4 1.10 and zstd 1.5.7 CLIs (`-b -B64K`), entropy bounds computed from the CPU
twin's GLZ streams, and a C simulator of our LZ4 encoder that reproduces our ratios
(1.9445× vs 1.943× recorded).

**Where the ratio goes** (LZ4 format unless noted):

| Configuration | Ratio |
|---|---|
| gpucomp level 1 (greedy, 1 candidate per slot, block 128) | 1.94× |
| + lazy parse | 2.00× |
| + in-block visibility, lazy | 2.04× |
| hash chains D=2 / 4 / 16, lazy, hash_log 12 | 2.17 / 2.28 / 2.40× |
| optimal parse, D=64, hash_log 16 | 2.47× |
| lz4 -1 / -9 / -12 | 2.07 / 2.48 / 2.49× |
| zstd -1 without literal compression | 2.31× |
| zstd -1 / -3 / -9 / -19 | 2.76 / 2.88 / 3.08 / 3.26× |
| zstd -1 at 256 KiB / 1 MiB chunks | 2.89 / 2.88× |
| lz4 -9 at 256 KiB / 1 MiB chunks | 2.66 / 2.71× |
| GLZ + order-0 entropy bound (level-1 matches; block 1 with hash_log 16) | 2.53 / 2.72× |

- **The match finder is the first limit.** It keeps one candidate per hash slot;
  `probe_len` caps extension, not probes. Most of LZ4-HC's gain comes from the first
  4–16 candidates.
- **Entropy coding is the largest single lever** (+33% for zstd -1 over lz4 -1).
- **Larger chunks matter less** until the match finder is strong, then cost 9–10%.

**Throughput.** The one-lane-per-chunk parse was ~60 of ~110 ms. Greedy and lazy
parses decide each step from the current position only, so they can be computed
exactly in parallel. How quickly a greedy walk started at a segment boundary lands on
the true path (Silesia, level-1 matches): median 4 B, p90 28 B, p99 157 B, p99.9
4.7 KB, max 9.9 KB. A fix-up loop is therefore needed for the tail. Published GPU
entropy coders run at tens to hundreds of GB/s (DietGPU, Tian et al., GDeflate's
32-lane layout), so an entropy stage shouldn't be the bottleneck. GPU LZ compression
is hard everywhere: nvCOMP LZ4 compresses text-like data at about 4.6 GB/s on an A100,
which has several times the M4 Pro's bandwidth.

**Decision:** reorder the roadmap (plan.md M9, M9e). Order: exact segmented parse,
lazy rule, in-block visibility, hash chains as levels, then GLZ v2 with an entropy
stage. Chunk size and typed filters come later.

## Segmented parse: exact, 32 lanes per chunk
`encode_parse_seg.wgsl` replaces the one-lane-per-chunk parse for LZ4 and for GLZ
without groups. GLZ with dependency elimination keeps the serial kernel, because
its group state depends on parse history.

Without groups, a greedy step depends only on its position, so the serial parse can
be computed exactly with one workgroup per chunk and one lane per segment
(`PARSE_SEGMENTS` = 32, `seg_len(n)` positions each):
- **Phase A, speculative walks.** Each lane walks from its segment start, marks the
  positions it visits (bit 31 of the match word, so `probe_len` is now ≤ 32767),
  and records its exit.
- **Phase B, fix-up rounds.** Each lane re-walks from the previous lane's exit until
  it meets a mark (its speculative walk is then correct from there) or leaves the
  segment. Rounds repeat until no exit changes.
- **Phase C, final walk.** Each lane writes its sequences in place in its own
  segment's scratch words (the j-th match starts ≥ 4j past the segment start).
  A scan then fixes each first sequence's anchor.

The output is byte-identical: all existing twin and matrix tests pass, plus new
stitching tests in `tests/parse.rs`.

**Emit** looks up sequence i through a per-chunk table of each segment's first
sequence index (`segs`, 32 × u32 per chunk; the serial parse writes a one-segment
table). The first version did this lookup, a binary search, in every lane for every
sequence of the long-literal loop, and emit went from 13 to 69 ms. Caching each
lane's address in workgroup memory brought it to about 16 ms.

**Long runs.** Inside a run, every lane's walk lands in the same match, and each
lane extended it to the run's end: zeros fell 4.89 → 1.45 GB/s. Fixes:
- Speculative extension stops 256 B past the segment. Such an exit stays "unknown"
  until the lane is confirmed on the true path.
- A confirmed long match is extended by the whole workgroup (`coop_extend`: 16 B
  per lane per step, minimum first mismatch).
- The final walk takes a crossing match's end from the known exit instead of
  extending again.

After these, zeros runs at 4.59 GB/s. With one lane per chunk, M8's serial parse kept
the SIMD lanes busy on separate chunks, so zeros stays slightly below it.

Recorded (parse-seg vs M8), GPU compress, Silesia:
- LZ4 kernel 1.98 → 2.72 GB/s (+38%), end to end 1.74 → 2.29 (+32%).
- GLZ end to end 1.73 → 2.28.
- `--filters auto` 1.09 → 1.41; `exhaustive` 0.58 → 0.78.
- Single Silesia files gain 2–7× (e.g. `xml` kernel 0.12 → 0.78). Small files used
  to leave the serial parse too few chunks to run side by side.
- Synthetics: mixed +7%, random ±0, text −2%, zeros −6%; GLZ g64 (serial parse)
  unchanged.

Kernel split on Silesia: match finding ~34 ms, parse ~28 ms (was ~60), emit ~16 ms.
Match finding is now the largest kernel.

## Lazy parse at level 1
Rule (`cpu::lz4::encode::defer_match`, mirrored in both GPU parse kernels): at a
position with a match, emit a literal instead if the next position's **phase-1**
(probe-capped) match is longer. The step still depends only on position (it reads
match words p and p + 1), so the segmented parse stays exact. In phase C, a lane's
read at p + 1 == seg_end is cached before the barrier, because the next lane
overwrites that word with sequences. Lazy applies to GLZ too, including with groups
(the deferral comes before the group check). `Params::lazy` / `EncodeParams::lazy`
default to true; greedy stays available.

Using the capped lengths is nearly free and almost as good as comparing fully
extended matches (Silesia CPU twin: greedy 1.9447×, lazy capped 1.9947×, lazy full
1.9964×).

Recorded (lazy vs parse-seg), Silesia:
- LZ4 1.943 → 1.993× at the same speed (kernel 2.72, end to end 2.29 → 2.31 GB/s).
- GLZ 1.908 → 1.955×.
- `--filters auto` 1.993 → 2.045×, above lz4_flex's 2.036×.
- Decode gets faster with fewer sequences: GLZ kernel +5–11%, LZ4 auto-filtered
  decode +6–8%.
- Text 1.816 → 1.922× (+5% compress speed).
- GLZ g64 (serial parse) compresses 3–5% slower for +1.3% ratio.

## In-block visibility: not worth it at level 1 (negative result)
Positions in a 128-position match-finding block don't see each other. The CPU twin
says full visibility (a plain sequential hash table, so the output wouldn't depend
on block size) would raise Silesia from 1.993× to 2.041× with the lazy parse. That
was implemented exactly, and it passed every twin test. GPU kernel times:

| Variant | Silesia ratio | Silesia kernel | text 256 MiB kernel |
|---|---|---|---|
| none (level 1 today) | 1.993× | 77.8 ms | 57.8 ms |
| scan back to the first lane sharing the low 8 hash bits (exact) | 2.039× | 267 ms | 230 ms |
| same tables, scan disabled | 1.993× | 91.7 ms | 72.1 ms |
| window K=1 (recompute the hash of p−1) | 2.010× | 82.8 ms | 69.0 ms |
| window K=4 | 2.015× | 97.0 ms | 75.8 ms |
| window K=8 | 2.018× | 102.8 ms | 91.7 ms |

- **The exact scan is divergent.** Each 32-wide SIMD group waits for its longest
  scan, so it costs the maximum, not the average.
- **Its extra memory and barrier also cost ~14 ms.** The first-lane table, the
  hash array and one more barrier take workgroup memory past 16 KB.
- **Bounded windows cost more speed than they add ratio** at every K.

Finding the previous equal key across a 128-lane block cheaply needs subgroup
shuffles or a sort, neither of which is worth it here.

**Decision:** keep block semantics. Hash chains get their ratio from older
candidates, and the parsing research found that without in-block links they lose
only ~2% with the lazy parse (D=4: 2.233× vs 2.277× linked).

## Hash chains and levels (M9)
**Semantics** (CPU twin `find_matches`, GPU `encode_matches.wgsl`):
- A position's first candidate is the table entry: the latest position with its hash
  from an earlier block.
- With `depth` > 1, the position also stores that entry as its chain link
  (`chain`, one u32 per input byte). It then follows links to older candidates,
  up to `depth` of them. Links skip same-block positions, as the table does.
- The longest match wins; ties go to the nearest candidate (strictly longer
  replaces). A candidate beyond 64 KiB ends the walk.
- GPU shortcuts that keep the result exact:
  - skip extending a candidate whose byte at the current best length differs;
  - stop once the best match reaches the probe cap;
  - issue the next link's load before extending.

  The skip and the early stop cut depth 16 from 472 to 355 ms on Silesia. Depth 2
  and 4 barely change. Their cost is the dependent global loads along the chain.
- `storageBarrier()` between blocks makes links visible to later blocks. It's only
  compiled in with DEPTH > 1.

**Level 1 keeps its own code path.** The general loop at DEPTH = 1 cost 6–7% on the
GPU (Silesia kernel 2.72 → 2.55 GB/s in the first recorded run) even with the barrier
compiled out, so DEPTH = 1 runs the original single-candidate code. The CPU twin
likewise (within 2.5% of before, single-threaded).

**Levels:** `Params::for_level` / `EncodeParams::for_level` map 1/2/3 to depth 1/4/16
(clamped), and `--level` in the CLI applies to the GPU encoder and its CPU twins. The
level is stored in the header, and from level 2 it also widens the filter candidates
(M7).

Recorded (M9-levels), GPU LZ4, 64 KiB chunks:

| Level (depth) | Silesia ratio | kernel GB/s | end to end GB/s | text ratio |
|---|---|---|---|---|
| 1 (1) | 1.993× | 2.71 | 2.27 | 1.922× |
| 2 (4) | 2.212× | 1.30 | 1.20 | 2.261× |
| 3 (16) | 2.319× | 0.61 | 0.56 | 2.612× |

lz4_flex (multi-threaded CPU) is 2.036× at 5.3 GB/s for reference. Level 2 beats it
on ratio by 9% but is far slower, so the deep levels need a faster candidate search.
Next is multi-way buckets in workgroup memory: no dependent global loads, though
fewer hash bits. Depth-2 numbers (2.118×, ~1.8 GB/s kernel) are in
the CPU/GPU measurements above if a level between 1 and 2 is wanted.

## M9e step 1: GLZ-E format and CPU codec
**What gets entropy-coded.** Two designs were estimated on Silesia (order-0 entropy
plus per-chunk table and lane overheads, 64 KiB chunks):
- **A:** Huffman-code GLZ's byte streams as they are: tokens, offset low bytes,
  offset high bytes and literals, with extension values left raw. Level 1 2.482×,
  level 3 2.727×.
- **B:** zstd-style log2 bucket codes for lengths and offsets, plus raw extra bits.
  2.500× and 2.759×.

A gives up about 1% and keeps GLZ's arrays. Decoding is "entropy-decode back into
the exact GLZ block, then decode GLZ", so sequence checks, error codes and the GPU
GLZ decode kernel are reused. On the encoder side it's one transcoding pass after
GLZ emit. **Chosen: A**, as a new codec 3 (GLZ-E). GLZ (codec 2) is unchanged.

**Streams** (`cpu::huffman`, spec in FORMAT.md): raw, RLE or Huffman.
- **Huffman:** canonical codes of at most 11 bits, stored as 128 bytes of
  nibble lengths. Up to 32 lanes (one per 512 symbols), each a contiguous symbol
  range with its own word-aligned, LSB-first bitstream. No atomics or bit
  interleaving in either direction. The 11-bit limit gives a 2048-entry decode
  table.
- **Table construction** is integer-only with fixed tie-breaks: sort by (count,
  symbol), two-queue Huffman taking the leaf on ties, JPEG Annex K.3 to limit lengths,
  then reassign lengths in sorted order. That way the GPU encoder can reproduce it
  exactly.

**Measured** (CPU encoder, Silesia, 64 KiB chunks): GLZ 1.95 / 2.16 / 2.26× → GLZ-E
2.50 / 2.68 / 2.75× at levels 1 / 2 / 3. The real Huffman codes landed on the
estimates, and level 3 matches zstd -1 at the same chunk size (2.76×).

**Cost on tiny or highly compressible chunks:** every stream stays raw, which adds
up to 6 words per chunk over GLZ (the literal count, four mode words, padding). On
4 KiB chunks of trivially repetitive text, GLZ-E was 10% larger than GLZ. A test
bounds this.

The GPU encoder and decoder still reject codec 3; they're steps 2 and 3.

## M9e step 2: GLZ-E GPU decoder
**Design:** two kernels in one pass.
1. **`glze_decode.wgsl`:** one 32-lane workgroup per chunk rebuilds the chunk's
   exact GLZ block in a zeroed image buffer (bytes merged with `atomicOr`).
   - The host sizes each image from the chunk's 12-byte header
     (`plan::glze_layout`). A stored chunk's image is its payload; an invalid
     header gives 0, and the kernel reports the error.
   - Per stream, invocation 0 decides everything (mode, sizes, table validity,
     canonical codes, lane layout) and broadcasts it, so barriers stay uniform.
     The workgroup then copies raw streams, fills RLE streams, or fills a
     2048-entry decode table and decodes one Huffman lane per invocation.
2. **The unchanged GLZ kernel** decodes the images.

**Statuses:** the entropy kernel writes its own status buffer, and the host prefers
its error over the GLZ kernel's, which decoded a partial image (it bounds-checks
everything, so that's safe). Statuses 7–10 are new (stream truncated / bad mode / bad
table / lane overrun). Header errors reuse Truncated / BadSequence / SizeMismatch in
FORMAT.md's order. All 17 malformed-block cases give the same status as
`cpu::glze::decode_block`.

**Memory:** batch planning budgets each GLZ-E chunk's image at its uncompressed
size, and `prepare` checks the real total against the binding limit.

**Recorded (M9e-decode), Silesia:**

| | GLZ (1.955×) | GLZ-E (2.503×) |
|---|---|---|
| CPU compress, multi-threaded | 1.18 GB/s | 0.91 |
| CPU decode, multi-threaded | 7.20 | 2.89 |
| GPU decode, kernel | 4.87 | 2.92 |
| GPU decode, end to end | 2.10 | 1.64 |

Text: 1.92× → 3.01×. Mixed: 1.97× → 2.25×.

**The entropy stage is slow.** It costs ~29 ms on Silesia, about 7 GB/s, against tens
of GB/s for published GPU entropy decoders. Likely causes:
- invocation 0 validates and builds canonical codes serially (256 symbols × 4
  streams per chunk);
- table filling is unbalanced (a 1-bit code is 1024 entries written by one lane);
- ~6 barriers per stream;
- byte-wise `atomicOr` output.

Fixes to try: parallel code assignment (a prefix sum over lengths), filling the
table by index instead of by symbol, decoding 2 symbols per lookup, word-assembled
output, and decoding all four streams' tables at once. Random data (all chunks stored)
also pays an extra copy into the image (GLZ-E decode kernel 21 vs 57 GB/s).

## M9e step 3: GLZ-E GPU encoder (codec 3 complete)
**Design.** GLZ emit already writes each chunk's GLZ block into its output slot. For
GLZ-E:
- the slots are copied into the match scratch (free after emit, 4 bytes per input
  byte, always larger than the slots);
- `glze_encode.wgsl` (one 32-lane workgroup per chunk) transcodes each block back
  into its slot and sets `sizes[chunk]`.

Everything downstream is unchanged: sizes readback, stored-chunk decisions, packing,
`encode_blocks`, and filter-candidate selection (which compares GLZ-E sizes, as the
CPU does).

Details:
- **`params.glze`** makes GLZ emit write every block, including ones not smaller than
  the chunk, because the CPU transcodes every block.
- **`slot_for(GlzE)` = 1.5 × chunk + 64.** The largest GLZ block is about 1.27 ×
  chunk + 20 (literal runs of 15 each need an extension value, wide extensions cost
  4 bytes), and GLZ-E adds at most ~32 bytes.
- **No new binding.** A separate copy buffer would have been a 9th storage buffer,
  over WebGPU's default limit of 8.

**The kernel** mirrors `cpu::huffman::encode_stream` exactly:
1. a histogram with workgroup atomics;
2. a parallel rank sort by (count, symbol);
3. invocation 0 builds the two-queue tree (leaf first on ties), applies the JPEG K.3
   fold, hands out lengths in rank order and assigns canonical codes;
4. per-lane bit counts decide Huffman against raw (strictly smaller wins);
5. every invocation writes whole words it owns: table words, lane sizes, its lane's
   bitstream.

Tests show byte-identical output to the CPU twin at levels 1/3, 4/64 KiB chunks and
all three filter modes, in the matrix and in the CLI.

**Recorded (M9e), Silesia:**
- GPU GLZ-E compress: 2.503× at 1.48 GB/s kernel, 1.36 end to end.
- GPU GLZ compress: 1.955× at 2.67 / 2.26.
- Text: 3.01× at 1.34 end to end.

GLZ-E level 1 now beats LZ4 levels 2 and 3 on both axes (2.21× at 1.20, 2.32× at
0.56). That makes the entropy stage the better use of GPU time than more match
candidates, at least until (a) makes chains cheaper.

**The encode stage is slow** (~64 ms on Silesia, ~3.3 GB/s). Likely causes:
- histogram atomics contend on skewed streams (32 lanes hitting a few bins);
- three passes over each stream's symbols (histogram, bit counts, encode) with
  byte reads;
- invocation 0's serial tree build, about 10 barriers per stream;
- 32 lanes per chunk.

Fixes to try: per-subgroup sub-histograms, fusing the bit count into the histogram
(Σ count × length), encoding all four streams' tables at once, and more
invocations per chunk for the long literal stream.

## (a) Faster deep match search: buckets (rejected) and chain rings (kept)
**Multi-way buckets in workgroup memory, rejected.** Simulated on the CPU twin
(Silesia, lazy parse):
- 4 ways at hash_log 10 (16 KB): 2.065×. Chains of 4 (hash_log 12) give 2.214×.
- 8 ways at hash_log 10 (32 KB): 2.169×.
- 16 ways at hash_log 9 (32 KB): 2.179×.

Fitting the ways into workgroup memory costs more hash bits than the extra candidates
win back.

**Chain rings, same output and speed, less memory.** A chain walk from the current
head only visits block winners (the latest position with the hash in its block), and
each winner's link is the head it displaced. So per hash slot, a ring of the last
DEPTH − 1 displaced heads holds exactly the chain's candidates, newest first:
- each (slot, block) winner pushes its displaced head once;
- the ring and its push count sit in DEPTH contiguous words per slot (in the
  `chain` binding), cleared by the workgroup at chunk start.

Output is byte-identical (all chain twin tests). Lookups read one slot's contiguous
words instead of following DEPTH − 1 scattered links, but kernel time didn't move:
2 / 4 / 8 / 16 candidates take 121 / 169 / 249 / 363 ms against 117 / 162 / 239 /
355 ms before. **Link latency was never the cost.** Each extra candidate costs ~27 ms
of verification (unaligned input loads at random earlier positions) plus SIMD
divergence. The ring is kept for memory: `(1 << hash_log) × depth` words per chunk
instead of 4 bytes per input byte, i.e. 64 KB instead of 256 KB per 64 KiB chunk at
level 2, and 256 KB instead of 4 MB per 1 MiB chunk at level 3.

**Next idea, not done:** evaluate deep candidates only at the ~40% of positions the
parse visits (literal positions, match starts and their lazy look-ahead). That needs
the per-position chain links back, plus memoising the result in the match word
inside the segmented parse.

## GLZ-E entropy stages: occupancy and serial work
Profiled by temporarily patching the shader source at pipeline creation to skip
stages, then timing the dispatch on Silesia (212 MB).

**Encode (65 ms → ~18 ms).**
- **Before:** one 32-lane workgroup per chunk ran every stream in turn, with the
  Huffman tree built on invocation 0 while 31 lanes waited. That build alone cost
  ~30 ms; the bit-count and encode passes 17 ms; the 256×256 rank sort 8 ms; the
  histogram 4.5 ms.
- **Now three kernels:**
  1. `glze_hist`: word-wise histograms of all four streams (the offset words feed
     both offset streams), then a bitonic sort of `count << 8 | symbol` keys gives
     the rank order.
  2. `glze_build`: **one invocation per (chunk, stream)** builds the tree and
     canonical codes in global build data, so ~13,000 independent serial builds hide
     each other's latency (~2 ms).
  3. `glze_encode`: 128 invocations, a 32-lane group per stream, with all four
     streams at once.
- **Build data lives in the group buffer,** unused by GLZ-E, because the encoder was
  already at WebGPU's limit of 8 storage buffers. That needed **`group_buf` to
  become `array<u32>`.** As `array<vec2<u32>>`, a one-component store compiled to a
  read-modify-write of both words, and neighbouring invocations writing neighbouring
  words lost updates (68 symbols got length 0 on a random chunk). The serial GLZ
  parse now stores its (start, end) pairs as two words.
- **Tried and not kept as wins** (each changed nothing measurable):
  - a one-word cache in the bit-count and encode loops;
  - four streams concurrently in the write kernel.

**Decode (entropy kernel alone, 29 ms → 17 ms).**
- The kernel is one 32-lane SIMD group per chunk, and lane decoding is a dependency
  chain (table lookup, then the next peek).
- **What didn't help** (both kept as harmless):
  - a two-word bit reader;
  - collecting output bytes into words (one `atomicOr` per word instead of per
    byte).
- **What did:**
  - halving the decode table to 16-bit entries, two per word, filled with atomics
    into a zeroed table: 29 → 20.3 ms;
  - packing the code lengths (a byte each) and codes (16 bits each): 20.3 → 17.2 ms.
- **Why:** workgroup memory went from ~10 KB to ~5 KB per workgroup, so more SIMD
  groups are resident per core to hide the chains' latency. **On this GPU,
  workgroup memory per SIMD group is the lever for latency-bound kernels.**
- **What's left:** lane decoding ~8 ms, table fill ~3.3 ms (unbalanced by symbol),
  invocation 0's validation and codes ~3.1 ms.

## Decoder buffer reuse
Every decode call used to allocate its GPU buffers fresh: the upload
(`create_buffer_init`), output, status, descriptor and image buffers, and a new
readback staging buffer per batch. On a 212 MB decode that's hundreds of MB of new
memory, page-faulted on first touch, every call.

Now the decoder keeps **two sets** of reusable buffers. Batches alternate between
them, so batch k + 1 never touches what batch k still has in flight.
- Bindings and clears use exact sizes, because kernels take their chunk counts from
  `arrayLength`.
- Uploads go through a reused mapped staging buffer and a GPU copy.
- `prepare_file` (the bench's kernel-only path) keeps fresh buffers.
- Test: a warmed-up decoder allocates nothing on repeat calls, single-batch and
  multi-batch.

Silesia GPU decode, end to end, same binary with and without the cache:
- GLZ: 2.06 → **3.38 GB/s** (103 → 62.7 ms).
- GLZ-E: 1.82 → **2.68 GB/s** (116.5 → 78.9 ms).
- A 64 KB file: ~3.5 ms either way. Fixed latencies dominate there; the extra
  map/poll for the upload costs nothing measurable.

## Parse segments: 32 → 128
Segments are internal to the encoder (not part of the format), so more of them just
means shorter serial walks and more lanes in flight per chunk. Level-1 LZ4 kernels on
Silesia, same session: 32 / 64 / 128 segments = 78.0 / 76.1 / 74.7 ms. **128 kept
(−4.3%).** Each chunk's segment table grows to 512 B.

Going past 32 exposed a **latent bug**: the fix-up rounds tracked "needs cooperative
extension" in one u32 bitmask (`1u << k`). For k ≥ 32 that wraps, so the wrong lane's
long match got extended. It wasn't caught at 64, and 4 parse tests failed at 128. It's
now one word per 32 segments. (The first 64-segment timing, 73.8 ms, came from the
buggy kernel.) The stitching tests now take their boundaries from `PARSE_SEGMENTS`.

## Small experiments (negative or minor)
- **Hash table size at level 1.** Silesia kernels:

  | hash_log | Table | LZ4 | GLZ-E |
  |---|---|---|---|
  | 12 (kept) | 16 KB | 1.993× in 74.5 ms | 2.503× in 93.9 ms |
  | 11 | 8 KB | 1.902× in 66.9 ms | 2.417× in 86.7 ms |
  | 10 | 4 KB | 1.798× in 65.8 ms | 2.320× in 86.6 ms |
  | 13 | 32 KB | 2.056× in 117.7 ms | 2.562× in 137.3 ms |

  Occupancy costs the match kernel ~8 ms at 16 KB, but recovering it costs 4.6% of
  ratio. 32 KB buys +3% ratio for +58% time, a worse trade than chain candidates.
- **Repeat offsets as a pure re-coding** (code an offset equal to the previous one as
  0). Only 3.6% of matches repeat the last offset, and 1.8–2.1% the one before. The
  offset streams would shrink 23.3 → 23.0 MB (~0.3% of the output). Not worth a format
  change. zstd's gain comes from a parser that seeks repeat-offset matches, which is
  path-dependent and doesn't fit the segmented parse.
- **Chain candidates compared against p's 16 bytes held in registers** (`extend16`,
  for `probe_len` ≤ 16). 2 / 4 / 16 candidates: 115.2 / 162.5 / 350.2 →
  116.6 / 159.1 / 335.0 ms. Kept (exact and harmless). The per-candidate cost is the
  candidate side's unaligned loads and divergence.
- **GLZ decode short copies a word at a time** (assemble each output word in a
  register, one `atomicOr` per word, taking overlapping match bytes from the
  register): **slower**. Silesia GLZ kernel 43.0 → 47.6 ms, GLZ-E 58.0 → 62.3 ms.
  The byte-wise atomics weren't the bottleneck; the branchier loops cost more.
  Reverted.
- **probe_len at level 1:** 16 / 12 / 8 / 4 give LZ4 1.9932 / 1.9924 / 1.9892 /
  1.9433× in 74.7 / 72.5 / 70.6 / 69.1 ms (GLZ-E 2.5025 / 2.5018 / 2.4997 /
  2.4753×). Probe 8 saves ~5% for −0.1–0.2% ratio. Not adopted: it changes
  level-1 output for little. At probe 4 the lazy rule degrades, because every capped
  length ties.

## Deep matches: separate full-occupancy pass (replaces the ring)
The ring (above) made chain lookups contiguous, but the deep candidate loop still ran
inside the match kernel, whose 16 KB hash table limits occupancy.

**Two alternatives tried, both byte-identical to the twin:**
1. **Deferred into the parse.** The match kernel stores the first candidate's match
   and a per-position link. The parse computes a position's deep match the first time
   it visits it (about 40% of positions) and memoises it in the match word: DEEP is
   bit 30, so `probe_len` ≤ 16383. **No gain:** 2 / 4 / 16 candidates in 117.5 /
   158.0 / 313.7 ms. The evaluations sit on the walks' serial path, so 2.5× fewer of
   them cost as much.
2. **A separate deep pass,** one invocation per position and no workgroup memory,
   between match finding and the parse; it memoises every position, so the parse
   finds them computed. **111.7 / 144.1 / 277.4 ms**, against 116.6 / 159.1 / 335.0
   for the ring: −4% / −9% / −17%. **Kept.**

The first extra candidate still costs ~37 ms. Reading every position's links and
random earlier input is close to memory-bound here.

**Cost:** links take 4 bytes per input byte again (the ring's memory saving is gone).
The memory model sizes batches accordingly. `deep_match` (encode_deep.wgsl) is shared
by the deep pass and both parse kernels; in the parse it normally just returns the
memo.

## Decoder: wait for each batch's own submission
`finish` used to poll with `wait_indefinitely`, which also waited for the next
batch's kernel (already dispatched), so the 2-deep pipeline overlapped little. Now:
- `dispatch` puts the output and status readback copies in the batch's own
  submission and records its index;
- `finish` waits for that submission alone (`PollType::Wait { submission_index }`);
- the reused upload staging buffer waits only for its own last copy.

Single batch (Silesia): GLZ 62.7 → 60.2 ms, GLZ-E 78.9 → 73.1 ms.

**Smaller batches to overlap transfers didn't pay off** (budgets forcing 1–8
batches): GLZ 60.2 → 56.0 ms at the smallest, but GLZ-E 73.1 → 84.5 ms. The
remaining overhead is mostly the CPU copy out of the mapped staging, which doesn't
overlap well. The default budget stays.

## Numeric data: GLZ-E with filter selection vs zstd
The bench's numeric inputs, 32 MiB each, 64 KiB chunks (GLZ-E encoded by the CPU
twin; the GPU writes the same files):

| Input | GLZ + auto | GLZ-E | GLZ-E + auto | GLZ-E + auto, level 3 | zstd -1 | zstd -3 | zstd -19 |
|---|---|---|---|---|---|---|---|
| f32-points | 1.38× | 1.14× | **1.42×** | 1.56× | 1.15× | 1.15× | 1.41× |
| i16-audio | 1.08× | 1.03× | **1.19×** | 1.24× | 1.04× | 1.04× | 1.04× |
| sorted-u32 | 1.99× | 1.29× | **4.08×** | 4.12× | 1.31× | 1.32× | 2.91× |
| u64-timestamps | 2.54× | 2.81× | 3.17× | **8.96×** | 3.90× | 3.90× | 3.90× |

Byte-shuffle groups slowly changing bytes, and the entropy stage then exploits them.
LZ alone (GLZ + auto) or zstd's general-purpose modelling can't. At level 1, GLZ-E +
auto beats zstd -19 on three of four inputs. On timestamps, level 3's deeper matches
find the structure (8.96×).

## CPU reference on this Mac (12 cores), Silesia
| | Ratio | Compress | Decompress |
|---|---|---|---|
| zstd -1 -T0 (whole file) | 2.891× | 6.47 GB/s | 1.84 GB/s (single-threaded) |
| zstd -3 -T0 | 3.201× | 3.13 GB/s | 1.67 GB/s |
| zstd -1, 64 KiB independent blocks | 2.763× | 0.74 GB/s (1 thread) | 1.64 GB/s |
| lz4 -1, 64 KiB blocks | 2.069× | 0.84 GB/s (1 thread) | 4.74 GB/s |

Multi-threaded zstd compresses ~3× faster than gpucomp's GPU GLZ-E path on this Mac,
at a better ratio (whole-file, so no random access). gpucomp's GPU decode (2.7–3.4
GB/s end to end) beats zstd's single-threaded decoder, but not a multi-threaded CPU
decode of independent chunks (lz4_flex 12.8 GB/s).

## M10: final report
**A generated report with hand-written prose.** `bench/REPORT.md` is rendered by
`bench::final_report` from `bench/REPORT.template.md`. The template holds the prose;
its `{{section}}` placeholders (setup, versus, compress, decompress, per-file,
synthetic, numeric, profile, platform) are filled from the newest corpus run of each
adapter. An unknown placeholder is an error. `gpucomp bench --report` writes it
whenever the template exists. A run on another machine, such as a discrete GPU,
regenerates every number without touching the prose. With several adapters, each
section repeats per adapter.

**Bench additions:**
- **zstd baselines** through the `zstd` crate (0.14.2, `zstdmt` feature), in the
  bench crate only. `cpu.zstd-N` compresses 64 KiB chunks as separate frames on all
  rayon threads, the same random access as a `.gpcz` file. `cpu.zstd-N-file`
  compresses the whole input as one frame with zstd's own workers (like `zstd
  -T0`) and decodes it on one thread. Levels 1 and 3. They also run on the numeric
  inputs. This closes M9's acceptance gap: `BENCHMARKS.md` now has a
  ratio-vs-throughput table with zstd next to every level.
- **GLZ-E at levels 2 and 3** in both directions on CPU and GPU (the GPU file must
  equal the twin's, as at level 1), and **GLZ-E in the filters suite**.
- **Per-stage profile.** `GpuEncoder::stages` / `dispatch_stages` and
  `GpuDecoder::stages` / `dispatch_stages` run a prefix of the kernel list, so the
  bench times each prefix (`gpu.<path>.upto.<stage>`) and the report subtracts
  neighbours. This replaces the temporary `take(upto)` hacks of earlier profiling.
  Prefixes are timed by wall clock around submit and wait, since a prefix can't
  carry timestamps; the submit overhead cancels in the differences.
- **Host CPU** in each run (`cpu`, e.g. `Apple M4 Pro, 12 threads`). Older runs
  load with it empty.

**The recorded run, and one discarded attempt.** The first M10 run measured 2–5%
below `deep-pass` on unchanged GPU rows, and up to 10–13% below on multi-threaded
CPU rows and upload/readback. Spotlight (`mds_stores`) and `mediaanalysisd` were
using ~1.5 cores. A same-session A/B of the last commit against the working tree
(Silesia, median of 15, three alternating rounds) showed the M10 changes cost
nothing:

| Kernel | `8144d84` | M10 tree |
|---|---|---|
| LZ4 encode | 74.6–74.8 ms | 74.7–74.8 ms |
| GLZ-E encode | 93.7–93.8 ms | 93.8–94.0 ms |
| LZ4 decode | 61.8–62.5 ms | 62.3–62.7 ms |
| GLZ-E decode | 57.9–58.6 ms | 57.8–58.5 ms |

That run was deleted. It was re-recorded once those services had finished, keeping
the second run whatever it showed. Its GPU rows are within ±3% of `deep-pass`.
Multi-threaded CPU rows still vary ±5–11% between runs with desktop apps open.

**Findings** (Silesia, end to end; details in bench/REPORT.md):
- **zstd in-process agrees with its CLI:** 64 KiB chunks give 2.763× at level 1 (the
  CLI's `-B64K` bench gave 2.763×), and the whole file 2.891×. Multi-threaded whole-file
  compression measured 5.90 GB/s here against 6.47 from `zstd -T0`.
- **GPU compression doesn't beat the CPU on this Mac.** GPU LZ4 level 1 runs at
  2.36 GB/s against lz4_flex's 5.43 (0.43×). GLZ-E level 3 matches zstd -1's ratio on
  64 KiB chunks (2.75× vs 2.76×) at 0.68 GB/s against 5.80.
- **Against the same algorithm, the GPU wins:** 2.3× the LZ4 twin, 2.2× the GLZ-E
  twin at level 1, and 4.9× at level 3. The twin itself is 5.3× slower than
  lz4_flex, so the gap is the parallel algorithm's extra work per byte.
- **Profile, level 1 LZ4 encode:** match 34 ms, parse 26 ms, emit 15 ms, transfers
  and host work 15 ms (16%). The deep pass is ~200 ms, about two thirds of level 3.
  GLZ-E adds ~18 ms to encoding and ~16 ms to decoding. Decode transfers are 20–30%.
- **Numeric data (256 MiB inputs):** GLZ-E with filters beats zstd -3 on the whole
  file for sorted u32 (4.08× vs 1.38×) and i16 audio (1.25× vs 1.04×). zstd stays
  ahead at level 1 on u64 timestamps (3.91× vs 3.17×) and f32 points (1.59× vs 1.45×).

## Optimization research (2026-10-11)
Started from the M10 profile (bench/REPORT.md, "Where the time goes"). Every change
below is byte-identical to the CPU twin, so the existing twin and matrix tests guard
it; none of them could fail first. Kernel times are Silesia medians of 12 (wall
clock of `dispatch_stages` prefixes), same session.

**Statistics first** (CPU twin over Silesia, level 1): the parse steps through
55 M literal bytes (26% of input) in 9.7 M runs. 37% of first hash candidates
are collisions (a different 4-byte prefix), and so are 30% of level-3 chain
candidates.

**Kept:**
- **The parse copies phase A's path instead of walking it again.** Splitting the
  parse showed phase A (speculative walks) ≈ 12.9 ms, phase B (fix-ups) ≈ 1–2 ms and
  phase C (re-walk to write sequences) ≈ 11–12 ms. Phase C repeated phase A's work.
  Now phase A records each match it takes in one u32 (start in the segment, 14 bits;
  length, 18 bits) in the `chain` buffer, which the parse never reads otherwise.
  Phase B notes where each lane's corrected walk merged into its phase-A walk.
  Phase C walks only up to that merge point and copies the remaining records,
  reading each offset back from its match word (still intact: sequences are written
  behind the matches they describe). **Parse 26.5 → 18.2 ms.** Memory: `chain` now
  exists at level 1 too, sized c/4 words per chunk (+1 byte per input byte); levels
  2–3 reuse their links buffer. 256 MiB still fits one batch.
- **Emit keeps long literal runs in workgroup memory.** After each block of 64
  sequences, every lane looped over all 64, reading each literal length from global
  memory, to find the rare runs of 64+ bytes. Each lane now stores its own run
  (dst, src, len) in workgroup memory. **Emit ~15.5 → ~13.8 ms.**
- **Emit writes short literal runs a word at a time** (`put_run`: one `atomicOr`
  per output word, masked at the edges, instead of one per byte). **~13.8 →
  ~12.0 ms.** The same idea made the GLZ *decoder* 10% slower (above); in emit the
  loop is simple and each lane owns its run.
- **Match finding is software-pipelined.** The candidate's first word is loaded
  before the first barrier and used after the insert; the next block's input word
  is loaded before the second barrier. **34.2 → 32.6 ms.**

Level-1 kernels: LZ4 75 → 63.4 ms, GLZ-E 95.7 → 82.9 ms.

**Rejected (measured):**
- **Skipping literal runs in the parse** (scan 4 match words per iteration to the
  next match, instead of stepping a byte at a time): parse −0.4 ms, within noise.
  Literal steps were cheap; the walks' cost is the match steps' dependent loads.
- **16-bit hash-table entries** (two buckets per word, halving the 16 KB table, with
  a compare-and-swap max): match kernel 34.0 → 35.1–36.7 ms on Silesia, and 51 →
  119–125 ms on zeros, where every lane contends for one bucket. **Workgroup memory
  is not what limits match finding**, contrary to the hash_log notes above (hash_log
  11's speed-up came from finding fewer matches, not from occupancy).
- **Extension as a separate full-occupancy pass** (the deep pass's recipe, applied
  to level 1): match kernel 34 → 14.5 ms, but the extend pass took 25 ms (total
  ~40). The split costs a scratch round trip and the shared input word. It did
  measure the match kernel: ~14.5 ms hashing and table with barriers, ~9 ms
  loading each candidate's first word, ~11 ms extension (and the barriers alone
  ~5 ms).

**A fast CPU twin** (`cpu::lz4::fast::sequences`, used by the LZ4 and GLZ encoders;
GLZ with dependency groups keeps the reference parse). It returns exactly
`parse(find_matches(..))`, tested on fixtures at every level and 200 proptest cases
over block, hash, probe, lazy and depth. Every position is still inserted block by
block (the GPU's semantics, and cheap: 5.2 GB/s alone), but matches are computed only
where the parse looks, about 0.42 per input byte with the lazy look-ahead cached,
comparing 8 bytes at a time, and deep candidates are skipped unless their byte at the
current best length matches. Single thread, 64 MiB of Silesia:

| Level | Reference find + parse | Fast |
|---|---|---|
| 1 | 0.16 GB/s | 0.29 GB/s (1.8×) |
| 2 | 0.055 GB/s | 0.145 GB/s (2.6×) |
| 3 | 0.020 GB/s | 0.076 GB/s (3.8×) |

Multi-threaded twin on Silesia: LZ4 1.2 → 2.1 GB/s, GLZ-E 0.97 → 1.7, GLZ-E level 3
0.15 → 0.6. The reference functions stay: the GPU tests compare intermediate buffers
with them. Rejected: a branch-free level-1 lookup (one 8-byte compare, length from
trailing zeros): 0.29 → 0.26 GB/s.

**CPU+GPU together (`gpu::hybrid`, `--hybrid`).** This chains three earlier
properties: every GPU chunk is byte-identical to its CPU twin's, chunks are
independent, and container files splice (`format::splice`, a plan of independent
copies). So the GPU can compress chunks from the front while the CPU compresses
chunks from the back, and **the file equals the GPU-only file whatever the split**.
The schedule can then adapt to measured speeds without making the output
nondeterministic. Decompression is the same with the CPU's own decoders. Silesia,
same session (a background process used about a core throughout):

| | GPU alone | CPU alone | CPU+GPU |
|---|---|---|---|
| Compress LZ4 level 1 | 2.80 | 2.16 (twin) | **3.93** (+40%) |
| Compress GLZ-E level 1 | 2.27 | 1.68 (twin) | **3.19** (+41%) |
| Compress GLZ-E level 3 | 0.72 | 0.63 (twin) | **1.19** (+65%) |
| Decompress LZ4 | 2.41 | 14.97 (lz4_flex) | 14.24 (the GPU stays out) |
| Decompress GLZ | 3.57 | 7.30 | 7.00 (the GPU stays out) |
| Decompress GLZ-E | 2.85 | 3.01 | **4.12** (+37%) |

What the scheduler had to learn, with numbers:
- **An extra GPU batch now costs ~1.4 ms** (Silesia in 1 / 7 / 26 / 51 batches:
  75.7 / 84.1 / 107.5 / 147.7 ms). The M8 note's 55 ms predates the segmented parse.
- **A GPU pass has a latency floor:** a chunk is decoded by one workgroup, partly
  serially. On an idle machine, GPU decode of 64 / 256 / 1024 chunks takes
  12.4 / 12.7 / 33.7 ms for LZ4 and 4.6 / 6.3 / 25.6 ms for GLZ-E. A 4 MiB probe
  measures only the floor: GLZ-E's probe looked like 0.5 GB/s, so the GPU wrongly
  dropped out. **The rule now:** the GPU waits for the CPU's rate, joins only if the
  CPU alone would need over 40 ms for what's left, starts with half of it, then takes
  its share by measured rates, and stops below a quarter (it only adds contention).
  With 4 MiB probes instead, compression was 2.23 / 1.68 GB/s (LZ4 / GLZ-E) against
  3.87 / 3.22.
- **CPU side, compression:** one worker per rayon thread but one (the GPU thread
  needs a core for its copies), each claiming 4 chunks. Groups of 48 chunks
  compressed by all threads at once waited on their slowest thread: 2.94 against
  3.44 GB/s.
- **CPU side, decompression:** one claimant, few large claims (half its share of
  what's left, all of it once the GPU is out). Each parallel call's tail costs
  ~0.5 ms on this 8P+4E CPU, which matters at 14 GB/s: LZ4 in 7 calls of 32 MiB ran
  at ~10.7 GB/s against 14.5 for one call. Per-thread workers with small claims also
  measured ~10 GB/s, spending ~30% more CPU time per byte. Order of claims, nested
  rayon calls and pre-faulting the output were ruled out as causes; the reason is
  not understood.
- **Joining the parts:** parallel copies and freeing the parts on a background
  thread took LZ4 from 3.35 to 3.70 GB/s.

**Recorded run `hybrid`** (Silesia; `mediaanalysisd` used ~1–1.5 cores for the whole
run and didn't settle in over 30 minutes of waiting, but the CPU baselines landed within
±2% of M10):

| | M10 | `hybrid` |
|---|---|---|
| GPU LZ4 level 1 compress, kernel / end to end | 2.81 / 2.36 | **3.41 / 2.80** |
| GPU GLZ-E level 1 compress, end to end | 1.98 | **2.26** |
| CPU twin, LZ4 / GLZ-E / GLZ-E level 3 | 1.03 / 0.90 / 0.14 | **1.97 / 1.66 / 0.62** |
| CPU+GPU compress, LZ4 / GLZ-E / GLZ-E level 3 | | **3.65 / 2.97 / 1.16** |
| CPU+GPU decompress, LZ4 / GLZ-E / GLZ-E level-3 file | | 14.35 (GPU out) / **4.09 / 4.52** |

Against multi-threaded lz4_flex (5.35 GB/s), GPU LZ4 compression went from 0.43× to
0.52×, and CPU+GPU reaches 0.68×. bench/REPORT.md's prose was rewritten for these
numbers. The "same algorithm" lead shrank (GPU over its twin: 2.3× → 1.4× for LZ4),
because the twin now skips match finding where its serial parse doesn't look.

## Catching up with lz4_flex and zstd (2026-10-11)
Where the gap was (Silesia, `hybrid` run): lz4_flex compresses at 5.35 GB/s and
CPU+GPU LZ4 at 3.65; zstd -1 in 64 KiB chunks gets 2.76× at 5.69 GB/s and decodes at
10.5, GLZ-E level 1 gets 2.50× and the CPU decoded it at 2.97.

**The CPU twin can't reach lz4_flex.** A sampled profile of the twin (LZ4 level 1,
one thread): computing matches ~52%, hash-table inserts ~17%, extension ~13%, emit
~12%. The hot instructions sit right after the "is it a collision" and "is it a
match" branches: mispredictions.
- **Kept:** a cursor-based LZ4 emit with 16-byte literal copies, a one-pass GLZ emit,
  and a one-pass Huffman encoder (lanes written speculatively, kept if smaller than
  raw). On 12 threads: LZ4 twin 2.15 → 2.31 GB/s, GLZ-E twin 1.70 → 1.83.
- **Rejected:** computing every position's level-1 match branch-free in a block, then
  parsing the array, instead of on demand. One thread: 0.27 → 0.29 GB/s. But 12
  threads: LZ4 2.15 → 2.0 and GLZ-E 1.80 → 1.46, with a fresh 256 KB match array per
  chunk (large allocations are mapped and unmapped by the kernel); even reusing a
  per-thread array, GLZ-E stayed slower (1.71), from the extra memory traffic. The
  first recorded run caught this regression; it was deleted and re-recorded.

To write the GPU's files, the twin must insert every position and match every
position the parse visits; lz4_flex does neither. It stays ~3× faster per core.

**So the GPU adds to lz4_flex instead** (`hybrid::compress_static`, `--gpu-share`): a
fixed split, lz4_flex (on all threads but one) taking the first chunks and the GPU
the last share of them, at the same time. The file differs from the GPU-only one but
depends only on the input and settings, so it stays deterministic. A prototype that
wrote two separate files (lz4_flex alone 6.21 GB/s in that session): GPU share 0.2 /
0.25 / 0.3 / 0.35 / 0.4 → 7.25 / 7.88 / 8.01 / 7.28 / 6.35 GB/s. Joining the two
files cost the real thing most of that: splicing (a fresh 105 MB buffer) took
4.7–6.7 ms of ~31, leaving +6%. Now the CPU writes its data straight into the
file after room for the table (`cpu::container::compress_into`), and only the GPU's
part (`GpuEncoder::compress_into`) is appended: **7.1–7.4 GB/s at 0.3 against
lz4_flex's 6.4, +11–15%** (ratio 2.02×). The bench row uses 0.3
(`HYBRID_LZ4_FLEX_SHARE`).

**CPU decoders** (byte-identical, same errors; the GPU GLZ-E tests compare error
statuses):
- **Huffman:** a 64-bit bit buffer refilled bytes at a time, decoding straight into
  the output, four lanes interleaved, one overrun check per lane (bits only
  accumulate, so it catches what a check per symbol would); `to_glz` decodes tokens
  and literals in place. GLZ-E 2.97 → 4.50 GB/s.
- **GLZ:** 16-byte wild copies for literal runs and for matches with offsets ≥ 16
  (output is written in order and matches read only finished bytes). GLZ **7.2 →
  13.7 GB/s** (lz4_flex: 13.3), GLZ-E **→ 7.1 GB/s** (2.4× in all).
- Rejected: two symbols per lookup (a pair table per long stream): 7.08 → 6.82 GB/s.

With these decoders the hybrid's GPU stays out of decoding on this machine (the CPU
finishes Silesia's GLZ-E in ~31 ms); letting it join below 40 ms made every codec
slower (GLZ-E 6.83 → 5.58 GB/s).

**GPU kernels** (byte-identical):
- **Match finding** compares the candidate's next 12 bytes with independent loads
  instead of extend's word-by-word loop: 32.9 → 30.6 ms. The same change to `extend`
  itself made the parse slower (19.4 → 22.4 ms: most of its extensions stop at once),
  so it was reverted.
- **`extend16`** compares the whole probe at once: deep pass, level 2 / 3, 68.7 /
  206 → 60.6 / 196 ms.
- **Tagged chain links:** each link carries an 11-bit tag of its candidate's first 4
  bytes above the 21-bit position (the match kernel already loads that word), and the
  deep walk skips a candidate whose tag differs from the position's without loading
  its bytes (30% of level-3 candidates are collisions). Deep pass **60.6 / 196 →
  54.5 / 155 ms**; level-3 kernels 256 → 215 ms.
- Rejected: parallel host copies for upload and readback (rayon; fresh zeroed pages
  for readback): 78.9 → 77.7 ms end to end, not worth the code.
