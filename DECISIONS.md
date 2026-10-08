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
