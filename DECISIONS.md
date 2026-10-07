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
