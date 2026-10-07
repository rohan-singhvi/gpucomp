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
