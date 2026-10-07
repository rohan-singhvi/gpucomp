# GPU Compression/Decompression in wgpu — Implementation Plan

## 0. Instructions for the implementing agent

- Work **milestone by milestone**, in order. Don't start the next milestone until the current one meets its acceptance criteria and all tests pass.
- After each milestone: run `cargo test --workspace`, `cargo clippy --workspace -- -D warnings`, and `cargo fmt --check`. Then stop and tell the user the milestone is ready, with a suggested commit message naming it. **The user does all committing and pushing; the agent never runs `git commit` or `git push`.**
- The CPU reference implementation is the source of truth for **decoding**. Every GPU-decoded result must be byte-identical to the original input.
- GPU **compression** output does not have to match the CPU encoder byte-for-byte; parsing choices can differ. It must, however, (a) be a valid stream that the CPU reference decoder (and, for LZ4, `lz4_flex`) decodes back to the original input, and (b) be deterministic: the same input and settings must produce the same bytes on every run.
- If something in this plan is wrong or impossible (e.g. a wgpu API changed), choose the closest working alternative, note it in `DECISIONS.md` with a one-paragraph rationale, and continue.
- Use the latest stable versions of `wgpu` and other crates when starting, and pin them in `Cargo.toml`. Don't rely on remembered API signatures; check the docs or the crate source for the version you pinned.
- Must run on **macOS (Metal backend)** and **Windows (D3D12 and Vulkan backends)**. Don't use features those backends lack unless they are behind a runtime check with a fallback.
- Development is **test-driven**: write a failing test, watch it fail on its assertion, then write the code.
- **Benchmark continuously.** After every milestone, and after any change meant to improve speed or ratio, run `gpucomp bench --record` (which writes raw JSON to `bench/results/`) and regenerate `BENCHMARKS.md` (see §7a). Both belong in the user's next commit.

## 1. Goal and scope

Build a Rust library and CLI that **compresses and decompresses data on the GPU** through wgpu (WGSL compute shaders). Use a chunked, parallel-friendly format, with CPU reference implementations of both directions.

**Primary goals (equal weight):**
1. GPU compression that produces valid streams with a competitive ratio and high throughput.
2. GPU decompression that is fast and byte-exact.

Both directions get honest benchmarks against CPU baselines.

**Also core to the design:**
3. **Seekable / random access.** Any byte range `[offset, offset + len)` of the original data can be decompressed by reading and decoding only the chunks that overlap it, on the CPU and on the GPU, without touching the rest of the file. This holds for files of any size, including streamed ones (M8).
4. **Automatic per-chunk filter selection.** The core codec is general-purpose. In front of it, the encoder tries a few reversible filters on each chunk (none, byte-shuffle, delta), keeps whichever gives the smallest compressed chunk, and records the choice in the chunk table. The GPU tries the candidates in parallel, which suits the hardware well (M7).
5. **Compression levels, throughput first.** Level 1 is the fast default: greedy parsing and a small hash table. Higher levels add more match candidates, smarter parsing, a wider filter search and eventually entropy coding, much as zstd does (M9). Throughput is built first. A slow GPU compressor has little reason to exist next to zstd on a CPU.

**Secondary goals (later milestones):** a GPU-friendly codec designed for both directions, and entropy coding.

**Definition of done: compression is not optional.** The project is not finished unless **all** of the following hold on Metal and on D3D12/Vulkan:
- `gpucomp compress --gpu` produces valid files for every codec the project ships (LZ4-block and GLZ). The CPU decoder and the GPU decoder both read them back exactly.
- GPU compression is faster end to end than rayon multi-threaded CPU compression on files of at least 100 MB on a discrete GPU. If it isn't, `bench/REPORT.md` must explain why, with profiling data.
- Every codec has a GPU encoder. Don't add a codec that only has a GPU decoder.
- Any milestone that changes a format updates **both** the encoder and the decoder, CPU and GPU, in the same milestone.

**Non-goals (for now):**
- Compatibility with zstd, gzip or zip streams. Those formats are serial by design.
- Matching the compression ratio of slow CPU modes (LZ4-HC, zstd high levels). The target is LZ4 fast-mode ratio or better.
- Web/WASM target. Keep it possible, but don't test it.
- Multi-GPU.
- A C API / FFI layer. The Rust library and the CLI are the interfaces.

## 2. Tech stack

- Rust, stable toolchain, 2021 or newer edition
- `wgpu` for GPU access, with WGSL shaders
- `pollster` (or similar) to block on async wgpu calls in the CLI and tests
- `bytemuck` for casting POD structs into buffers
- `lz4_flex` as the CPU LZ4 block baseline (encoder and decoder) and as an independent validator of GPU-produced streams
- `rayon` for the multi-threaded CPU baselines
- `clap` for the CLI
- `criterion` for CPU benchmarks, plus a custom harness for GPU timings
- `proptest` for round-trip property tests
- `cargo-fuzz` for the CPU decoder (optional; Linux/macOS only)
- `xxhash-rust` (xxh3) for chunk checksums

## 3. Repository layout

```
gpucomp/
├── Cargo.toml              # workspace
├── plan.md
├── DECISIONS.md            # agent-maintained log of deviations/choices
├── BENCHMARKS.md           # short, generated summary of benchmark history (§7a)
├── bench/results/          # raw benchmark runs, one JSON file per run (committed)
├── crates/
│   ├── format/             # container format: headers, chunk table, (de)serialization
│   ├── cpu/                # CPU encoders + CPU reference decoders
│   ├── gpu/                # wgpu context, pipelines, buffer management, WGSL shaders
│   │   ├── src/encode/     # GPU compression pipelines
│   │   ├── src/decode/     # GPU decompression pipelines
│   │   └── shaders/        # *.wgsl, loaded with include_str!
│   ├── cli/                # `gpucomp compress|decompress|bench|info`
│   └── bench/              # benchmark harness + report generation
├── testdata/               # small fixtures committed; large corpora downloaded by script
└── scripts/
    └── fetch_corpus.(sh|ps1)   # downloads Silesia + Canterbury corpora
```

## 4. Container format (v1)

The file is split into **independent chunks** (default 64 KiB uncompressed; configurable, power of two, 4 KiB–1 MiB). No back-reference crosses a chunk boundary, so every chunk can be compressed and decompressed in parallel.

```
Header (little-endian, 32 bytes):
  magic          [u8; 4] = b"GPCZ"
  version        u16     = 1
  codec          u16     // 0 = stored, 1 = LZ4-block, 2 = GLZ (milestone 6)
  chunk_size     u32     // uncompressed size of every chunk except possibly the last
  chunk_count    u32
  total_size     u64     // total uncompressed size
  flags          u32     // bit0: per-chunk checksums present
  level          u8      // compression level used (informational; decoding ignores it)
  reserved       [u8; 3] // must be zero

Chunk table: chunk_count entries of 24 bytes:
  comp_offset    u64     // byte offset of the chunk's data, relative to start of the data section
  comp_size      u32     // top bit set = chunk stored raw
  uncomp_size    u32
  checksum       u32     // low 32 bits of xxh3 of the uncompressed chunk (0 if flag off)
  filter         u8      // 0 = none, 1 = byte-shuffle, 2 = delta (see §4a)
  filter_width   u8      // element width in bytes for shuffle/delta: 1, 2, 4 or 8 (0 for none)
  reserved       u16     // must be zero

Data section: chunk payloads, each **padded to a 4-byte boundary**.
```

`comp_offset` is `u64` so files larger than 4 GiB stay seekable. The GPU never sees it: shaders only get `u32` offsets relative to the current batch (§6).

Rules:
- If a chunk doesn't compress (comp_size ≥ uncomp_size), store it raw and set the top bit of `comp_size`. This applies to both the CPU and GPU encoders, and the GPU decode path copies stored chunks directly.
- Chunk payload offsets are 4-byte aligned, which keeps WGSL word addressing simple.
- Uncompressed chunk sizes are multiples of 4, except the last chunk.
- LZ4 match offsets are 16-bit (max 65535), so for chunk sizes above 64 KiB the encoders must cap the match distance at 65535.
- Document the format in `crates/format/FORMAT.md`, and keep it in sync.
- The `filter` and `level` fields are in the format from M1, so it doesn't need a version bump later. Until M7, encoders always write `filter = 0`, and decoders reject filter ids they don't implement.

**Random access.** Every chunk except the last has the same uncompressed size, so the chunks covering the byte range `[offset, offset + len)` are `offset / chunk_size ..= (offset + len - 1) / chunk_size`. A range read takes the header, the chunk table and only those chunks' payloads; it decodes them and trims the first and last chunk. Both decoders (CPU and GPU) expose it: `decompress_range(reader, offset, len)` in the library and `gpucomp decompress --offset N --length M` in the CLI. Any later layout change (M8's streaming) must keep the chunk table locatable in O(1) reads.

### 4a. Filters

A filter is a reversible byte transform applied to a chunk **before** LZ compression and undone **after** decompression. Filters never cross chunk boundaries. With `w = filter_width` and `n` = chunk length, there are `n / w` whole elements, and the trailing `n % w` bytes always pass through unchanged.

- **none (0):** identity.
- **byte-shuffle (1):** a transpose. Byte `j` of element `i` moves to position `j * (n / w) + i`, so all first bytes come first, then all second bytes, and so on. This groups the slowly-changing high bytes of numeric arrays together.
- **delta (2):** each element becomes `elem[i] - elem[i-1]` (wrapping, little-endian `w`-byte integers), with `elem[0]` unchanged. Good for sorted or smoothly varying integers and timestamps.

**Selection:** for each chunk, the encoder compresses the chunk under each candidate `(filter, width)` in the level's candidate set and keeps the smallest output. Ties go to the lower `(filter, width)` pair, so output is deterministic. Level 1 tries `{none, shuffle-4, delta-4}`, and higher levels add widths 2 and 8. On the GPU, the candidates are independent work items run in parallel, e.g. one workgroup per `(chunk, candidate)`, followed by a selection pass.

## 5. Milestones

### M0 — Scaffolding and GPU smoke test
- Set up the workspace, CI (GitHub Actions: `macos-latest` and `windows-latest`; build plus CPU tests; GPU tests may be skipped in CI if no adapter is present), and the fmt/clippy config.
- Add a `gpu::Context` that requests an adapter (high-performance), logs the adapter info, backend and relevant limits, and creates a device that requests the adapter's **actual** limits for storage buffer binding size, buffer size and workgroup storage size.
- Add a trivial compute shader (an XOR of each `u32` with a constant), plus a test that compares the result to a CPU computation.
- Bootstrap the benchmark harness (§7a). It records the platform's ceilings, which every later end-to-end number is compared against: host→GPU upload GB/s, GPU→host readback GB/s, a trivial kernel's GB/s, and CPU `memcpy` GB/s.
- **Accept when:** `cargo run -p cli -- info` prints the adapter, backend and limits on both macOS and Windows, the smoke test passes, and `BENCHMARKS.md` has the first baseline run.

### M1 — CPU path: container + LZ4-block codec
- `format` crate: serialize and deserialize the header and chunk table, with validation (bounds, overlaps, sizes, unknown filter ids, nonzero reserved fields). Add a pure helper that maps a byte range to the chunks covering it.
- Random access on the CPU: `decompress_range` reads only the header, the table and the chunks it needs (test this with a reader that counts the bytes it serves). The CLI gets `decompress --offset --length`.
- `cpu` crate:
  - A chunked encoder using `lz4_flex` block compression per chunk. This is the CPU compression baseline.
  - A **hand-written** greedy LZ4 block encoder that uses the same algorithm the GPU encoder will use in M3: hash table, greedy parse, same hash function and table size. This is the GPU encoder's reference and lets you debug the GPU against a CPU twin.
  - A **hand-written** LZ4 block decoder. Keep it simple, readable and bounds-checked.
- Run per-chunk work in parallel with `rayon` for both directions. These are the fair CPU baselines.
- Tests: round-trip on empty input, 1 byte, exactly one chunk, chunk ± 1, random data (incompressible), zeros, text, and proptest on arbitrary inputs. Check that both encoders' output decodes with both decoders (the hand-written one and `lz4_flex`).
- Tests for range reads: ranges inside one chunk, ranges spanning chunk boundaries, a range covering the last (short) chunk, a zero-length range, and out-of-bounds ranges (rejected with an error).
- **Accept when:** all round-trip and range tests pass, the CLI can `compress`, `decompress` and range-decompress files on the CPU, and the CPU baselines are recorded in `BENCHMARKS.md`.

### M2 — GPU LZ4 decompression, naive (one invocation per chunk)
- Upload the compressed data section and the chunk table to storage buffers. Allocate an output buffer that is zero-initialized and sized to the total, rounded up to a multiple of 4.
- In WGSL, **each invocation decodes one entire chunk serially**, mirroring the CPU decoder exactly.
- Read bytes through a helper: `byte(i) = (src[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu`.
- Writes: chunk output offsets are 4-aligned, so each invocation owns whole words. Accumulate bytes into a `u32` register and flush whole words. Handle the final partial word carefully.
- Match copies must handle overlap (offset < match length) with byte-serial semantics: read back what was just written.
- Bounds-check every read and write in the shader. On a malformed stream, write an error code into a per-chunk status buffer instead of hanging or overrunning.
- Random access on the GPU: upload and dispatch only the chunks a range needs.
- **Accept when:** GPU output is byte-identical to the original across the whole M1 test suite (range reads included), on Metal and D3D12 (and Vulkan on Windows if available). The CLI gets `decompress --gpu`.

This version will be slow. That's expected; it is the correctness baseline.

### M3 — GPU LZ4 compression (one workgroup per chunk)
This produces standard LZ4 block payloads inside the container, so every GPU-compressed file is checkable with `lz4_flex` and the M1 decoder.

Each workgroup compresses one chunk in four phases:

1. **Match finding (parallel).** Process the chunk in blocks of `WG_SIZE` positions. For each block:
   - Every invocation hashes the 4 bytes at its position and looks up a candidate in a hash table held in `var<workgroup>` memory. The table holds only positions from **earlier** blocks.
   - Every invocation verifies the candidate and extends the match forward, capped by the LZ4 rules below and the distance limit. It records `(match_len, offset)` for its position in a per-chunk scratch buffer (storage memory).
   - Barrier, then all invocations insert their positions into the hash table with `atomicMax`, so each bucket deterministically keeps the latest position. Barrier again.
   - Size the table to fit `max_compute_workgroup_storage_size` (16 KiB is the guaranteed minimum, so default to 2048–4096 `u32` entries). Make the size a pipeline-override constant.
   - Optional improvement, once the basics work: also check matches within the current block (positions that precede the invocation's own position).
2. **Parse (serial, cheap).** One invocation walks the scratch buffer greedily. At each position, take the match if `match_len ≥ 4`, otherwise emit a literal and advance by 1. Record the chosen sequences (literal run start/length, match length, offset) into a sequence buffer, along with each sequence's encoded byte size. This loop is O(chunk size) with no searching, so it is cheap compared with phase 1. If profiling shows it dominates, try a parallel parse later (e.g. pointer jumping over "next position" links) and record the result in `DECISIONS.md`.
3. **Emit (parallel).** Run a prefix sum over the sequences' encoded sizes in the workgroup to get each sequence's output offset, then let invocations write sequences in parallel. Sequences can share output words at their boundaries, so use `atomicOr` into a zero-initialized output, or have each invocation build whole words. Same tradeoff as M5; start with `atomicOr`.
4. **Packing (separate dispatch).** Each chunk compresses into its own worst-case-sized slot (`n + n/255 + 16`, rounded up to 4). Write each chunk's compressed size, or flag it as stored if it didn't shrink. A second dispatch (or the host, for the first version) prefix-sums the sizes into `comp_offset`s and copies the payloads into a packed data section. Build the chunk table from the same results.

LZ4 block rules the GPU encoder must obey (check them against the LZ4 block format spec before implementing):
- Minimum match length is 4.
- The last 5 bytes of a block are always literals.
- The last match must start at least 12 bytes before the end of the block.
- Offsets are 1–65535.
- Literal and match lengths beyond 15 use the 255-continuation encoding.

Determinism: never let the result depend on scheduling order. Use `atomicMax` for hash inserts, a deterministic tie-break on equal-length matches (prefer the smaller offset), and no atomic counters whose values feed into output layout.

- **Accept when:**
  - Every GPU-compressed test input decodes correctly with the M1 hand-written decoder, `lz4_flex`, and the M2 GPU decoder.
  - The same input compressed twice gives identical bytes.
  - Compression ratio is within ~10% of `lz4_flex` on the Silesia corpus. Record the gap in the report; if it's larger, note which phase is responsible.
  - The CLI gets `compress --gpu`.
- This encoder is **level 1** (§1 goal 5). Keep the hash table size and the number of candidates per position as parameters, not hard-coded, because M9 builds the higher levels on them.

### M4 — Cross-path validation matrix
Add an integration test that runs every combination over the test corpus:

| Compressed by | Decompressed by |
|---|---|
| CPU (`lz4_flex`) | CPU, GPU |
| CPU (hand-written greedy) | CPU, GPU |
| GPU | CPU (hand-written), `lz4_flex`, GPU |

Each one must reproduce the original input exactly. Also run the malformed-input suite against the GPU decoder here.

- **Accept when:** the full matrix passes on every available backend.

### M5 — GPU LZ4 decompression, cooperative (one workgroup per chunk)
- Each workgroup (start with 32 or 64 invocations; make it a pipeline-override constant) decodes one chunk.
- Use a sequential parse with parallel copies: invocation 0 (or all invocations redundantly) parses the next sequence header, then all invocations copy the literal bytes and the match bytes cooperatively, with `workgroupBarrier()` and `storageBarrier()` between dependent steps.
- For overlapping matches (offset < length), copy in rounds of `offset` bytes, with a barrier between rounds.
- Writes from multiple invocations can hit the same `u32` word. Two options:
  1. Declare the output as `array<atomic<u32>>`, zero-initialized, and use `atomicOr` to merge bytes. Simple, correct, and possibly slower.
  2. Assign whole words to invocations, and handle the word-unaligned head and tail of each copy with a single invocation.
  Implement option 1 first, then try option 2 and keep whichever benchmarks faster. Record the result in `DECISIONS.md`. Apply the winner to M3's emit phase too.
- **Accept when:** M4's matrix still passes and the benchmark shows a clear speedup over M2. Report the numbers.

### M6 — GPU-friendly codec "GLZ" (codec id 2), both directions
LZ4's interleaved token stream forces a serial parse on decode and a serial byte layout on encode. GLZ splits each chunk into **separate streams**, so both directions become prefix sums plus parallel copies:

```
per chunk:
  seq_count     u32
  lit_lens      [u16 or u32]   // literal length per sequence
  match_lens    [u16 or u32]   // match length per sequence
  offsets       [u16 or u32]   // match offset per sequence
  literals      [u8]           // all literal bytes concatenated
```

**Encode (GPU):** reuse M3's match-finding and parse phases unchanged. The emit phase becomes simpler: fixed-width fields are written at index `i` of each stream, and literals are placed via a prefix sum over `lit_len`. No variable-length token encoding is needed.

**Decode (GPU):**
1. Run a parallel prefix sum over `lit_len + match_len` to get each sequence's output start, and a prefix sum over `lit_len` to get each sequence's literal source.
2. Copy all literals in parallel. They don't depend on anything.
3. Resolve matches. A match only depends on earlier output, so process it in passes, or within a workgroup in sequence order with cooperative copies. Measure both approaches.

**Encoder-side dependency elimination (from Gompresso, see §9):** add an encoder option that refuses matches whose source bytes fall inside another match from the same group of `G` sequences (e.g. G = 32). The decoder can then resolve a whole group of matches in one parallel step with no waiting. This costs some ratio; measure it with the option on and off. This is the clearest example of the encoder and decoder being designed together.

- Add a GLZ CPU encoder (same greedy algorithm as M1's hand-written encoder) and a CPU decoder.
- Start with fixed-width fields for simplicity. Varint or bitpacking can come later if the benchmarks justify it.
- Extend M4's matrix to cover GLZ.
- **Accept when:** GLZ round-trips in every CPU/GPU combination, and a benchmark table compares GLZ with LZ4-block on ratio and on GPU throughput in **both** directions.

### M7 — Automatic per-chunk filter selection (both directions)
Implements §4a for both codecs.
- CPU and GPU forward and inverse transforms for shuffle and delta at widths 1, 2, 4 and 8, all tested against each other. Proptest: `inverse(forward(x)) == x` for every width and length, including lengths that aren't a multiple of the width.
- GPU encoder: run the candidates for every chunk in parallel, then a selection pass picks the smallest output (with the deterministic tie-break) and writes `filter`/`filter_width` into the chunk table. The packing step copies only the winners.
- GPU decoder: after LZ decoding, apply the inverse filter per chunk in a separate dispatch (or fused, if benchmarks show it helps).
- Measure on general files (Silesia) and on numeric data: generate f32 point clouds, sorted u32/u64 arrays and i16 audio-like signals into `testdata/` with a script. Report the ratio gain, the throughput cost of trying K candidates, and how often each filter wins.
- If trying every candidate costs too much at level 1, try a cheap estimator (e.g. the matched-byte count from phase 1 of M3, without a full emit) and record the result in `DECISIONS.md`.
- Extend M4's matrix: every (encoder, decoder) pair, with filters on.
- **Accept when:** filtered files round-trip in every CPU/GPU combination, the GPU picks the same filter as the CPU reference on every test chunk (or `DECISIONS.md` explains why not), and `BENCHMARKS.md` shows the ratio/throughput effect.

### M8 — Large files and streaming (both directions)
- Handle inputs larger than `max_storage_buffer_binding_size` and `max_buffer_size` by processing them in **batches of chunks**, for both compression and decompression.
- Pipeline the batches with double or triple buffering, so uploading batch N+1 overlaps with processing batch N and reading back batch N−1.
- Compression writes chunk payloads to the output file as batches finish. The chunk table is known only at the end, so either write it at the end of the file with a footer pointer (and update the format spec and `version` accordingly), or reserve its space up front, since `chunk_count` is known from the input size. Pick one and record it in `DECISIONS.md`. Either way, random access must still need only O(1) reads to find the table.
- Range reads on large files load only the needed chunks into GPU memory, without batching the whole file.
- Respect `max_compute_workgroups_per_dimension` (65535) by using a 2D dispatch, or by looping inside the shader, when the chunk count is large.
- Map readback buffers asynchronously, and poll the device correctly.
- **Accept when:** a file of at least 4 GiB (generated test data) round-trips correctly GPU→GPU, with bounded GPU memory use, and a range read near the end of it touches only the chunks it needs. Make the memory bound configurable and log it.

### M9 — Compression levels
Builds on M3's parameterised encoder (§1 goal 5). Levels change only the encoder, so every level must decode with the same decoders.
- **Level 1** (default): M3 as built. Greedy parse, small workgroup hash table, one candidate per bucket, level-1 filter candidates.
- **Middle levels:** more candidates per position (multi-way buckets or short hash chains, possibly in storage memory), lazy matching in the parse phase (check whether position `p+1` has a longer match before committing to `p`), and wider filter search.
- **High levels:** optimal or near-optimal parsing within a chunk, and the entropy stage (M11) once it exists.
- `--level N` in the CLI; record the level in the header. The CPU twin encoder supports the same levels, so the GPU can be checked against it.
- **Accept when:** every level round-trips in the M4 matrix, ratio improves monotonically with level on Silesia, and `BENCHMARKS.md` has a ratio-vs-throughput table for each level next to `lz4_flex` and, for reference, zstd levels 1/3 on the CPU.

### M10 — Final benchmarks and report
- Run the Silesia corpus (each file and the concatenated tarball) plus synthetic data: zeros, random, and repetitive text.
- For **both compression and decompression**, measure and report separately:
  - kernel-only time (GPU timestamp queries when the `TIMESTAMP_QUERY` feature is available; otherwise note that kernel time is unavailable)
  - end-to-end time: upload + process + readback (+ packing for compression)
  - the CPU baselines: `lz4_flex` single-threaded, rayon multi-chunk `lz4_flex`, and the hand-written greedy codec
- Report throughput in GB/s of uncompressed data, compression ratio for each encoder, and the adapter/backend name.
- For compression, include a ratio-vs-throughput table so the GPU encoder's tradeoff is visible.
- Note the difference between discrete GPUs (PCIe transfer cost) and Apple Silicon (unified memory) explicitly in the report.
- Generate a Markdown report at `bench/REPORT.md`.

`BENCHMARKS.md` already holds the history by now. M10 is the full, polished run that `bench/REPORT.md` is generated from.

### M11 (stretch) — Experiments
Pick based on the M10 results. Each is independent.
- **Entropy stage:** interleaved rANS or Huffman, with N independent lanes per chunk, applied to the GLZ literal stream. Encode and decode both on the GPU.
- **GDeflate codec (codec id 3):** a GPU encoder (M3-style match finding, then distributing symbols across 32 sub-streams with Huffman coding) and a WGSL decoder ported from the Microsoft HLSL reference. Validate against the reference implementation in both directions.
- **Multi-byte symbols (from GPULZ):** match on 2- or 4-byte units for numeric data.
- **More filters:** bitpack, float-specific transforms (e.g. XOR with the previous value), and per-chunk selection over them, added to M7's framework.
- **Zero-copy path on Apple Silicon:** measure whether mapped or shared buffers reduce end-to-end time.

## 6. wgpu / WGSL gotchas (read before writing shaders)

- WGSL has **no `u8` type**. Storage buffers are `array<u32>`, so all byte access goes through shift and mask helpers. Put the helpers in a shared WGSL snippet and concatenate it at pipeline creation time.
- Multiple invocations writing different bytes of the same `u32` is a **data race** unless you use atomics or give each invocation whole words. This affects compression output just as much as decompression output.
- 64-bit integers and 64-bit atomics are not portable. Use `u32` offsets inside a batch, and keep `u64` on the host only. This is one more reason to batch large files.
- Subgroup operations are an optional feature. If you use them, check for support at runtime and keep a workgroup-memory fallback.
- Respect the adapter's limits, which can differ a lot between Metal and D3D12. Request what you need when creating the device, and fail with a clear message if the hardware can't provide it. Workgroup storage size directly limits the compressor's hash table.
- `storageBarrier()` only orders storage-memory accesses within a workgroup. There's no cross-workgroup synchronization within a dispatch, so use separate dispatches for global phases (e.g. the packing prefix sum across chunks).
- Compressed output size isn't known before compressing. Allocate worst-case slots per chunk, then pack. Never let a chunk write past its slot.
- Shaders that loop forever on malformed input can trigger a GPU timeout (TDR on Windows) and lose the device. Bound every loop by the input size. Also keep per-dispatch work bounded during compression; very large batches on slow GPUs can hit the timeout too.
- Buffer mapping is asynchronous. The device must be polled for the map callback to fire.

## 7. Testing strategy

- **Decode rule:** GPU decode output == original input, byte-for-byte, for every test input and every backend that's available.
- **Encode rule:** GPU encode output must decode correctly with every decoder (M4 matrix) and must be deterministic across runs.
- Unit tests in each crate. Integration tests in `crates/gpu/tests/` skip with a printed message (not a failure) when no adapter is found.
- Use proptest for round-trips across random sizes, chunk sizes and data distributions (random, low-entropy, long runs, data with matches right at the LZ4 end-of-block limits).
- When the GPU encoder misbehaves, compare its intermediate buffers (per-position matches, chosen sequences) with the M1 hand-written greedy encoder, which uses the same algorithm. Expose a debug flag that reads those buffers back.
- Malformed-input tests: truncated chunks, offsets pointing before the chunk start, oversized lengths. Both decoders must return an error and must not panic, hang or write out of bounds.
- Fuzz the CPU decoders with `cargo-fuzz`. Feed any crashing inputs found back into the GPU tests as fixtures.
- Verify checksums after decode when the flag is set, and expose `--verify` in the CLI.

## 7a. Benchmark log

The goal is a running, data-backed record of what each change bought.

- `gpucomp bench --record` runs the benchmark suite and writes one JSON file per run to `bench/results/<date>-<milestone>-<adapter>.json`. Each file records the git commit, the milestone label, the adapter and backend, the OS, and one row per measurement: name, input, size, direction, throughput (GB/s), compression ratio, and the timing source (`gpu-timestamp`, `wall-e2e`, `cpu`). These files are the raw data and belong in version control.
- `gpucomp bench --report` regenerates `BENCHMARKS.md` from all the JSON files. Keep it **short**: a "current best" table (one row per path: CPU baselines, GPU compress, GPU decompress, with GB/s and ratio), a "history" table (one row per run showing the headline numbers and what changed), and the platform ceilings from M0. Detail stays in the JSON.
- Measure **as much as is cheap to measure**: for each path, kernel time and end-to-end time separately, every codec × level × filter mode that exists, CPU single- and multi-threaded baselines, and the transfer ceilings. Use the median of N runs after warm-up.
- Small synthetic inputs run in CI-like time. Corpus runs (Silesia) are opt-in, with `--corpus`.

## 8. CLI

```
gpucomp compress   <in> <out> [--gpu|--cpu] [--codec lz4|glz] [--level N] [--filters auto|none] [--chunk-size 64K] [--checksum] [--backend metal|dx12|vulkan]
gpucomp decompress <in> <out> [--gpu|--cpu] [--offset N --length M] [--backend metal|dx12|vulkan] [--verify]
gpucomp bench      [<in>]     [--codec ...] [--direction compress|decompress|both] [--runs N] [--json] [--record] [--report] [--corpus]
gpucomp info       [<file>]   # with no file: print adapter/backends/limits; with file: print header, chunk stats and filter histogram
```

## 9. Prior work to read and build on

Read the relevant entry before starting each milestone. Most of these use CUDA, so the ideas transfer but the code doesn't: CUDA "shared memory" = WGSL `var<workgroup>`, "warp" ≈ subgroup (optional in wgpu), "`__syncthreads()`" = `workgroupBarrier()`.

| Work | What it contributes | Where it applies |
|---|---|---|
| **GPULZ** — Zhang et al., ICS '23 ([arXiv 2304.07342](https://arxiv.org/abs/2304.07342), [code](https://github.com/hipdac-lab/ICS23-GPULZ)) | The most recent GPU LZSS **compressor**. Analyses why earlier GPU LZSS compressors were slow, then encodes with a per-block prefix sum so every thread writes its own symbols. Also uses multi-byte symbols (2/4-byte units) for numeric data, improving both speed and ratio. Partition sizes are tuned to the GPU's shared-memory size. | M3 (encode phases), M6, M11 (multi-byte mode for numeric data) |
| **CULZSS** and its follow-ups — Ozsoy & Swany, CLUSTER '11; Ozsoy, Swany & Chauhan, ICPADS '12 / FGCS '13 | The original GPU LZSS compressor. Splits the work into a substring-matching stage and an encoding stage, and pipelines CPU and GPU work for streaming. GPULZ treats it as the baseline it improves on, so read it for the problems to avoid. | M3, M8 (pipelining) |
| **Gompresso** — Sitaridi et al., ICPP '16 ([arXiv 1606.00519](https://arxiv.org/abs/1606.00519)) | Massively parallel decompression of LZ77 (byte-level and Huffman variants). Two techniques for back-references: iterative resolution on the GPU, and changing the **compressor** to remove dependencies so threads never wait. Reports a ratio cost of 10% or less. | M5, M6 (dependency elimination) |
| **GDeflate** — Uralsky (NVIDIA), [IETF draft-uralsky-gdeflate-00](https://www.ietf.org/archive/id/draft-uralsky-gdeflate-00.html); [Microsoft reference implementation](https://github.com/microsoft/DirectStorage/tree/main/GDeflate) (Apache-2.0, includes an HLSL decoder) | DEFLATE reformatted into 32 interleaved sub-streams per 64 KB page, giving 32-way parallel decoding with essentially the same ratio as DEFLATE. The reference compressor is CPU-only. | M11: a possible third codec, adding a **GPU** GDeflate encoder, which would be novel |
| **crush-gpu** ([docs.rs](https://docs.rs/crate/crush-gpu/latest)) | The closest existing project: Rust + wgpu + WGSL, GDeflate-inspired tiles, GPU decompression only (compression runs on the CPU). Its README reports wgpu decompression in the hundreds of MiB/s. | Benchmark comparison target. Our differentiator is GPU compression. |
| **DietGPU** — Meta ([GitHub](https://github.com/facebookresearch/dietgpu)), MIT | GPU rANS entropy encoder **and** decoder, operating at hundreds of GB/s on an A100. Designed to also serve as the entropy stage after LZ or RLE matching. | M11 entropy stage |
| **ryg_rans** + "Interleaved entropy coders" — Giesen ([GitHub](https://github.com/rygorous/ryg_rans), [arXiv 1402.3392](https://arxiv.org/abs/1402.3392)) | Public-domain rANS reference code, plus the interleaving technique that fills a SIMD/GPU group with independent coders. | M11 entropy stage (read first; it's the simplest) |
| **Recoil** — Lin et al., ICPP '23 ([arXiv 2306.12141](https://arxiv.org/abs/2306.12141)) | Decodes a single rANS stream in parallel by storing intermediate states as metadata, so the parallelism can match the decoder's hardware (a large GPU vs a small CPU). | M11, if the entropy stage should scale across very different GPUs |
| **Massively Parallel Huffman Decoding on GPUs** — Weißenberger & Schmidt, ICPP '18 | Parallel decoding of standard Huffman codes using their self-synchronizing property. | M11, if Huffman is chosen over rANS |
| **Light Loss-Less (LLL)** — Funasaka, Nakano & Ito, 2016 | A format designed from scratch for parallel GPU decompression, with a ratio comparable to LZSS and LZW. | Design inspiration for GLZ (M6) |

Expected outcome: no existing project combines **GPU compression and decompression**, **cross-vendor** (wgpu: Metal + D3D12 + Vulkan), and **open source**. Most published GPU compressors are CUDA-only. That combination is this project's reason to exist, so protect it when making tradeoffs.

## 10. Resolved questions

1. **Target data:** a general-purpose core with automatic per-chunk filter selection (none, shuffle, delta), so numeric data benefits without a separate mode. See §4a and M7.
2. **Ratio vs throughput:** throughput first, with compression levels the way zstd offers them. Level 1 is fast greedy; higher levels trade speed for ratio. See M9.
3. **C API / FFI:** a non-goal.
4. **License:** MIT NON-AI (already in the repository).
5. **Random access:** required. Any byte range can be decompressed without decoding the whole file. See §4 and M1/M2/M8.