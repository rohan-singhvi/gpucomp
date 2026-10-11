# gpucomp: final benchmark report

This report answers the two questions in the [plan](../plan.md). How fast, and how well, does gpucomp compress and decompress on a GPU? And does GPU compression beat multi-threaded CPU compression?

Every table comes from the newest recorded benchmark run. `gpucomp bench --report` regenerates this page from `bench/REPORT.template.md`, which holds the prose, and `bench/results/*.json`, which holds the data. [BENCHMARKS.md](../BENCHMARKS.md) has the history of every run, and [DECISIONS.md](../DECISIONS.md) has the reasoning behind the design, including the experiments that didn't pay off.

## Summary

- **Both directions work on the GPU, and the files are exact.** Every GPU-compressed file is byte-for-byte the file its CPU twin writes, and every encoder round-trips through every decoder.
- **Ratio.** On Silesia in 64 KiB chunks, GLZ-E (GLZ plus Huffman coding) reaches 2.50× at level 1 and 2.75× at level 3. Level 3 matches zstd -1 on the same independent chunks (2.76×). LZ4-format output is 1.99× to 2.32× by level, against lz4_flex's 2.04×.
- **Compression speed: the GPU loses on this machine.** At level 1, multi-threaded lz4_flex and zstd -1 compress more than twice as fast as the GPU end to end. At equal ratio, zstd -1 is more than 8× faster than GPU GLZ-E level 3.
- **Same algorithm, the GPU wins.** Running the algorithm that produces gpucomp's files, the GPU is 2–5× faster than all 12 CPU cores. The CPU baselines win because their serial algorithms do several times less work per byte.
- **Decompression: the GPU loses too.** The GPU decodes at about 2.7–3.4 GB/s end to end. A multi-threaded CPU decoding the same independent chunks is 4–5× faster with lz4_flex or zstd, and about as fast with gpucomp's own GLZ-E decoder. The GPU beats only single-threaded decoders, such as zstd on a whole-file frame.
- **Numeric data** is where the format shines. With per-chunk filters, GLZ-E compresses sorted integers 3× smaller than zstd -3 on the whole file.
- **Not measured: a discrete GPU.** The plan's goal of beating the CPU is defined for one. See [Discrete GPUs vs unified memory](#discrete-gpus-vs-unified-memory).

## Setup

{{setup}}

- **Data.** The [Silesia corpus](https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia) (12 files, 202 MiB, also concatenated as `silesia/all`), four synthetic inputs of 256 MiB (zeros, random bytes, pseudo-English text, and a mix of the three), and four numeric inputs of 256 MiB.
- **Chunks.** Every file uses 64 KiB independent chunks, so any byte range decodes on its own.
- **Throughput** is GB/s (10^9 bytes per second) of uncompressed data, the median of 10 runs after 2 warm-ups. **Ratio** is original size ÷ compressed size.
- **End to end** on the GPU starts and ends in host memory: upload, kernels, readback, and for compression packing the chunks into the file. It's what an application sees.
- **GPU kernel** is the same work with the data already on the GPU. wgpu's Metal backend advertises timestamp queries but returns zeros, so kernel times are the wall clock around submit and wait.
- **CPU** rows use every core through rayon unless they say "1 thread". A *twin* is the CPU encoder that writes byte-for-byte the same file as the GPU encoder.
- **zstd** runs in-process through the `zstd` crate. "64 KiB chunks" compresses each chunk as its own frame on all threads, which gives the same random access as gpucomp. "Whole file" is one frame compressed by zstd's worker threads, like `zstd -T0`; it has no random access, and it decodes on one thread.

## Head to head

{{versus}}

## Compression: ratio vs throughput

Fastest first. ◆ marks the Pareto frontier: no other path is both faster and smaller.

{{compress}}

## Decompression

{{decompress}}

## Silesia, file by file

Compression, as ratio · end-to-end GB/s:

{{files-compress}}

Decompression, end-to-end GB/s:

{{files-decompress}}

## Synthetic inputs

Compression, as ratio · end-to-end GB/s:

{{synthetic-compress}}

Decompression, end-to-end GB/s:

{{synthetic-decompress}}

## Numeric data

Ratios, then the GPU's end-to-end GB/s with filter selection (`--filters auto`). The filters are byte-shuffle and delta over 4-byte elements, chosen per chunk.

{{numeric}}

Filters plus entropy coding win clearly where a transform exposes structure. Delta turns sorted integers into small, repetitive gaps, and byte-shuffle groups the slowly changing high bytes of audio samples. Order-0 Huffman coding then exploits what LZ matching alone can't. zstd stays ahead on the timestamps and float points at level 1. Deeper matching changes that for timestamps: GLZ-E with filters at level 3 reached 8.96× against zstd's 3.90× on 32 MiB inputs (DECISIONS "Numeric data"). Level 3 with filters isn't part of this benchmark.

## Where the time goes

Each kernel's time is the difference between timing the kernels up to it and up to the one before, with the data on the GPU. "Transfers and host work" is the rest of the end-to-end time: upload, readback, packing, and host-side copies. A stage much shorter than the kernels before it is within their timing noise, and a negative difference shows as 0.

{{profile}}

- **Level-1 encoding is three kernels of similar weight.** Match finding is limited by occupancy: its 16 KB workgroup hash table caps how many workgroups run at once. The parse walks each of 128 segments per chunk serially. Emit writes each sequence's bytes.
- **Levels 2 and 3 are dominated by the deep-match pass,** about two thirds of level 3. Following 16 chain links per position, each a dependent read of earlier input, is bound by memory latency.
- **GLZ-E's entropy stage** adds about a fifth to end-to-end encoding and decoding time. Its largest part is writing (encode) or reading (decode) the bitstreams lane by lane.
- **Transfers and host work** are a roughly fixed 13–19 ms on 202 MiB: 12–16% of level-1 compression and 20–30% of decompression.

## Platform ceilings

{{platform}}

## Does GPU compression beat the CPU?

**No, not on this machine.** The fastest GPU compression path runs at under half the speed of multi-threaded lz4_flex or zstd -1, end to end. The plan's goal is defined for a discrete GPU, which hasn't been measured; the next section covers what would change there. Three causes account for the gap on this machine, in order of size.

1. **The parallel algorithm does more work.** gpucomp finds a match at every position, then parses and emits in parallel passes over the whole chunk. lz4_flex makes one pass, probes once per position it visits, and skips ahead quickly through data that doesn't compress. The CPU twin, which runs gpucomp's algorithm, is about 5× slower than lz4_flex on the same cores. The GPU runs that algorithm 2–2.5× faster than the twin, which isn't enough to close a 5× gap.
2. **The kernels are latency-bound, not bandwidth-bound.** The trivial XOR kernel in the platform table streams data at about 100 GB/s; the level-1 encoder kernels process under 3 GB/s. The time goes to occupancy (the workgroup hash table), serial parse walks, divergence between lanes, and dependent memory reads in the deep-match pass. DECISIONS.md records the experiments that tried to shift each of these, including the many that didn't help.
3. **Host↔GPU copies** cost 12–16% of level-1 compression and 20–30% of decompression, even though this GPU shares memory with the CPU. They go through wgpu's mapped staging buffers at 7–11 GB/s, far below the CPU's 65 GB/s memcpy.

**Where the GPU does win:** against the same algorithm on the CPU (2–5×, growing with level), and against single-threaded decoders, such as zstd on a whole-file frame, which also has no random access.

## Discrete GPUs vs unified memory

Everything above was measured on Apple Silicon, where the CPU and GPU share one memory (LPDDR5X, about 273 GB/s on the M4 Pro). Here "upload" and "readback" are copies between ordinary and GPU-visible buffers in the same memory, not bus transfers. The 7–11 GB/s in the platform table is the cost of wgpu's staging path, not of the hardware. A zero-copy path through mapped GPU buffers could remove most of it (plan M11).

On a discrete GPU, the data crosses PCIe instead. PCIe 4.0 x16 carries at most about 32 GB/s each way, and less in practice. Compression uploads the input and reads back only the compressed chunks; decompression does the reverse. M8 splits large files into batches and keeps two in flight, so a transfer can overlap another batch's kernels. That gained little here, because the copies are CPU work that competes with the rest of the host side (DECISIONS "Decoder: wait for each batch's own submission"). Over PCIe, a DMA engine does the copying, so overlap should pay off more, but the link's bandwidth still bounds end-to-end speed.

The kernels should gain more than the transfers lose. A desktop GPU has two to six times the shader ALUs, and roughly 2–4× the memory bandwidth, of this 20-core integrated GPU. The kernels here are limited by latency and occupancy, which more cores in flight help directly. Whether that is enough to beat a desktop CPU, which also has more cores than this laptop, is the open question. The benchmark answers it without code changes. Run the reproduction command below on a Windows machine with D3D12 or Vulkan, and this report gains a section for that adapter. Those backends also return real GPU timestamps, so its kernel rows will be true GPU time.

## Reproducing

```sh
scripts/fetch_corpus.sh       # downloads Silesia into testdata/corpus/silesia
cargo run --release -p cli -- bench --label <label> --corpus testdata/corpus/silesia --record --report
```

The run takes about half an hour on the machine above. `--record` saves the run as JSON in `bench/results/`, and `--report` regenerates `BENCHMARKS.md` and this page. To change the prose, edit `bench/REPORT.template.md`, then run `gpucomp bench --report-only --report`.
