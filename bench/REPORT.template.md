# gpucomp: final benchmark report

This report answers the two questions in the [plan](../plan.md). How fast, and how well, does gpucomp compress and decompress on a GPU? And does GPU compression beat multi-threaded CPU compression?

Every table comes from the newest recorded benchmark run. `gpucomp bench --report` regenerates this page from `bench/REPORT.template.md`, which holds the prose, and `bench/results/*.json`, which holds the data. [BENCHMARKS.md](../BENCHMARKS.md) has the history of every run, and [DECISIONS.md](../DECISIONS.md) has the reasoning behind the design, including the experiments that didn't pay off.

## Summary

- **Both directions work on the GPU, and the files are exact.** Every GPU-compressed file is byte-for-byte the file its CPU twin writes, and every encoder round-trips through every decoder.
- **Ratio.** On Silesia in 64 KiB chunks, GLZ-E (GLZ plus Huffman coding) reaches 2.50× at level 1 and 2.75× at level 3. Level 3 matches zstd -1 on the same independent chunks (2.76×). LZ4-format output is 1.99× to 2.32× by level, against lz4_flex's 2.04×.
- **The fastest LZ4 compressor here is lz4_flex plus the GPU.** With lz4_flex compressing 70% of the chunks and the GPU the other 30% at the same time (`--hybrid --gpu-share 0.3`), LZ4 compression runs about 1.3× faster than multi-threaded lz4_flex alone, at the same ratio. The split is fixed, so the output is deterministic.
- **The GPU alone still loses.** At level 1 it compresses at about half the speed of multi-threaded lz4_flex or zstd -1 end to end. At equal ratio, zstd -1 is about 4.5× faster than GLZ-E level 3 even with the CPU and GPU together.
- **CPU and GPU, identical output.** Every chunk encodes to the same bytes on either device, so `--hybrid` shares one input between the GPU and every CPU core by their measured speeds and still writes the GPU-only file: 36–46% faster than the GPU alone.
- **Same algorithm: the GPU leads by about 1.4×** over a CPU implementation writing the same files on all 12 cores. The CPU can skip work the GPU's parallel parse needs: it computes matches only where its serial parse looks.
- **Decompression is the CPU's.** The GPU decodes at about 2.7–3.5 GB/s end to end. On all cores, lz4_flex decodes LZ4 at about 14 GB/s, gpucomp's own decoders run GLZ at about 12.6 and GLZ-E at about 6.9, and zstd -1 in 64 KiB chunks at about 10.9. `--hybrid` leaves the GPU out of decoding on this machine: the CPU finishes before a GPU pass would pay off.
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
- **CPU+GPU** rows (`--hybrid`) share the chunks between the GPU and every CPU core, by measured speed. The compressed file is identical to the GPU-only one.
- Multi-threaded rows vary by about ±5–10% between runs on this laptop with background activity; [BENCHMARKS.md](../BENCHMARKS.md) notes each run's conditions.
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

- **Match finding is now the largest level-1 kernel.** Each chunk's workgroup steps through its positions 128 at a time, with two barriers per block; the time splits into hashing and the table, loading each candidate's first word, and extending matches. Halving its workgroup memory changed nothing, so occupancy is not the limit. The parse copies the path its speculative walks recorded, instead of walking it twice. Emit writes each sequence's bytes.
- **Levels 2 and 3 are dominated by the deep-match pass,** about two thirds of level 3. Each chain link carries a tag of its candidate's first bytes, so the walk skips hash collisions without loading them. Following 16 chain links per position, each a dependent read of earlier input, is bound by memory latency.
- **GLZ-E's entropy stage** adds about a fifth to end-to-end encoding and decoding time. Its largest part is writing (encode) or reading (decode) the bitstreams lane by lane.
- **Transfers and host work** are a roughly fixed 12–18 ms on 202 MiB: 14–19% of level-1 compression and 21–30% of decompression. Their share grows as the kernels get faster.

## Platform ceilings

{{platform}}

## Does GPU compression beat the CPU?

**Only as a second processor, on this machine.** On its own, the GPU compresses at about half the speed of multi-threaded lz4_flex or zstd -1, end to end. Working next to the CPU, it pays: lz4_flex on 70% of the chunks plus the GPU on the other 30% compress LZ4 about 1.3× faster than lz4_flex alone, at the same ratio. The plan's goal is defined for a discrete GPU, which hasn't been measured; the next section covers what would change there. Three causes account for the GPU's own gap on this machine, in order of size.

1. **The parallel algorithm does more work.** The GPU finds a match at every position, because its parallel parse needs them all, then parses and emits in parallel passes over the whole chunk. A serial parse needs matches only where it looks, about 40% of positions, and lz4_flex also skips ahead quickly through data that doesn't compress. The CPU twin, which writes the same files as the GPU, uses the first of these shortcuts and is still well behind lz4_flex on the same cores. The GPU runs its algorithm about 1.4× faster than that twin.
2. **The kernels are latency-bound, not bandwidth-bound.** The trivial XOR kernel in the platform table streams data at about 90 GB/s; the level-1 encoder kernels process about 3.5 GB/s. The time goes to barrier-synchronised match finding, serial parse walks, divergence between lanes, and dependent memory reads in the deep-match pass. DECISIONS.md records the experiments that tried to shift each of these, including the many that didn't help.
3. **Host↔GPU copies** cost 14–19% of level-1 compression and 21–30% of decompression, even though this GPU shares memory with the CPU. They go through wgpu's mapped staging buffers at 8–11 GB/s, far below the CPU's ~70 GB/s memcpy.

**Where the GPU does win:** as a second processor next to the CPU (compression with lz4_flex, or with identical output via the twin); against the same algorithm on the CPU, by about 1.4×; and against single-threaded decoders, such as zstd on a whole-file frame, which also has no random access.

## Discrete GPUs vs unified memory

Everything above was measured on Apple Silicon, where the CPU and GPU share one memory (LPDDR5X, about 273 GB/s on the M4 Pro). Here "upload" and "readback" are copies between ordinary and GPU-visible buffers in the same memory, not bus transfers. The 7–11 GB/s in the platform table is the cost of wgpu's staging path, not of the hardware. A zero-copy path through mapped GPU buffers could remove most of it (plan M11).

On a discrete GPU, the data crosses PCIe instead. PCIe 4.0 x16 carries at most about 32 GB/s each way, and less in practice. Compression uploads the input and reads back only the compressed chunks; decompression does the reverse. M8 splits large files into batches and keeps two in flight, so a transfer can overlap another batch's kernels. That gained little here, because the copies are CPU work that competes with the rest of the host side (DECISIONS "Decoder: wait for each batch's own submission"). Over PCIe, a DMA engine does the copying, so overlap should pay off more, but the link's bandwidth still bounds end-to-end speed.

The hybrid scheduler measures both devices on every run, so on a discrete GPU it shifts the split toward whichever is faster, and keeps the GPU out of jobs too short to repay a pass.

The kernels should gain more than the transfers lose. A desktop GPU has two to six times the shader ALUs, and roughly 2–4× the memory bandwidth, of this 20-core integrated GPU. The kernels here are limited by latency and occupancy, which more cores in flight help directly. Whether that is enough to beat a desktop CPU, which also has more cores than this laptop, is the open question. The benchmark answers it without code changes. Run the reproduction command below on a Windows machine with D3D12 or Vulkan, and this report gains a section for that adapter. Those backends also return real GPU timestamps, so its kernel rows will be true GPU time.

## Reproducing

```sh
scripts/fetch_corpus.sh       # downloads Silesia into testdata/corpus/silesia
cargo run --release -p cli -- bench --label <label> --corpus testdata/corpus/silesia --record --report
```

The run takes about half an hour on the machine above. `--record` saves the run as JSON in `bench/results/`, and `--report` regenerates `BENCHMARKS.md` and this page. To change the prose, edit `bench/REPORT.template.md`, then run `gpucomp bench --report-only --report`.
