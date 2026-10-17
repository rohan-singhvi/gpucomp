# gpucomp: final benchmark report

This report answers the two questions in the [plan](../plan.md). How fast, and how well, does gpucomp compress and decompress on a GPU? And does GPU compression beat multi-threaded CPU compression?

Every table comes from the newest recorded benchmark run. `gpucomp bench --report` regenerates this page from `bench/REPORT.template.md`, which holds the prose, and `bench/results/*.json`, which holds the data. [BENCHMARKS.md](../BENCHMARKS.md) has the history of every run, and [DECISIONS.md](../DECISIONS.md) has the reasoning behind the design, including the experiments that didn't pay off.

## Summary

- **Both directions work on the GPU, and the files are exact.** Every GPU-compressed file is byte-for-byte the file its CPU twin writes, and every encoder round-trips through every decoder.
- **Ratio.** On Silesia in 64 KiB chunks, GLZ-E (GLZ plus Huffman coding) reaches 2.50× at level 1 and 2.75× at level 3. Level 3 matches zstd -1 on the same independent chunks (2.76×). LZ4-format output is 1.99× to 2.32× by level, against lz4_flex's 2.04×.
- **In memory, the fastest LZ4 compressor here is lz4_flex plus the GPU.** With lz4_flex compressing 75% of the chunks and the GPU the other 25% at the same time (`--hybrid --gpu-share 0.25`), LZ4 compression runs 1.15–1.3× faster than multi-threaded lz4_flex alone, at the same ratio. The split is fixed, so the output is deterministic. From a file, in a one-shot command, it is no faster than the CPU alone on this machine: the GPU's start-up takes as long as the CPU needs for Silesia.
- **From file to file, gpucomp's CPU path beats zstd.** The `gpucomp` command streams: it reads, compresses on every core and writes at the same time. It compresses Silesia from file to file faster than zstd -1 with all its threads, at LZ4's ratio, and decompresses it 2.7–3× faster than zstd, which decodes a file on one thread. The GPU's file paths pipeline their batches, but still take about 2.5× as long as the CPU's.
- **The GPU alone still loses.** At level 1 it compresses at about half the speed of multi-threaded lz4_flex or zstd -1 end to end. At equal ratio, zstd -1 compresses 4.5–5× faster than GLZ-E level 3 even with the CPU and GPU together.
- **CPU and GPU, identical output.** Every chunk encodes to the same bytes on either device, so `--hybrid` shares one input between the GPU and every CPU core by their measured speeds and still writes the GPU-only file: 25–35% faster than the GPU alone at level 1.
- **Same algorithm: the GPU leads by about 1.3×** over a CPU implementation writing the same files on all 12 cores. The CPU can skip work the GPU's parallel parse needs: it computes matches only where its serial parse looks.
- **Decompression is the CPU's.** The GPU decodes at about 2.7–3.8 GB/s end to end. On all cores, lz4_flex and gpucomp's own LZ4 decoder both decode LZ4 at about 13 GB/s, gpucomp's decoders run GLZ at about 14 and GLZ-E at 9.3–9.8 (level 1 to 3), and zstd -1 in 64 KiB chunks decodes at 9.4–10.9 between runs. At the same ratio, GLZ-E level 3 now decodes about as fast as zstd -1. `--hybrid` leaves the GPU out of decoding on this machine: the CPU finishes before a GPU pass would pay off.
- **Numeric data** is where the format shines. With per-chunk filters, GLZ-E compresses sorted integers nearly 4× smaller than zstd -3 on the whole file, and at level 2 it compresses 64-bit timestamps over 3× smaller than zstd -19.
- **Not measured: a discrete GPU.** The plan's goal of beating the CPU is defined for one. See [Discrete GPUs vs unified memory](#discrete-gpus-vs-unified-memory).

## Setup

{{setup}}

- **Data.** The [Silesia corpus](https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia) (12 files, 202 MiB, also concatenated as `silesia/all`), four synthetic inputs of 256 MiB (zeros, random bytes, pseudo-English text, and a mix of the three), and four numeric inputs of 256 MiB.
- **Chunks.** Every file uses 64 KiB independent chunks, so any byte range decodes on its own.
- **Throughput** is GB/s (10^9 bytes per second) of uncompressed data, the median of 10 runs after 2 warm-ups. **Ratio** is original size ÷ compressed size.
- **End to end** on the GPU starts and ends in host memory: upload, kernels, readback, and for compression packing the chunks into the file. It's what an application sees. Compression runs in batches of 16 MiB, two in flight, so uploads and readbacks overlap another batch's kernels.
- **GPU kernel** is the same work with the data already on the GPU. wgpu's Metal backend advertises timestamp queries but returns zeros, so kernel times are the wall clock around submit and wait.
- **CPU** rows use every core through rayon unless they say "1 thread". A *twin* is the CPU encoder that writes byte-for-byte the same file as the GPU encoder.
- **CPU+GPU** rows (`--hybrid`) share the chunks between the GPU and every CPU core, by measured speed. The compressed file is identical to the GPU-only one.
- Multi-threaded rows vary by about ±5–10% between runs on this laptop with background activity; [BENCHMARKS.md](../BENCHMARKS.md) notes each run's conditions. This run had little background load. Between runs, the in-memory lz4_flex row has ranged from about 5 to 5.8 GB/s; a best-of-ten measurement on an idle machine reaches about 7.
- **From file to file** rows run the way the `gpucomp` command does: read the input file, compress or decompress, write the output file, with the files in the page cache. The GPU rows create their GPU context and coder on every run, as one command does.
- **zstd** runs in-process through the `zstd` crate. "64 KiB chunks" compresses each chunk as its own frame on all threads, which gives the same random access as gpucomp. "Whole file" is one frame compressed by zstd's worker threads, like `zstd -T0`; it has no random access, and it decodes on one thread.

## Head to head

{{versus}}

## Compression: ratio vs throughput

Fastest first. ◆ marks the Pareto frontier: no other path is both faster and smaller.

{{compress}}

## Decompression

{{decompress}}

## From file to file

{{file-to-file}}

- **gpucomp's CPU path streams.** A reader thread reads groups of 256 KiB, rayon compresses or decompresses several groups at once on every core, and the calling thread writes the results in order. Small groups, many in flight, keep the working set in cache; the stream compresses as fast as the in-memory functions, so reading and writing cost almost nothing. The files are byte-for-byte the in-memory ones.
- **lz4_flex + GPU** (`--gpu-share`) streams too: the CPU compresses the head of the file while the GPU starts up on its own thread, then reads and compresses the tail. It's still slower than the CPU alone here, and more so in a fresh process than in this benchmark, which keeps the GPU driver warm. A new process needs 30–45 ms before its first GPU result, and on larger files the CPU slows by about a sixth while the GPU runs (DECISIONS "Streamed fixed split").
- **zstd -1** compresses with its own worker threads and decodes a frame on one thread, as `zstd -1 -T0` and `zstd -d` do. Its files are about 30% smaller.
- **The GPU path pipelines its batches.** Compression alternates 16 MiB batches between two sets of buffers: the next batch uploads, and the one before is read back and written, while the GPU encodes. Decompression writes each batch to the file straight from the GPU's readback buffer. About 20 ms still go to creating the GPU context and compiling the kernels, and the file I/O shares a thread with the GPU's submissions. DECISIONS.md ("Streaming CLI paths", "Pipelined GPU batches") has the breakdown.

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

Ratios, then the GPU's end-to-end GB/s with filter selection (`--filters auto`). The filters are byte-shuffle, delta and delta-shuffle over 4-byte elements (level 1), chosen per chunk.

{{numeric}}

Filters plus entropy coding win clearly where a transform exposes structure. Delta-shuffle turns sorted integers and audio samples into small gaps whose high bytes form long runs of zeros, and byte-shuffle groups the slowly changing bytes of floats. Order-0 Huffman coding then exploits what LZ matching alone can't. zstd stays ahead on the timestamps and float points at level 1, which tries only 4-byte elements. From level 2, 8-byte delta-shuffle changes that for timestamps: GLZ-E reached 12.6× against zstd -19's 3.90× on 32 MiB inputs, and 1.49× on the float points against 1.41× (DECISIONS "Delta-shuffle filter"). Levels 2 and 3 with filters aren't part of this benchmark.

## Where the time goes

Each kernel's time is the difference between timing the kernels up to it and up to the one before, with the data on the GPU. "Transfers and host work" is the rest of the end-to-end time: upload, readback, packing, and host-side copies. A stage much shorter than the kernels before it is within their timing noise, and a negative difference shows as 0.

{{profile}}

- **Match finding is now the largest level-1 kernel.** Each chunk's workgroup steps through its positions 128 at a time, with two barriers per block; the time splits into hashing and the table, loading each candidate's first word, and extending matches. Halving its workgroup memory changed nothing, so occupancy is not the limit. The parse copies the path its speculative walks recorded, instead of walking it twice. Emit writes each sequence's bytes.
- **Levels 2 and 3 are dominated by the deep-match pass,** about two thirds of level 3. Each chain link carries a tag of its candidate's first bytes, so the walk skips hash collisions without loading them. Following 16 chain links per position, each a dependent read of earlier input, is bound by memory latency.
- **GLZ-E's entropy stage** adds about a fifth to end-to-end encoding and decoding time. Its largest part is writing (encode) or reading (decode) the bitstreams lane by lane.
- **Transfers and host work** take 10–17 ms on 202 MiB: 14–15% of level-1 compression and 17–21% of decompression. Compression pipelines its batches, so most uploads and readbacks overlap kernels; what's left is the first upload, the last readback, and the host's waits between submissions. The copies themselves are cheap on unified memory: a copy into a mapped staging buffer runs at memcpy speed, and the GPU copies 202 MiB between buffers in about 2 ms. Most of the rest is writing the output into fresh memory, which page-faults at about 20 GB/s, and the waits between submissions.

## Platform ceilings

{{platform}}

## Does GPU compression beat the CPU?

**Only as a second processor, on this machine.** On its own, the GPU compresses at about half the speed of multi-threaded lz4_flex or zstd -1, end to end. Working next to the CPU in a running program, it pays: lz4_flex on 75% of the chunks plus the GPU on the other 25% compress LZ4 1.15–1.3× faster than lz4_flex alone, at the same ratio. In a one-shot command it doesn't, because the GPU's start-up costs about as much as the CPU needs. The plan's goal is defined for a discrete GPU, which hasn't been measured; the next section covers what would change there. Three causes account for the GPU's own gap on this machine, in order of size.

1. **The parallel algorithm does more work.** The GPU finds a match at every position, because its parallel parse needs them all, then parses and emits in parallel passes over the whole chunk. A serial parse needs matches only where it looks, about 40% of positions, and lz4_flex also skips ahead quickly through data that doesn't compress. The CPU twin, which writes the same files as the GPU, uses the first of these shortcuts and is still well behind lz4_flex on the same cores. The GPU runs its algorithm about 1.3× faster than that twin.
2. **The kernels are latency-bound, not bandwidth-bound.** The trivial XOR kernel in the platform table streams data at about 90–110 GB/s; the level-1 encoder kernels process about 3.5 GB/s. The time goes to barrier-synchronised match finding, serial parse walks, divergence between lanes, and dependent memory reads in the deep-match pass. DECISIONS.md records the experiments that tried to shift each of these, including the many that didn't help.
3. **Transfers and host work** cost 14–15% of level-1 compression and 17–21% of decompression, even though this GPU shares memory with the CPU. The copies run near memcpy speed; the time goes to page-faulting fresh output memory, staging copies on the host and waits between submissions.

**Where the GPU does win:** as a second processor next to the CPU in a running program (compression with lz4_flex, or with identical output via the twin); against the same algorithm on the CPU, by about 1.3×; and against single-threaded decoders, such as zstd on a whole-file frame, which also has no random access.

## Discrete GPUs vs unified memory

Everything above was measured on Apple Silicon, where the CPU and GPU share one memory (LPDDR5X, about 273 GB/s on the M4 Pro). Here "upload" and "readback" are copies between ordinary and GPU-visible buffers in the same memory, not bus transfers. The 7–11 GB/s in the platform table includes a fresh staging buffer per call, page-faulted on first use; the coders reuse theirs, and their copies run near memcpy speed. A zero-copy path through mapped storage buffers (plan M11) would save only the GPU-side copies, about 3 ms of a 70 ms level-1 compression (DECISIONS "Streaming CLI paths").

On a discrete GPU, the data crosses PCIe instead. PCIe 4.0 x16 carries at most about 32 GB/s each way, and less in practice. Compression uploads the input and reads back only the compressed chunks; decompression does the reverse. Both coders keep two batches in flight, so a transfer can overlap another batch's kernels. Here that cut the encoder's transfers and host work from about 15 ms to 8 (DECISIONS "Pipelined GPU batches"); the copies are CPU work, and they compete with the rest of the host side. Over PCIe, a DMA engine does the copying, so overlap should pay off more, but the link's bandwidth still bounds end-to-end speed.

The hybrid scheduler measures both devices on every run, so on a discrete GPU it shifts the split toward whichever is faster, and keeps the GPU out of jobs too short to repay a pass.

The kernels should gain more than the transfers lose. A desktop GPU has two to six times the shader ALUs, and roughly 2–4× the memory bandwidth, of this 20-core integrated GPU. The kernels here are limited by latency and occupancy, which more cores in flight help directly. Whether that is enough to beat a desktop CPU, which also has more cores than this laptop, is the open question. The benchmark answers it without code changes. Run the reproduction command below on a Windows machine with D3D12 or Vulkan, and this report gains a section for that adapter. Those backends also return real GPU timestamps, so its kernel rows will be true GPU time.

## Reproducing

```sh
scripts/fetch_corpus.sh       # downloads Silesia into testdata/corpus/silesia
cargo run --release -p cli -- bench --label <label> --corpus testdata/corpus/silesia --record --report
```

The run takes about half an hour on the machine above. `--record` saves the run as JSON in `bench/results/`, and `--report` regenerates `BENCHMARKS.md` and this page. To change the prose, edit `bench/REPORT.template.md`, then run `gpucomp bench --report-only --report`.
