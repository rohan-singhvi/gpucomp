# gpucomp

GPU-accelerated LZ compression that isn't CUDA-only. Compress and decompress on Apple, AMD, Intel and NVIDIA GPUs with Rust and [wgpu](https://wgpu.rs) (WGSL compute shaders on Metal, D3D12 and Vulkan).

Data is split into independent chunks (64 KiB by default), so every chunk can be encoded and decoded in parallel, and any byte range can be read back without decoding the whole file.

> **Status:** milestones M0–M10 of 11 are done; M11 (experiments) is next. The file format may still change.
> On the Apple M4 Pro it's measured on, the GPU alone compresses at about half the speed of multi-threaded CPU compression. As a second processor next to lz4_flex, it makes LZ4 compression about 1.3× faster than lz4_flex alone. See [Performance](#performance) and the [final benchmark report](bench/REPORT.md).

## Features

- **Three codecs, both directions on CPU and GPU.**
  - **LZ4 block** (`lz4`): standard LZ4 blocks, readable by any LZ4 block decoder.
  - **GLZ** (`glz`): an LZ77 variant built for GPUs, with separate fixed-width streams. An optional *dependency elimination* mode (`--independent-groups`) lets the decoder run more matches in parallel, at some cost in ratio.
  - **GLZ-E** (`glze`): GLZ with its byte streams Huffman-coded in up to 32 independent lanes each; about 28% smaller than GLZ on Silesia.
- **Identical output on GPU and CPU.** The GPU encoder produces byte-for-byte the same file as its CPU twin, and every encoder round-trips through every decoder. The test suite enforces both.
- **CPU and GPU together.** `--hybrid` splits one file's chunks between the GPU and every CPU core by their measured speeds. Because each chunk encodes to the same bytes on either device, the file is identical to `--gpu` alone, however the work was split. Decompression shares the same way, and the GPU joins only when it helps.
- **lz4_flex plus the GPU.** `--hybrid --gpu-share 0.3` gives the GPU a fixed share of the chunks and the rest to the CPU's encoder: lz4_flex for LZ4, the CPU twin otherwise. The output is deterministic for the same settings, and for LZ4 it is the fastest compressor measured here.
- **Random access.** `decompress --offset N --length M` reads only the header, the chunk table and the chunks that overlap the range.
- **Per-chunk filters.** Optional byte-shuffle and delta filters for numeric data. `--filters auto` picks one per chunk by trial-encoding a sample, on CPU or GPU.
- **Large files in bounded memory.** GPU compression and decompression stream file to file in batches that fit a GPU memory budget (`--gpu-memory`, default 3 GiB). A 4 GiB file round-trips with about 480 MB of host memory.
- **Checksums.** Optional per-chunk checksums, verified with `--verify`.

## Quick start

You need Rust (the toolchain is pinned in `rust-toolchain.toml` and installed automatically by `rustup`) and a GPU that wgpu supports.

```sh
cargo build --release
alias gpucomp=target/release/gpucomp

gpucomp info                                    # show the GPU adapter, backend and limits

gpucomp compress --gpu big.bin big.gpcz         # compress on the GPU (LZ4 block)
gpucomp compress --gpu --codec glz --filters auto big.bin big.gpcz
gpucomp compress --gpu --level 3 big.bin big.gpcz  # smaller, slower
gpucomp compress --gpu --hybrid --codec glze big.bin big.gpcz   # GPU and CPU together
gpucomp compress --gpu --hybrid --gpu-share 0.3 big.bin big.gpcz  # lz4_flex + GPU, fastest LZ4
gpucomp compress big.bin big.gpcz               # CPU, multi-threaded (lz4_flex)

gpucomp info big.gpcz                           # codec, chunks, ratio, filters used
gpucomp decompress --gpu big.gpcz big.out       # whole file on the GPU
gpucomp decompress --offset 1000000 --length 4096 big.gpcz slice.bin   # one range
```

Useful options (see `gpucomp <command> --help` for all of them):

| Option | Meaning |
|---|---|
| `--codec lz4\|glz\|glze\|stored` | Block codec (default `lz4`). `glze` adds Huffman coding to GLZ: ~28% smaller, decodes a little slower than GLZ |
| `--level 1\|2\|3` | Compression level: more match candidates per position (1, 4, 16) for a smaller file; default 1 |
| `--chunk-size 64K` | Chunk size, a power of two from 4K to 1M |
| `--filters none\|auto\|exhaustive` | Per-chunk filter selection |
| `--checksum` / `--verify` | Store / check per-chunk checksums |
| `--gpu`, `--backend metal\|dx12\|vulkan` | Run on the GPU, optionally forcing a backend |
| `--hybrid` | With `--gpu`: use every CPU core too (same output; reads the whole input into memory) |
| `--gpu-share 0.3` | With `--hybrid`: give the GPU this fixed share of the chunks, and the CPU's encoder (lz4_flex for LZ4) the rest. Deterministic, but not the GPU-only file |
| `--gpu-memory 512M` | Cap the GPU memory per batch |

## Library use

The `gpu` crate exposes the encoder and decoder. The `cpu` crate has the same operations on the CPU (`cpu::container::{compress, decompress, decompress_range}`).

```rust
use gpu::decode::GpuDecoder;
use gpu::encode::{GpuCompressOptions, GpuEncoder};
use gpu::{Context, ContextOptions};

let ctx = Context::new(&ContextOptions::default())?;

let encoder = GpuEncoder::new(&ctx, Default::default())?;
let file = encoder.compress(&ctx, &data, &GpuCompressOptions::default())?;

let decoder = GpuDecoder::new(&ctx);
assert_eq!(decoder.decompress(&ctx, &file, false)?, data);
```

For files larger than memory, use `GpuEncoder::compress_stream`, `GpuDecoder::decompress_stream` and `GpuDecoder::decompress_range`. To use the CPU and GPU together on an in-memory input, use `gpu::hybrid::compress` and `gpu::hybrid::decompress`.

## Performance

Apple M4 Pro (12 CPU cores, 20-core GPU), [Silesia corpus](https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia) (212 MB), 64 KiB chunks. GPU numbers are end to end: input in host memory, output back in host memory.

| Path | Ratio | Compress GB/s | Decompress GB/s |
|---|---|---|---|
| GPU, LZ4 (default) | 1.99× | 2.87 | 2.70 |
| GPU, LZ4 `--level 2` | 2.21× | 1.64 | 2.70 |
| GPU, LZ4 `--level 3` | 2.32× | 0.93 | 2.70 |
| GPU, GLZ | 1.95× | 2.84 | 3.48 |
| GPU, **GLZ-E** (Huffman-coded GLZ) | **2.50×** | 2.31 | 2.81 |
| GPU, GLZ-E `--level 3` | **2.75×** | 0.87 | 3.03 |
| CPU+GPU (`--hybrid`), LZ4 | 1.99× | 3.90 | 13.9 |
| CPU+GPU, GLZ-E | 2.50× | 3.16 | 6.42 |
| CPU+GPU, GLZ-E `--level 3` | 2.75× | 1.27 | 7.15 |
| **lz4_flex + GPU** (`--hybrid --gpu-share 0.3`), LZ4 | 2.04× | **7.57** | — |
| CPU, GLZ (gpucomp), 12 threads | 1.95× | 2.19 | 12.6 |
| CPU, GLZ-E (gpucomp), 12 threads | 2.50× | 1.68 | 6.85 |
| CPU `lz4_flex`, 12 threads | 2.04× | 5.8 | 14.2 |
| CPU zstd -1, 64 KiB chunks, 12 threads | 2.76× | 5.9 | 10.9 |
| CPU zstd -1, whole file (no random access) | 2.89× | 6.0 | 1.65 (1 thread) |

LZ4 decompression on the GPU is measured on the default level's file. zstd in 64 KiB chunks gives the same random access as gpucomp. CPU+GPU decompression leaves the GPU out on this machine, because the CPU finishes before a GPU pass would pay off. The lz4_flex + GPU file is an ordinary LZ4 file, so it decodes like the other LZ4 rows.

**On numeric data**, GLZ-E with `--filters auto` beats `zstd -19` on most inputs. Byte-shuffle groups slowly changing bytes, and the entropy stage then exploits them. 32 MiB inputs, 64 KiB chunks:

| Input | GLZ-E `--filters auto` | `--level 3` | zstd -1 | zstd -19 |
|---|---|---|---|---|
| f32 points | 1.42× | 1.56× | 1.15× | 1.41× |
| i16 audio | 1.19× | 1.24× | 1.04× | 1.04× |
| sorted u32 | 4.08× | 4.12× | 1.31× | 2.91× |
| u64 timestamps | 3.17× | 8.96× | 3.90× | 3.90× |

**Honest comparison:** on this machine, the GPU alone compresses at about half the speed of multi-threaded `lz4_flex` or zstd -1. With identical output, the CPU and GPU together reach about two thirds. The GPU pays as a second processor: lz4_flex on 70% of the chunks and the GPU on the rest compress LZ4 about 1.3× faster than lz4_flex alone, at the same ratio. zstd keeps its lead in ratio per unit of speed, and decoding belongs to the CPU: the GPU decodes at about 3 GB/s, the CPU at 7–14. The GPU's parallel algorithm needs a match at every position, while a serial parse needs them at only about 40%, so the GPU does more work per byte. Its kernels are limited by latency, not bandwidth: barrier-synchronised match finding, serial parse walks and dependent memory reads. Host↔GPU copies take another 14–30% end to end. A discrete GPU, against a typical desktop CPU, is the setting the project targets, but it hasn't been measured yet.

The [final benchmark report](bench/REPORT.md) has the full picture: per-file results, a ratio-vs-throughput table, a per-kernel profile, and a note on discrete GPUs vs unified memory. [BENCHMARKS.md](BENCHMARKS.md) has every run's history, and [DECISIONS.md](DECISIONS.md) has the reasoning behind design choices, including negative results.

## Repository layout

| Path | Contents |
|---|---|
| [crates/format](crates/format) | The `.gpcz` container: header, chunk table, range→chunk mapping. Spec in [FORMAT.md](crates/format/FORMAT.md) |
| [crates/cpu](crates/cpu) | CPU LZ4 and GLZ encoders and decoders, filters, the multi-threaded container |
| [crates/gpu](crates/gpu) | wgpu context, GPU encoder and decoder (WGSL kernels), GPU filters |
| [crates/bench](crates/bench) | Benchmark suites and the generators of `BENCHMARKS.md` and `bench/REPORT.md` (prose in `bench/REPORT.template.md`) |
| [crates/cli](crates/cli) | The `gpucomp` command-line tool |
| [plan.md](plan.md) | Goals and milestones |

## Development

```sh
cargo test --workspace                                   # unit, property and GPU tests (~1 min)
cargo test --release -p gpu --test stream -- --ignored   # 4 GiB streaming round trip
cargo clippy --workspace --all-targets -- -D warnings

scripts/fetch_corpus.sh                                  # download the Silesia/Canterbury corpora
gpucomp bench --corpus testdata/corpus/silesia           # benchmarks (add --record --report to log a run
                                                         # and regenerate BENCHMARKS.md and bench/REPORT.md)
```

GPU tests are skipped when no adapter is available. CI runs on macOS and Windows.

## License

[MIT NON-AI License](LICENSE).
