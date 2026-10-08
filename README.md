# gpucomp

GPU-accelerated LZ compression that isn't CUDA-only. Compress and decompress on Apple, AMD, Intel and NVIDIA GPUs with Rust and [wgpu](https://wgpu.rs) (WGSL compute shaders on Metal, D3D12 and Vulkan).

Data is split into independent chunks (64 KiB by default), so every chunk can be encoded and decoded in parallel, and any byte range can be read back without decoding the whole file.

> **Status:** work in progress (milestone M8 of 11). The file format may still change.
> GPU compression is not yet faster than multi-threaded CPU LZ4. See [Performance](#performance).

## Features

- **Two codecs, both directions on CPU and GPU.**
  - **LZ4 block** (`lz4`): standard LZ4 blocks, readable by any LZ4 block decoder.
  - **GLZ** (`glz`): an LZ77 variant built for GPUs, with separate fixed-width streams. An optional *dependency elimination* mode (`--independent-groups`) lets the decoder run more matches in parallel, at some cost in ratio.
- **Identical output on GPU and CPU.** The GPU encoder produces byte-for-byte the same file as its CPU twin, and every encoder round-trips through every decoder. The test suite enforces both.
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
gpucomp compress big.bin big.gpcz               # CPU, multi-threaded (lz4_flex)

gpucomp info big.gpcz                           # codec, chunks, ratio, filters used
gpucomp decompress --gpu big.gpcz big.out       # whole file on the GPU
gpucomp decompress --offset 1000000 --length 4096 big.gpcz slice.bin   # one range
```

Useful options (see `gpucomp <command> --help` for all of them):

| Option | Meaning |
|---|---|
| `--codec lz4\|glz\|stored` | Block codec (default `lz4`) |
| `--chunk-size 64K` | Chunk size, a power of two from 4K to 1M |
| `--filters none\|auto\|exhaustive` | Per-chunk filter selection |
| `--checksum` / `--verify` | Store / check per-chunk checksums |
| `--gpu`, `--backend metal\|dx12\|vulkan` | Run on the GPU, optionally forcing a backend |
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

For files larger than memory, use `GpuEncoder::compress_stream`, `GpuDecoder::decompress_stream` and `GpuDecoder::decompress_range`.

## Performance

Apple M4 Pro, [Silesia corpus](https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia) (212 MB), 64 KiB chunks, end to end (input in host memory, output back in host memory):

| Path | Ratio | GB/s |
|---|---|---|
| CPU `lz4_flex`, multi-threaded compress | 2.04× | 5.3 |
| CPU `lz4_flex`, multi-threaded decompress | — | 12.6 |
| GPU compress, LZ4 | 1.94× | 1.74 |
| GPU compress, LZ4, `--filters auto` | 1.99× | 1.09 |
| GPU compress, GLZ | 1.91× | 1.73 |
| GPU decompress, LZ4 (kernel only) | — | 3.3 |
| GPU decompress, GLZ (kernel only) | — | 4.7 |

The GPU doesn't beat a good multi-threaded CPU yet. The encoder's bottleneck is the parse, which runs one GPU thread per chunk. Host↔GPU transfers (6–10 GB/s here) also cap end-to-end throughput. Full results and history are in [BENCHMARKS.md](BENCHMARKS.md), and the reasoning behind design choices, including negative results, is in [DECISIONS.md](DECISIONS.md).

## Repository layout

| Path | Contents |
|---|---|
| [crates/format](crates/format) | The `.gpcz` container: header, chunk table, range→chunk mapping. Spec in [FORMAT.md](crates/format/FORMAT.md) |
| [crates/cpu](crates/cpu) | CPU LZ4 and GLZ encoders and decoders, filters, the multi-threaded container |
| [crates/gpu](crates/gpu) | wgpu context, GPU encoder and decoder (WGSL kernels), GPU filters |
| [crates/bench](crates/bench) | Benchmark suites and the `BENCHMARKS.md` generator |
| [crates/cli](crates/cli) | The `gpucomp` command-line tool |
| [plan.md](plan.md) | Goals and milestones |

## Development

```sh
cargo test --workspace                                   # unit, property and GPU tests (~1 min)
cargo test --release -p gpu --test stream -- --ignored   # 4 GiB streaming round trip
cargo clippy --workspace --all-targets -- -D warnings

scripts/fetch_corpus.sh                                  # download the Silesia/Canterbury corpora
gpucomp bench --corpus testdata/corpus/silesia           # benchmarks (add --record --report to log a run)
```

GPU tests are skipped when no adapter is available. CI runs on macOS and Windows.

## License

[MIT NON-AI License](LICENSE).
