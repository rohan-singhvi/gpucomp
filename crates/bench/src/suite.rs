//! Benchmark suites. Each returns measurements; the CLI wraps them in a [`Run`](crate::record::Run).

use std::time::{Duration, Instant};

use gpu::wgpu::BufferUsages;
use gpu::{Context, GpuTimer, XorKernel};

use crate::record::{Measurement, Timing};
use crate::stats::{gbps, median_of};

#[derive(Clone, Debug)]
pub struct SuiteConfig {
    /// Bytes processed per iteration.
    pub bytes: usize,
    pub warmup: usize,
    pub runs: usize,
}

/// M0 platform ceilings: the fastest any end-to-end GPU path could go on this
/// machine. CPU `memcpy`; host→GPU upload; GPU→host readback; a trivial
/// bandwidth-bound kernel (kernel-only and end-to-end).
pub fn platform(ctx: Option<&Context>, cfg: &SuiteConfig) -> anyhow::Result<Vec<Measurement>> {
    let n = cfg.bytes & !3; // whole u32 words for the XOR kernel
    let data = pseudo_random(n);
    let row = |name: &str, elapsed: Duration, timing| Measurement {
        name: name.into(),
        input: "random".into(),
        bytes: n as u64,
        gbps: gbps(n as u64, elapsed),
        ratio: None,
        timing,
    };
    let time = |f: &mut dyn FnMut() -> anyhow::Result<()>| {
        median_of(cfg.warmup, cfg.runs, || {
            let start = Instant::now();
            f()?;
            Ok(start.elapsed())
        })
    };

    let mut dst = vec![0u8; n];
    let memcpy = time(&mut || {
        dst.copy_from_slice(std::hint::black_box(&data));
        std::hint::black_box(&dst);
        Ok(())
    })?;
    let mut rows = vec![row("cpu.memcpy", memcpy, Timing::Cpu)];
    let Some(ctx) = ctx else { return Ok(rows) };

    let buffer = ctx.upload(
        &data,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    );
    let upload = time(&mut || {
        ctx.queue.write_buffer(&buffer, 0, &data);
        ctx.queue.submit([]);
        Ok(ctx.wait()?)
    })?;
    rows.push(row("gpu.upload", upload, Timing::WallE2e));

    let readback = time(&mut || Ok(ctx.read_buffer(&buffer, n as u64).map(drop)?))?;
    rows.push(row("gpu.readback", readback, Timing::WallE2e));

    let kernel = XorKernel::new(ctx);
    let (kernel_time, timing) = match GpuTimer::new(ctx) {
        Some(timer) => (
            median_of(cfg.warmup, cfg.runs, || {
                kernel.dispatch(ctx, &buffer, 0x5A5A_5A5A, Some(&timer));
                Ok(timer.read(ctx)?)
            })?,
            Timing::GpuTimestamp,
        ),
        None => (
            time(&mut || {
                kernel.dispatch(ctx, &buffer, 0x5A5A_5A5A, None);
                Ok(ctx.wait()?)
            })?,
            Timing::WallGpu,
        ),
    };
    rows.push(row("gpu.xor.kernel", kernel_time, timing));

    let words: Vec<u32> = bytemuck_words(&data);
    let e2e = time(&mut || Ok(ctx.xor_u32(&words, 0x5A5A_5A5A).map(drop)?))?;
    rows.push(row("gpu.xor.e2e", e2e, Timing::WallE2e));
    Ok(rows)
}

/// Synthetic benchmark inputs of `n` bytes each: `zeros`, `random`, `text`
/// (pseudo-English from a fixed vocabulary) and `mixed` (thirds of each).
pub fn synthetic_inputs(n: usize) -> Vec<(String, Vec<u8>)> {
    let third = n / 3;
    let mixed = [
        pseudo_text(third),
        pseudo_random(third),
        vec![0; n - 2 * third],
    ]
    .concat();
    vec![
        ("zeros".into(), vec![0; n]),
        ("random".into(), pseudo_random(n)),
        ("text".into(), pseudo_text(n)),
        ("mixed".into(), mixed),
    ]
}

/// Pseudo-English: words from a fixed 512-word vocabulary with a skewed
/// (mostly frequent words) distribution, plus punctuation and line breaks.
pub fn pseudo_text(n: usize) -> Vec<u8> {
    const SYLLABLES: [&str; 16] = [
        "ka", "lo", "mi", "ne", "ru", "ta", "se", "vi", "on", "el", "ar", "ist", "pre", "com",
        "tion", "ing",
    ];
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let vocabulary: Vec<String> = (0..512)
        .map(|_| {
            let r = next();
            (0..1 + r % 3)
                .map(|k| SYLLABLES[((r >> (8 + 4 * k)) & 15) as usize])
                .collect()
        })
        .collect();
    let mut out = Vec::with_capacity(n + 32);
    while out.len() < n {
        let r = next();
        let word = if r % 4 == 0 { r >> 8 } else { (r >> 8) % 48 } as usize % 512;
        out.extend_from_slice(vocabulary[word].as_bytes());
        out.extend_from_slice(match r % 23 {
            0 => b".\n",
            1 | 2 => b", ",
            _ => b" ",
        });
    }
    out.truncate(n);
    out
}

/// CPU codec baselines for each input: `lz4_flex` single- and multi-threaded,
/// the greedy (GPU-twin) encoder, and both decoders. Every row carries the
/// compression ratio of the file that path produced or read.
pub fn codecs(inputs: &[(String, Vec<u8>)], cfg: &SuiteConfig) -> anyhow::Result<Vec<Measurement>> {
    use cpu::container::{
        compress, decompress, CompressOptions, Decoder, DecompressOptions, Encoder,
    };

    let one_thread = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let flex = CompressOptions::default();
    let greedy = CompressOptions {
        encoder: Encoder::Greedy(Default::default()),
        ..flex
    };
    let decode = |decoder| DecompressOptions {
        decoder,
        verify: false,
    };

    let mut rows = Vec::new();
    for (name, data) in inputs {
        let n = data.len() as u64;
        let mut row = |label: &str, elapsed: Duration, file_len: usize| {
            rows.push(Measurement {
                name: label.into(),
                input: name.clone(),
                bytes: n,
                gbps: gbps(n, elapsed),
                ratio: Some(n as f64 / file_len as f64),
                timing: Timing::Cpu,
            });
        };
        let time = |f: &dyn Fn() -> anyhow::Result<()>| {
            median_of(cfg.warmup, cfg.runs, || {
                let start = Instant::now();
                f()?;
                Ok(start.elapsed())
            })
        };

        let flex_file = compress(data, &flex)?;
        let greedy_file = compress(data, &greedy)?;
        let t = time(&|| Ok(one_thread.install(|| compress(data, &flex)).map(drop)?))?;
        row("cpu.lz4_flex.compress.1t", t, flex_file.len());
        let t = time(&|| Ok(compress(data, &flex).map(drop)?))?;
        row("cpu.lz4_flex.compress.mt", t, flex_file.len());
        let t = time(&|| Ok(compress(data, &greedy).map(drop)?))?;
        row("cpu.greedy.compress.mt", t, greedy_file.len());

        // Check correctness once, outside the timed loops.
        for decoder in [Decoder::Lz4Flex, Decoder::HandWritten] {
            anyhow::ensure!(
                decompress(&flex_file, &decode(decoder))? == *data,
                "{decoder:?} output differs from the input"
            );
        }
        let t = time(&|| {
            Ok(one_thread
                .install(|| decompress(&flex_file, &decode(Decoder::Lz4Flex)))
                .map(drop)?)
        })?;
        row("cpu.lz4_flex.decompress.1t", t, flex_file.len());
        let t = time(&|| Ok(decompress(&flex_file, &decode(Decoder::Lz4Flex)).map(drop)?))?;
        row("cpu.lz4_flex.decompress.mt", t, flex_file.len());
        let t = time(&|| Ok(decompress(&flex_file, &decode(Decoder::HandWritten)).map(drop)?))?;
        row("cpu.handwritten.decompress.mt", t, flex_file.len());
    }
    Ok(rows)
}

/// Deterministic incompressible bytes (xorshift64).
pub fn pseudo_random(n: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut out = Vec::with_capacity(n + 8);
    while out.len() < n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.extend_from_slice(&state.to_le_bytes());
    }
    out.truncate(n);
    out
}

fn bytemuck_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pseudo_random_is_deterministic_and_exact_length() {
        assert_eq!(pseudo_random(1001).len(), 1001);
        assert_eq!(pseudo_random(64), pseudo_random(64));
        assert_ne!(pseudo_random(64), vec![0u8; 64]);
    }
}
