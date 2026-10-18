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

/// GPU decompression of each input (compressed by `lz4_flex`, 64 KiB chunks)
/// with the naive (M2) and cooperative (M5, default) kernels: kernel-only
/// (timestamps, else submit + wait) and end to end (parse, upload, decode,
/// readback).
pub fn gpu_decode(
    ctx: &Context,
    inputs: &[(String, Vec<u8>)],
    cfg: &SuiteConfig,
) -> anyhow::Result<Vec<Measurement>> {
    use gpu::decode::{DecodeKernel, DecoderConfig};
    let naive = DecoderConfig {
        kernel: DecodeKernel::Naive,
        ..DecoderConfig::default()
    };
    gpu_decode_configs(
        ctx,
        inputs,
        cfg,
        &[("naive", naive), ("coop", DecoderConfig::default())],
    )
}

/// [`gpu_decode`] for arbitrary decoder configurations, labelled
/// `gpu.lz4.decompress.<label>.{kernel,e2e}`; rows are grouped by label.
pub fn gpu_decode_configs(
    ctx: &Context,
    inputs: &[(String, Vec<u8>)],
    cfg: &SuiteConfig,
    configs: &[(&str, gpu::decode::DecoderConfig)],
) -> anyhow::Result<Vec<Measurement>> {
    use cpu::container::{compress, CompressOptions};
    use gpu::decode::GpuDecoder;

    let timer = GpuTimer::new(ctx);
    let files = inputs
        .iter()
        .map(|(_, data)| compress(data, &CompressOptions::default()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut rows = Vec::new();
    for (label, config) in configs {
        let decoder = GpuDecoder::with_config(ctx, *config)?;
        for ((name, data), file) in inputs.iter().zip(&files) {
            let n = data.len() as u64;
            let ratio = Some(n as f64 / file.len() as f64);
            let row = |kind: &str, elapsed: Duration, timing| Measurement {
                name: format!("gpu.lz4.decompress.{label}.{kind}"),
                input: name.clone(),
                bytes: n,
                gbps: gbps(n, elapsed),
                ratio,
                timing,
            };
            anyhow::ensure!(
                decoder.decompress(ctx, file, false)? == *data,
                "{label} GPU decode of {name} differs from the input"
            );
            let Some(prepared) = decoder.prepare_file(ctx, file)? else {
                continue; // empty input
            };
            let (kernel, timing) = match &timer {
                Some(timer) => (
                    median_of(cfg.warmup, cfg.runs, || {
                        decoder.dispatch(ctx, &prepared, Some(timer));
                        Ok(timer.read(ctx)?)
                    })?,
                    Timing::GpuTimestamp,
                ),
                None => (
                    median_of(cfg.warmup, cfg.runs, || {
                        let start = Instant::now();
                        decoder.dispatch(ctx, &prepared, None);
                        ctx.wait()?;
                        Ok(start.elapsed())
                    })?,
                    Timing::WallGpu,
                ),
            };
            drop(prepared);
            let e2e = median_of(cfg.warmup, cfg.runs, || {
                let start = Instant::now();
                decoder.decompress(ctx, file, false)?;
                Ok(start.elapsed())
            })?;
            rows.push(row("kernel", kernel, timing));
            rows.push(row("e2e", e2e, Timing::WallE2e));
        }
    }
    Ok(rows)
}

/// GPU compression of each input (64 KiB chunks) at levels 1 to 3: kernel-only
/// and end to end (upload, encode, readback, packing, checksums off). Level 1
/// keeps the original row names (`gpu.lz4.compress.*`); levels 2 and 3 are
/// `gpu.lz4.l2.compress.*` and `gpu.lz4.l3.compress.*`.
pub fn gpu_encode(
    ctx: &Context,
    inputs: &[(String, Vec<u8>)],
    cfg: &SuiteConfig,
) -> anyhow::Result<Vec<Measurement>> {
    use gpu::encode::{EncodeParams, GpuCompressOptions, GpuEncoder};

    let timer = GpuTimer::new(ctx);
    let mut rows = Vec::new();
    for level in 1..=3u8 {
        let encoder = GpuEncoder::new(ctx, EncodeParams::for_level(level))?;
        let options = GpuCompressOptions {
            level,
            ..GpuCompressOptions::default()
        };
        let prefix = match level {
            1 => "gpu.lz4.compress".to_string(),
            l => format!("gpu.lz4.l{l}.compress"),
        };
        for (name, data) in inputs {
            if data.is_empty() {
                continue;
            }
            let file = encoder.compress(ctx, data, &options)?;
            let n = data.len() as u64;
            let ratio = Some(n as f64 / file.len() as f64);
            let row = |label: &str, elapsed: Duration, timing| Measurement {
                name: format!("{prefix}.{label}"),
                input: name.clone(),
                bytes: n,
                gbps: gbps(n, elapsed),
                ratio,
                timing,
            };

            let prepared = encoder.prepare(ctx, data, &options)?;
            let (kernel, timing) = match &timer {
                Some(timer) => (
                    median_of(cfg.warmup, cfg.runs, || {
                        encoder.dispatch(ctx, &prepared, Some(timer));
                        Ok(timer.read(ctx)?)
                    })?,
                    Timing::GpuTimestamp,
                ),
                None => (
                    median_of(cfg.warmup, cfg.runs, || {
                        let start = Instant::now();
                        encoder.dispatch(ctx, &prepared, None);
                        ctx.wait()?;
                        Ok(start.elapsed())
                    })?,
                    Timing::WallGpu,
                ),
            };
            drop(prepared);
            rows.push(row("kernel", kernel, timing));

            let e2e = median_of(cfg.warmup, cfg.runs, || {
                let start = Instant::now();
                encoder.compress(ctx, data, &options)?;
                Ok(start.elapsed())
            })?;
            rows.push(row("e2e", e2e, Timing::WallE2e));
        }
    }
    Ok(rows)
}

/// GLZ (codec 2) in both directions, plain and with dependency elimination
/// over groups of 64 sequences (`glz-g64`): CPU encoder and reference decoder
/// (multi-threaded), and with a GPU, kernel-only and end-to-end encode and
/// decode. Also GLZ-E (codec 3, `glze`), both ways, at levels 1 to 3 (`glze`,
/// `glze.l2`, `glze.l3`). Names read `<device>.<codec>.<direction>.<timing>`.
/// 64 KiB chunks.
pub fn glz(
    ctx: Option<&Context>,
    inputs: &[(String, Vec<u8>)],
    cfg: &SuiteConfig,
) -> anyhow::Result<Vec<Measurement>> {
    use cpu::container::{compress, decompress, CompressOptions, DecompressOptions, Encoder};
    use gpu::decode::GpuDecoder;
    use gpu::encode::{EncodeParams, GpuCompressOptions, GpuEncoder};

    let gpu = match ctx {
        Some(ctx) => Some((
            ctx,
            (1..=3)
                .map(|level| GpuEncoder::new(ctx, EncodeParams::for_level(level)))
                .collect::<Result<Vec<_>, _>>()?,
            GpuDecoder::new(ctx),
            GpuTimer::new(ctx),
        )),
        None => None,
    };
    let time = |f: &dyn Fn() -> anyhow::Result<()>| {
        median_of(cfg.warmup, cfg.runs, || {
            let start = Instant::now();
            f()?;
            Ok(start.elapsed())
        })
    };
    let mut rows = Vec::new();
    for (name, data) in inputs {
        if data.is_empty() {
            continue;
        }
        let n = data.len() as u64;
        for (label, codec, groups, level) in [
            ("glz", format::Codec::Glz, None, 1),
            ("glz-g64", format::Codec::Glz, Some(64), 1),
            ("glze", format::Codec::GlzE, None, 1),
            ("glze.l2", format::Codec::GlzE, None, 2),
            ("glze.l3", format::Codec::GlzE, None, 3),
        ] {
            let cpu_options = CompressOptions {
                codec,
                encoder: Encoder::Glz(cpu::glz::GlzParams {
                    lz: cpu::lz4::encode::Params::for_level(level),
                    independent_groups: groups,
                }),
                level,
                ..CompressOptions::default()
            };
            let file = compress(data, &cpu_options)?;
            let mut row = |kind: &str, elapsed: Duration, timing| {
                let (device, rest) = kind.split_once('.').unwrap();
                rows.push(Measurement {
                    name: format!("{device}.{label}.{rest}"),
                    input: name.clone(),
                    bytes: n,
                    gbps: gbps(n, elapsed),
                    ratio: Some(n as f64 / file.len() as f64),
                    timing,
                });
            };
            let reference = DecompressOptions {
                verify: false,
                ..DecompressOptions::default()
            };
            anyhow::ensure!(
                decompress(&file, &reference)? == *data,
                "CPU GLZ round trip"
            );
            let t = time(&|| Ok(compress(data, &cpu_options).map(drop)?))?;
            row("cpu.compress.mt", t, Timing::Cpu);
            if groups.is_none() {
                let t = time(&|| Ok(decompress(&file, &reference).map(drop)?))?;
                row("cpu.decompress.mt", t, Timing::Cpu);
            }

            let Some((ctx, encoders, decoder, timer)) = &gpu else {
                continue;
            };
            let encoder = &encoders[usize::from(level) - 1];
            let gpu_options = GpuCompressOptions {
                codec,
                independent_groups: groups,
                level,
                ..GpuCompressOptions::default()
            };
            anyhow::ensure!(
                encoder.compress(ctx, data, &gpu_options)? == file,
                "GPU {label} file differs from the CPU twin"
            );
            anyhow::ensure!(
                decoder.decompress(ctx, &file, false)? == *data,
                "GPU GLZ decode"
            );
            let kernel =
                |dispatch: &dyn Fn(Option<&GpuTimer>)| -> anyhow::Result<(Duration, Timing)> {
                    Ok(match timer {
                        Some(timer) => (
                            median_of(cfg.warmup, cfg.runs, || {
                                dispatch(Some(timer));
                                Ok(timer.read(ctx)?)
                            })?,
                            Timing::GpuTimestamp,
                        ),
                        None => (
                            median_of(cfg.warmup, cfg.runs, || {
                                let start = Instant::now();
                                dispatch(None);
                                ctx.wait()?;
                                Ok(start.elapsed())
                            })?,
                            Timing::WallGpu,
                        ),
                    })
                };
            let prepared = encoder.prepare(ctx, data, &gpu_options)?;
            let (t, timing) = kernel(&|timer| encoder.dispatch(ctx, &prepared, timer))?;
            drop(prepared);
            row("gpu.compress.kernel", t, timing);
            let t = time(&|| Ok(encoder.compress(ctx, data, &gpu_options).map(drop)?))?;
            row("gpu.compress.e2e", t, Timing::WallE2e);
            if let Some(prepared) = decoder.prepare_file(ctx, &file)? {
                let (t, timing) = kernel(&|timer| decoder.dispatch(ctx, &prepared, timer))?;
                row("gpu.decompress.kernel", t, timing);
            }
            let t = time(&|| Ok(decoder.decompress(ctx, &file, false).map(drop)?))?;
            row("gpu.decompress.e2e", t, Timing::WallE2e);
        }
    }
    Ok(rows)
}

/// Per-stage GPU profile of each input. For each encoder (LZ4 and GLZ-E at
/// levels 1 and 3) and decoder (cooperative LZ4, GLZ, GLZ-E), the median time
/// of every prefix of its kernels, as wall clock around submit + wait with the
/// data already on the GPU (prefixes can't be timed with timestamps). Rows are
/// `gpu.<codec>[.lN].<direction>.upto.<stage>`, with the throughput of the
/// prefix through that stage and the file's ratio; the report subtracts
/// consecutive prefixes to get each stage's own time.
pub fn profile(
    ctx: &Context,
    inputs: &[(String, Vec<u8>)],
    cfg: &SuiteConfig,
) -> anyhow::Result<Vec<Measurement>> {
    use cpu::container::{compress, CompressOptions, Encoder};
    use format::Codec;
    use gpu::decode::GpuDecoder;
    use gpu::encode::{EncodeParams, GpuCompressOptions, GpuEncoder};

    let time = |dispatch: &dyn Fn()| {
        median_of(cfg.warmup, cfg.runs, || {
            let start = Instant::now();
            dispatch();
            ctx.wait()?;
            Ok(start.elapsed())
        })
    };
    let glz = |codec| CompressOptions {
        codec,
        encoder: Encoder::Glz(Default::default()),
        ..CompressOptions::default()
    };
    let decoder = GpuDecoder::new(ctx);
    let mut rows = Vec::new();
    for (name, data) in inputs {
        if data.is_empty() {
            continue;
        }
        let n = data.len() as u64;
        let mut row = |label: String, elapsed: Duration, compressed: usize| {
            rows.push(Measurement {
                name: label,
                input: name.clone(),
                bytes: n,
                gbps: gbps(n, elapsed),
                ratio: Some(n as f64 / compressed as f64),
                timing: Timing::WallGpu,
            });
        };
        for (label, codec, level) in [
            ("lz4", Codec::Lz4, 1),
            ("lz4.l3", Codec::Lz4, 3),
            ("glze", Codec::GlzE, 1),
            ("glze.l3", Codec::GlzE, 3),
        ] {
            let encoder = GpuEncoder::new(ctx, EncodeParams::for_level(level))?;
            let options = GpuCompressOptions {
                codec,
                level,
                ..GpuCompressOptions::default()
            };
            let size = encoder.compress(ctx, data, &options)?.len();
            let prepared = encoder.prepare(ctx, data, &options)?;
            for (k, stage) in encoder.stages(&prepared).into_iter().enumerate() {
                let t = time(&|| encoder.dispatch_stages(ctx, &prepared, k + 1))?;
                row(format!("gpu.{label}.compress.upto.{stage}"), t, size);
            }
        }
        for (label, options) in [
            ("lz4", CompressOptions::default()),
            ("glz", glz(Codec::Glz)),
            ("glze", glz(Codec::GlzE)),
        ] {
            let file = compress(data, &options)?;
            let Some(prepared) = decoder.prepare_file(ctx, &file)? else {
                continue;
            };
            for (k, stage) in decoder.stages(&prepared).into_iter().enumerate() {
                let t = time(&|| decoder.dispatch_stages(ctx, &prepared, k + 1))?;
                row(
                    format!("gpu.{label}.decompress.upto.{stage}"),
                    t,
                    file.len(),
                );
            }
        }
    }
    Ok(rows)
}

/// The GPU's share of the chunks in the fixed-split LZ4 row (tuned for an
/// Apple M4 Pro, where it makes both sides finish together).
pub const HYBRID_LZ4_FLEX_SHARE: f64 = 0.25;

/// CPU and GPU together (`gpu::hybrid`), end to end, for LZ4 and GLZ-E at
/// level 1 and GLZ-E at level 3: compression (the file must equal the GPU
/// encoder's) and decompression of that file (all rayon threads plus the GPU,
/// which joins only when it helps). Rows `hybrid.<codec>[.lN].<direction>.e2e`.
/// Last, `hybrid.lz4_flex.compress.e2e`: LZ4 on a fixed split, the GPU taking
/// `HYBRID_LZ4_FLEX_SHARE` of the chunks and lz4_flex the rest.
pub fn hybrid(
    ctx: &Context,
    inputs: &[(String, Vec<u8>)],
    cfg: &SuiteConfig,
) -> anyhow::Result<Vec<Measurement>> {
    use format::Codec;
    use gpu::decode::GpuDecoder;
    use gpu::encode::{EncodeParams, GpuCompressOptions, GpuEncoder};

    let decoder = GpuDecoder::new(ctx);
    let time = |f: &dyn Fn() -> anyhow::Result<()>| {
        median_of(cfg.warmup, cfg.runs, || {
            let start = Instant::now();
            f()?;
            Ok(start.elapsed())
        })
    };
    let mut rows = Vec::new();
    for (label, codec, level) in [
        ("lz4", Codec::Lz4, 1),
        ("glze", Codec::GlzE, 1),
        ("glze.l3", Codec::GlzE, 3),
    ] {
        let encoder = GpuEncoder::new(ctx, EncodeParams::for_level(level))?;
        let options = GpuCompressOptions {
            codec,
            level,
            ..GpuCompressOptions::default()
        };
        for (name, data) in inputs {
            if data.is_empty() {
                continue;
            }
            let n = data.len() as u64;
            let file = gpu::hybrid::compress(ctx, &encoder, data, &options)?;
            anyhow::ensure!(
                file == encoder.compress(ctx, data, &options)?,
                "hybrid {label} file of {name} differs from the GPU's"
            );
            anyhow::ensure!(
                gpu::hybrid::decompress(ctx, &decoder, &file, false)? == *data,
                "hybrid {label} decode of {name}"
            );
            let ratio = Some(n as f64 / file.len() as f64);
            let mut row = |direction: &str, elapsed: Duration| {
                rows.push(Measurement {
                    name: format!("hybrid.{label}.{direction}.e2e"),
                    input: name.clone(),
                    bytes: n,
                    gbps: gbps(n, elapsed),
                    ratio,
                    timing: Timing::WallE2e,
                });
            };
            let t = time(&|| Ok(gpu::hybrid::compress(ctx, &encoder, data, &options).map(drop)?))?;
            row("compress", t);
            let t = time(&|| Ok(gpu::hybrid::decompress(ctx, &decoder, &file, false).map(drop)?))?;
            row("decompress", t);
        }
    }
    let encoder = GpuEncoder::new(ctx, EncodeParams::default())?;
    let options = GpuCompressOptions::default();
    let flex = cpu::container::CompressOptions::default();
    for (name, data) in inputs {
        if data.is_empty() {
            continue;
        }
        let n = data.len() as u64;
        let compress = || {
            gpu::hybrid::compress_static(
                ctx,
                &encoder,
                data,
                &options,
                HYBRID_LZ4_FLEX_SHARE,
                &flex,
            )
        };
        let file = compress()?;
        anyhow::ensure!(
            cpu::container::decompress(&file, &Default::default())? == *data,
            "fixed-split LZ4 file of {name}"
        );
        let t = time(&|| Ok(compress().map(drop)?))?;
        rows.push(Measurement {
            name: "hybrid.lz4_flex.compress.e2e".into(),
            input: name.clone(),
            bytes: n,
            gbps: gbps(n, t),
            ratio: Some(n as f64 / file.len() as f64),
            timing: Timing::WallE2e,
        });
    }
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

/// zstd baselines at levels 1 and 3, every row with that file's ratio.
/// - `cpu.zstd-N`: independent 64 KiB chunks, one frame each, across rayon's
///   threads in both directions. This is the random access a `.gpcz` file
///   gives; the ratio counts the frames' bytes and no index.
/// - `cpu.zstd-N-file`: the whole input as one frame, compressed by zstd's own
///   workers (one per rayon thread, like `zstd -T0`) and decompressed on one
///   thread, since a single frame decodes serially.
pub fn zstd(inputs: &[(String, Vec<u8>)], cfg: &SuiteConfig) -> anyhow::Result<Vec<Measurement>> {
    use rayon::prelude::*;
    use zstd::bulk::{Compressor, Decompressor};

    let chunk = format::DEFAULT_CHUNK_SIZE as usize;
    let workers = u32::try_from(rayon::current_num_threads())?;
    let failed = |e: &mut std::io::Error| anyhow::anyhow!("zstd context: {e}");
    let compress_chunks = |data: &[u8], level: i32| -> anyhow::Result<Vec<Vec<u8>>> {
        data.par_chunks(chunk)
            .map_init(
                || Compressor::new(level),
                |c, piece| Ok(c.as_mut().map_err(failed)?.compress(piece)?),
            )
            .collect()
    };
    let decompress_chunks = |frames: &[Vec<u8>], n: usize| -> anyhow::Result<Vec<u8>> {
        let mut out = vec![0u8; n];
        out.par_chunks_mut(chunk).zip(frames).try_for_each_init(
            Decompressor::new,
            |d, (dst, frame)| -> anyhow::Result<()> {
                let len = dst.len();
                let written = d
                    .as_mut()
                    .map_err(failed)?
                    .decompress_to_buffer(frame, dst)?;
                anyhow::ensure!(
                    written == len,
                    "zstd chunk decoded to {written} bytes, not {len}"
                );
                Ok(())
            },
        )?;
        Ok(out)
    };
    let compress_file = |data: &[u8], level: i32| -> anyhow::Result<Vec<u8>> {
        let mut c = Compressor::new(level)?;
        c.multithread(workers)?;
        Ok(c.compress(data)?)
    };
    let time = |f: &dyn Fn() -> anyhow::Result<()>| {
        median_of(cfg.warmup, cfg.runs, || {
            let start = Instant::now();
            f()?;
            Ok(start.elapsed())
        })
    };

    let mut rows = Vec::new();
    for (name, data) in inputs {
        if data.is_empty() {
            continue;
        }
        let n = data.len() as u64;
        for level in [1, 3] {
            let mut row = |label: String, elapsed: Duration, compressed: usize| {
                rows.push(Measurement {
                    name: label,
                    input: name.clone(),
                    bytes: n,
                    gbps: gbps(n, elapsed),
                    ratio: Some(n as f64 / compressed as f64),
                    timing: Timing::Cpu,
                });
            };
            let frames = compress_chunks(data, level)?;
            let size = frames.iter().map(Vec::len).sum();
            anyhow::ensure!(
                decompress_chunks(&frames, data.len())? == *data,
                "zstd -{level} chunk round trip of {name}"
            );
            let t = time(&|| compress_chunks(data, level).map(drop))?;
            row(format!("cpu.zstd-{level}.compress.mt"), t, size);
            let t = time(&|| decompress_chunks(&frames, data.len()).map(drop))?;
            row(format!("cpu.zstd-{level}.decompress.mt"), t, size);

            let file = compress_file(data, level)?;
            let decompress_file = || Decompressor::new()?.decompress(&file, data.len());
            anyhow::ensure!(
                decompress_file()? == *data,
                "zstd -{level} whole-file round trip of {name}"
            );
            let t = time(&|| compress_file(data, level).map(drop))?;
            row(format!("cpu.zstd-{level}-file.compress.mt"), t, file.len());
            let t = time(&|| Ok(decompress_file().map(drop)?))?;
            row(
                format!("cpu.zstd-{level}-file.decompress.1t"),
                t,
                file.len(),
            );
        }
    }
    Ok(rows)
}

/// File-to-file wall time, as the CLI runs it, page cache warm, on files in
/// `dir` (created, then removed); outputs are flushed to disk between runs,
/// outside the timing:
/// - `file.cpu.lz4_flex.compress` / `file.cpu.lz4.decompress`: gpucomp's
///   streamed CPU paths (lz4_flex, then the hand-written decoder);
/// - `file.zstd-1.compress.mt` / `file.zstd-1.decompress.1t`: zstd's own
///   streams, like `zstd -1 -T0` and `zstd -d`;
/// - with `gpu`, `file.gpu.lz4.compress` / `file.gpu.lz4.decompress`: the
///   GPU streams, each run creating its context and coder as one CLI
///   invocation does; and `file.hybrid.lz4_flex.compress`, the streamed
///   lz4_flex + GPU split (`--gpu-share`), likewise.
pub fn files(
    gpu: Option<&gpu::ContextOptions>,
    inputs: &[(String, Vec<u8>)],
    cfg: &SuiteConfig,
    dir: &std::path::Path,
) -> anyhow::Result<Vec<Measurement>> {
    std::fs::create_dir_all(dir)?;
    let rows = file_rows(gpu, inputs, cfg, dir);
    std::fs::remove_dir_all(dir)?;
    rows
}

fn file_rows(
    gpu: Option<&gpu::ContextOptions>,
    inputs: &[(String, Vec<u8>)],
    cfg: &SuiteConfig,
    dir: &std::path::Path,
) -> anyhow::Result<Vec<Measurement>> {
    use std::fs::File;
    use std::io::{BufReader, BufWriter, Read, Write};

    use cpu::container::{compress_stream, decompress_stream, CompressOptions, DecompressOptions};

    /// `std::io::copy` with 1 MiB reads, so zstd isn't fed 8 KiB at a time.
    fn copy(reader: &mut impl Read, writer: &mut impl Write) -> std::io::Result<()> {
        let mut buffer = vec![0u8; 1 << 20];
        loop {
            match reader.read(&mut buffer)? {
                0 => return Ok(()),
                n => writer.write_all(&buffer[..n])?,
            }
        }
    }
    let workers = u32::try_from(rayon::current_num_threads())?;
    let (src, out) = (dir.join("input"), dir.join("output"));
    let (packed, zst, gpu_packed) = (
        dir.join("cpu.gpcz"),
        dir.join("input.zst"),
        dir.join("gpu.gpcz"),
    );
    // Repeated runs that each write hundreds of MB pile up dirty pages, and
    // the OS then throttles writes (Silesia decompressed to a file: best
    // 21 ms, median 51). Flushing the outputs after each run, outside the
    // timing, starts every run as a single command would.
    let flush = || -> anyhow::Result<()> {
        for path in [&out, &packed, &zst, &gpu_packed] {
            if let Ok(file) = File::open(path) {
                file.sync_all()?;
            }
        }
        Ok(())
    };
    let time = |f: &dyn Fn() -> anyhow::Result<()>| {
        median_of(cfg.warmup, cfg.runs, || {
            let start = Instant::now();
            f()?;
            let elapsed = start.elapsed();
            flush()?;
            Ok(elapsed)
        })
    };
    let mut rows = Vec::new();
    for (name, data) in inputs {
        if data.is_empty() {
            continue;
        }
        let n = data.len() as u64;
        std::fs::write(&src, data)?;
        let mut row = |label: &str, elapsed: Duration, file: &std::path::Path, timing| {
            let compressed = std::fs::metadata(file)?.len();
            rows.push(Measurement {
                name: label.into(),
                input: name.clone(),
                bytes: n,
                gbps: gbps(n, elapsed),
                ratio: Some(n as f64 / compressed as f64),
                timing,
            });
            anyhow::Ok(())
        };
        let check = |what: &str| -> anyhow::Result<()> {
            anyhow::ensure!(
                std::fs::read(&out)? == *data,
                "{what} file round trip of {name}"
            );
            Ok(())
        };

        let compress = || -> anyhow::Result<()> {
            let mut input = File::open(&src)?;
            let mut output = File::create(&packed)?;
            compress_stream(&mut input, n, &mut output, &CompressOptions::default())?;
            Ok(())
        };
        let decompress = || -> anyhow::Result<()> {
            let options = DecompressOptions {
                verify: false,
                ..DecompressOptions::default()
            };
            let mut output = File::create(&out)?;
            decompress_stream(&mut File::open(&packed)?, &mut output, &options)?;
            Ok(())
        };
        compress()?;
        decompress()?;
        check("gpucomp")?;
        row(
            "file.cpu.lz4_flex.compress",
            time(&compress)?,
            &packed,
            Timing::Cpu,
        )?;
        row(
            "file.cpu.lz4.decompress",
            time(&decompress)?,
            &packed,
            Timing::Cpu,
        )?;

        let compress = || -> anyhow::Result<()> {
            let mut encoder = zstd::stream::Encoder::new(File::create(&zst)?, 1)?;
            encoder.multithread(workers)?;
            copy(&mut File::open(&src)?, &mut encoder)?;
            encoder.finish()?;
            Ok(())
        };
        let decompress = || -> anyhow::Result<()> {
            let mut decoder = zstd::stream::Decoder::new(File::open(&zst)?)?;
            copy(&mut decoder, &mut File::create(&out)?)?;
            Ok(())
        };
        compress()?;
        decompress()?;
        check("zstd")?;
        row(
            "file.zstd-1.compress.mt",
            time(&compress)?,
            &zst,
            Timing::Cpu,
        )?;
        row(
            "file.zstd-1.decompress.1t",
            time(&decompress)?,
            &zst,
            Timing::Cpu,
        )?;

        let Some(context) = gpu else { continue };
        let compress = || -> anyhow::Result<()> {
            let ctx = gpu::Context::new(context)?;
            let encoder = gpu::encode::GpuEncoder::new(&ctx, gpu::encode::EncodeParams::default())?;
            let mut input = BufReader::new(File::open(&src)?);
            let mut output = BufWriter::new(File::create(&gpu_packed)?);
            let options = gpu::encode::GpuCompressOptions::default();
            encoder.compress_stream(&ctx, &mut input, n, &mut output, &options)?;
            output.flush()?;
            Ok(())
        };
        let decompress = || -> anyhow::Result<()> {
            let ctx = gpu::Context::new(context)?;
            let decoder = gpu::decode::GpuDecoder::new(&ctx);
            let mut input = BufReader::new(File::open(&gpu_packed)?);
            let mut output = BufWriter::new(File::create(&out)?);
            decoder.decompress_stream(&ctx, &mut input, &mut output, false)?;
            output.flush()?;
            Ok(())
        };
        compress()?;
        decompress()?;
        check("GPU")?;
        row(
            "file.gpu.lz4.compress",
            time(&compress)?,
            &gpu_packed,
            Timing::WallE2e,
        )?;
        row(
            "file.gpu.lz4.decompress",
            time(&decompress)?,
            &gpu_packed,
            Timing::WallE2e,
        )?;

        let compress = || -> anyhow::Result<()> {
            let options = gpu::encode::GpuCompressOptions::default();
            let cpu = cpu::container::CompressOptions::default();
            let gpu = || {
                let ctx = gpu::Context::new(context)?;
                let encoder =
                    gpu::encode::GpuEncoder::new(&ctx, gpu::encode::EncodeParams::default())?;
                Ok((ctx, encoder))
            };
            let mut head = File::open(&src)?;
            let input = gpu::hybrid::SplitInput {
                head: &mut head,
                tail: File::open(&src)?,
                len: n,
            };
            let mut output = File::create(&gpu_packed)?;
            let share = HYBRID_LZ4_FLEX_SHARE;
            gpu::hybrid::compress_static_stream(input, &mut output, &options, share, &cpu, gpu)?;
            Ok(())
        };
        compress()?;
        decompress()?;
        check("lz4_flex + GPU")?;
        row(
            "file.hybrid.lz4_flex.compress",
            time(&compress)?,
            &gpu_packed,
            Timing::WallE2e,
        )?;
    }
    Ok(rows)
}

/// Numeric benchmark inputs of exactly `n` bytes each, the data filters are
/// for (M7): `f32-points` (xyz of a smooth curve), `sorted-u32` (small random
/// gaps), `i16-audio` (two sines plus noise), `u64-timestamps` (≈1 ms
/// steps with jitter) and `f64-walk` (a random walk at full precision).
/// Deterministic.
pub fn numeric_inputs(n: usize) -> Vec<(String, Vec<u8>)> {
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let fill = |mut elem: Box<dyn FnMut(u64) -> Vec<u8> + '_>| {
        let mut out = Vec::with_capacity(n + 16);
        let mut i = 0u64;
        while out.len() < n {
            out.extend(elem(i));
            i += 1;
        }
        out.truncate(n);
        out
    };
    let points = fill(Box::new(|i| {
        let t = i as f32 * 1e-4;
        [
            (t * 3.0).sin() * 250.0,
            (t * 2.0).cos() * 250.0,
            t * 10.0 + (t * 7.0).sin(),
        ]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect()
    }));
    let mut value = 1_000u32;
    let sorted = fill(Box::new(|_| {
        value = value.wrapping_add((next() % 64) as u32);
        value.to_le_bytes().to_vec()
    }));
    let audio = fill(Box::new(|i| {
        let t = i as f32 / 44_100.0;
        let x = (t * 440.0 * std::f32::consts::TAU).sin() * 9000.0
            + (t * 97.0 * std::f32::consts::TAU).sin() * 3000.0
            + (next() % 256) as f32
            - 128.0;
        (x as i16).to_le_bytes().to_vec()
    }));
    let mut stamp = 1_700_000_000_000_000u64;
    let stamps = fill(Box::new(|_| {
        stamp += 1_000 + next() % 16;
        stamp.to_le_bytes().to_vec()
    }));
    // A random walk in f64 at full precision (prices, simulation state):
    // steps of about 0.1%, roughly normal (a sum of four uniforms).
    let mut price = 100.0f64;
    let walk = fill(Box::new(|_| {
        let step = (0..4).map(|_| (next() >> 11) as f64).sum::<f64>() / (1u64 << 53) as f64;
        price *= 1.0 + (step - 2.0) * 1e-3;
        price.to_le_bytes().to_vec()
    }));
    vec![
        ("f32-points".into(), points),
        ("sorted-u32".into(), sorted),
        ("i16-audio".into(), audio),
        ("u64-timestamps".into(), stamps),
        ("f64-walk".into(), walk),
    ]
}

/// M7 filters: for each input and codec (`lz4` = the greedy twin, `glz`, `glze`),
/// CPU multi-threaded compression with filters `none` and `auto` (level 1:
/// none, shuffle-4, delta-4, delta-shuffle-4), each row carrying that file's
/// ratio. With a GPU,
/// decode of both files (kernel; `auto` also end to end), so the kernels'
/// difference shows the inverse-filter stage's cost. Last, rows
/// `cpu.<codec>.auto.wins.<filter>` whose value (in the GB/s column) is the
/// *share of chunks* that filter won, with `stored` for chunks no candidate
/// shrank. 64 KiB chunks.
pub fn filters(
    ctx: Option<&Context>,
    inputs: &[(String, Vec<u8>)],
    cfg: &SuiteConfig,
) -> anyhow::Result<Vec<Measurement>> {
    use cpu::container::{compress, decompress, CompressOptions, Encoder, FilterMode};
    use format::{Codec, Index};
    use gpu::decode::GpuDecoder;

    let gpu = match ctx {
        Some(ctx) => Some((
            ctx,
            GpuDecoder::new(ctx),
            GpuTimer::new(ctx),
            gpu::encode::GpuEncoder::new(ctx, Default::default())?,
        )),
        None => None,
    };
    let time = |f: &dyn Fn() -> anyhow::Result<()>| {
        median_of(cfg.warmup, cfg.runs, || {
            let start = Instant::now();
            f()?;
            Ok(start.elapsed())
        })
    };
    let mut rows = Vec::new();
    for (name, data) in inputs {
        if data.is_empty() {
            continue;
        }
        let n = data.len() as u64;
        for (codec_name, codec, encoder) in [
            ("lz4", Codec::Lz4, Encoder::Greedy(Default::default())),
            ("glz", Codec::Glz, Encoder::Glz(Default::default())),
            ("glze", Codec::GlzE, Encoder::Glz(Default::default())),
        ] {
            let options = |filters| CompressOptions {
                codec,
                encoder,
                filters,
                ..CompressOptions::default()
            };
            let mut row = |label: String, value: f64, ratio: Option<f64>, timing| {
                rows.push(Measurement {
                    name: label,
                    input: name.clone(),
                    bytes: n,
                    gbps: value,
                    ratio,
                    timing,
                });
            };
            let mut files = Vec::new();
            for (mode_name, mode) in [("none", FilterMode::None), ("auto", FilterMode::Auto)] {
                let file = compress(data, &options(mode))?;
                anyhow::ensure!(
                    decompress(&file, &Default::default())? == *data,
                    "CPU {codec_name} {mode_name} round trip of {name}"
                );
                let t = time(&|| Ok(compress(data, &options(mode)).map(drop)?))?;
                let ratio = Some(n as f64 / file.len() as f64);
                row(
                    format!("cpu.{codec_name}.compress.{mode_name}.mt"),
                    gbps(n, t),
                    ratio,
                    Timing::Cpu,
                );
                files.push((mode_name, file));
            }

            if let Some((ctx, decoder, timer, gpu_encoder)) = &gpu {
                for (mode_name, file) in &files {
                    anyhow::ensure!(
                        decoder.decompress(ctx, file, false)? == *data,
                        "GPU {codec_name} {mode_name} decode of {name}"
                    );
                    let ratio = Some(n as f64 / file.len() as f64);
                    if let Some(prepared) = decoder.prepare_file(ctx, file)? {
                        let (t, timing) = match timer {
                            Some(timer) => (
                                median_of(cfg.warmup, cfg.runs, || {
                                    decoder.dispatch(ctx, &prepared, Some(timer));
                                    Ok(timer.read(ctx)?)
                                })?,
                                Timing::GpuTimestamp,
                            ),
                            None => (
                                time(&|| {
                                    decoder.dispatch(ctx, &prepared, None);
                                    Ok(ctx.wait()?)
                                })?,
                                Timing::WallGpu,
                            ),
                        };
                        row(
                            format!("gpu.{codec_name}.decompress.{mode_name}.kernel"),
                            gbps(n, t),
                            ratio,
                            timing,
                        );
                    }
                    if *mode_name == "auto" {
                        let t = time(&|| Ok(decoder.decompress(ctx, file, false).map(drop)?))?;
                        row(
                            format!("gpu.{codec_name}.decompress.auto.e2e"),
                            gbps(n, t),
                            ratio,
                            Timing::WallE2e,
                        );
                        // GPU filter selection; its file must equal the CPU's.
                        let gpu_options = gpu::encode::GpuCompressOptions {
                            codec,
                            filters: gpu::encode::FilterMode::Auto,
                            ..Default::default()
                        };
                        anyhow::ensure!(
                            gpu_encoder.compress(ctx, data, &gpu_options)? == *file,
                            "GPU {codec_name} auto file of {name} differs from the CPU's"
                        );
                        let t = time(&|| {
                            Ok(gpu_encoder.compress(ctx, data, &gpu_options).map(drop)?)
                        })?;
                        row(
                            format!("gpu.{codec_name}.compress.auto.e2e"),
                            gbps(n, t),
                            ratio,
                            Timing::WallE2e,
                        );
                    }
                }
            }

            let index = Index::parse(&files[1].1)?;
            let total = index.chunks.len() as f64;
            for (bucket, wins) in cpu::filter::candidates(1)
                .iter()
                .map(|&f| {
                    let count = index
                        .chunks
                        .iter()
                        .filter(|c| !c.stored && c.filter == f)
                        .count();
                    (f.to_string(), count)
                })
                .chain([(
                    "stored".to_string(),
                    index.chunks.iter().filter(|c| c.stored).count(),
                )])
            {
                row(
                    format!("cpu.{codec_name}.auto.wins.{bucket}"),
                    wins as f64 / total,
                    None,
                    Timing::Cpu,
                );
            }

            // Exhaustive selection (every candidate on whole chunks), for
            // comparison with Auto's sampled selection.
            let exhaustive = options(FilterMode::Exhaustive);
            let file = compress(data, &exhaustive)?;
            let ratio = Some(n as f64 / file.len() as f64);
            let t = time(&|| Ok(compress(data, &exhaustive).map(drop)?))?;
            row(
                format!("cpu.{codec_name}.compress.exhaustive.mt"),
                gbps(n, t),
                ratio,
                Timing::Cpu,
            );
            if let Some((ctx, _, _, gpu_encoder)) = &gpu {
                let gpu_options = gpu::encode::GpuCompressOptions {
                    codec,
                    filters: gpu::encode::FilterMode::Exhaustive,
                    ..Default::default()
                };
                anyhow::ensure!(
                    gpu_encoder.compress(ctx, data, &gpu_options)? == file,
                    "GPU {codec_name} exhaustive file of {name} differs from the CPU's"
                );
                let t = time(&|| Ok(gpu_encoder.compress(ctx, data, &gpu_options).map(drop)?))?;
                row(
                    format!("gpu.{codec_name}.compress.exhaustive.e2e"),
                    gbps(n, t),
                    ratio,
                    Timing::WallE2e,
                );
            }
        }
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
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|w| u32::from_le_bytes(*w)).collect()
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
