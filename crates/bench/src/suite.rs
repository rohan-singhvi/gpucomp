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
/// decode. Also GLZ-E (codec 3, `glze`): CPU both ways, GPU decode. 64 KiB
/// chunks.
pub fn glz(
    ctx: Option<&Context>,
    inputs: &[(String, Vec<u8>)],
    cfg: &SuiteConfig,
) -> anyhow::Result<Vec<Measurement>> {
    use cpu::container::{compress, decompress, CompressOptions, DecompressOptions, Encoder};
    use gpu::decode::GpuDecoder;
    use gpu::encode::{GpuCompressOptions, GpuEncoder};

    let gpu = match ctx {
        Some(ctx) => Some((
            ctx,
            GpuEncoder::new(ctx, Default::default())?,
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
        for (label, codec, groups) in [
            ("glz", format::Codec::Glz, None),
            ("glz-g64", format::Codec::Glz, Some(64)),
            ("glze", format::Codec::GlzE, None),
        ] {
            let cpu_options = CompressOptions {
                codec,
                encoder: Encoder::Glz(cpu::glz::GlzParams {
                    independent_groups: groups,
                    ..Default::default()
                }),
                ..CompressOptions::default()
            };
            let file = compress(data, &cpu_options)?;
            let mut row = |kind: &str, elapsed: Duration, timing| {
                rows.push(Measurement {
                    name: format!("{label}.{kind}"),
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

            let Some((ctx, encoder, decoder, timer)) = &gpu else {
                continue;
            };
            let gpu_options = GpuCompressOptions {
                codec,
                independent_groups: groups,
                ..GpuCompressOptions::default()
            };
            // The GPU GLZ-E encoder comes in M9e step 3.
            let gpu_encodes = codec == format::Codec::Glz;
            if gpu_encodes {
                anyhow::ensure!(
                    encoder.compress(ctx, data, &gpu_options)? == file,
                    "GPU GLZ file differs from the CPU twin"
                );
            }
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
            if gpu_encodes {
                let prepared = encoder.prepare(ctx, data, &gpu_options)?;
                let (t, timing) = kernel(&|timer| encoder.dispatch(ctx, &prepared, timer))?;
                drop(prepared);
                row("gpu.compress.kernel", t, timing);
                let t = time(&|| Ok(encoder.compress(ctx, data, &gpu_options).map(drop)?))?;
                row("gpu.compress.e2e", t, Timing::WallE2e);
            }
            if let Some(prepared) = decoder.prepare_file(ctx, &file)? {
                let (t, timing) = kernel(&|timer| decoder.dispatch(ctx, &prepared, timer))?;
                row("gpu.decompress.kernel", t, timing);
            }
            let t = time(&|| Ok(decoder.decompress(ctx, &file, false).map(drop)?))?;
            row("gpu.decompress.e2e", t, Timing::WallE2e);
        }
    }
    // Names read `<device>.<codec>.<direction>.<timing>`.
    for m in &mut rows {
        let (codec, rest) = m.name.split_once('.').unwrap();
        let (device, rest) = rest.split_once('.').unwrap();
        m.name = format!("{device}.{codec}.{rest}");
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

/// Numeric benchmark inputs of exactly `n` bytes each, the data filters are
/// for (M7): `f32-points` (xyz of a smooth curve), `sorted-u32` (small random
/// gaps), `i16-audio` (two sines plus noise) and `u64-timestamps` (≈1 ms
/// steps with jitter). Deterministic.
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
    vec![
        ("f32-points".into(), points),
        ("sorted-u32".into(), sorted),
        ("i16-audio".into(), audio),
        ("u64-timestamps".into(), stamps),
    ]
}

/// M7 filters: for each input and codec (`lz4` = the greedy twin, `glz`),
/// CPU multi-threaded compression with filters `none` and `auto` (level 1:
/// none, shuffle-4, delta-4), each row carrying that file's ratio. With a GPU,
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
    use format::{Codec, Filter, Index};
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
            for (bucket, wins) in [
                ("none", Filter::None),
                ("shuffle-4", Filter::Shuffle { width: 4 }),
                ("delta-4", Filter::Delta { width: 4 }),
            ]
            .map(|(bucket, f)| {
                let count = index
                    .chunks
                    .iter()
                    .filter(|c| !c.stored && c.filter == f)
                    .count();
                (bucket, count)
            })
            .into_iter()
            .chain([("stored", index.chunks.iter().filter(|c| c.stored).count())])
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
