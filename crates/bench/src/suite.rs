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
