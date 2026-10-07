//! The platform-ceiling suite on real hardware (GPU parts skip without an adapter).

use bench::record::Timing;
use bench::suite::{platform, SuiteConfig};
use gpu::{Context, ContextOptions, GpuError, GpuTimer};

const SMALL: SuiteConfig = SuiteConfig {
    bytes: 1 << 20,
    warmup: 1,
    runs: 3,
};

fn context() -> Option<Context> {
    match Context::new(&ContextOptions::default()) {
        Ok(ctx) => Some(ctx),
        Err(GpuError::NoAdapter(e)) => {
            eprintln!("skipping GPU test: no adapter ({e})");
            None
        }
        Err(e) => panic!("GPU context creation failed: {e}"),
    }
}

#[test]
fn without_gpu_only_cpu_ceilings_are_measured() {
    let rows = platform(None, &SMALL).unwrap();
    let names: Vec<_> = rows.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, ["cpu.memcpy"]);
    assert_eq!(rows[0].timing, Timing::Cpu);
    assert_eq!(rows[0].bytes, SMALL.bytes as u64);
}

#[test]
fn gpu_ceilings_are_all_measured_with_positive_throughput() {
    let Some(ctx) = context() else { return };
    let rows = platform(Some(&ctx), &SMALL).unwrap();
    let names: Vec<_> = rows.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "cpu.memcpy",
            "gpu.upload",
            "gpu.readback",
            "gpu.xor.kernel",
            "gpu.xor.e2e"
        ]
    );
    for m in &rows {
        assert!(m.gbps.is_finite() && m.gbps > 0.0, "{m:?}");
    }
}

#[test]
fn kernel_timing_source_follows_timer_availability() {
    let Some(ctx) = context() else { return };
    let rows = platform(Some(&ctx), &SMALL).unwrap();
    let kernel = rows.iter().find(|m| m.name == "gpu.xor.kernel").unwrap();
    let expected = if GpuTimer::new(&ctx).is_some() {
        Timing::GpuTimestamp
    } else {
        Timing::WallGpu
    };
    assert_eq!(kernel.timing, expected);
}
