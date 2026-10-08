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

#[test]
fn gpu_decode_suite_times_kernel_and_end_to_end_per_input() {
    let Some(ctx) = context() else { return };
    let inputs = vec![(
        "text".to_string(),
        bench::suite::synthetic_inputs(SMALL.bytes).remove(2).1,
    )];
    let rows = bench::suite::gpu_decode(&ctx, &inputs, &SMALL).unwrap();
    let names: Vec<_> = rows.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "gpu.lz4.decompress.naive.kernel",
            "gpu.lz4.decompress.naive.e2e",
            "gpu.lz4.decompress.coop.kernel",
            "gpu.lz4.decompress.coop.e2e",
        ]
    );
    assert_eq!(rows[1].timing, Timing::WallE2e);
    assert_eq!(rows[3].timing, Timing::WallE2e);
    for m in &rows {
        assert!(m.gbps.is_finite() && m.gbps > 0.0, "{m:?}");
        assert!(m.ratio.unwrap() > 1.5, "{m:?}");
    }
}

#[test]
fn gpu_encode_suite_times_kernel_and_end_to_end_per_input() {
    let Some(ctx) = context() else { return };
    let inputs = vec![(
        "text".to_string(),
        bench::suite::synthetic_inputs(SMALL.bytes).remove(2).1,
    )];
    let rows = bench::suite::gpu_encode(&ctx, &inputs, &SMALL).unwrap();
    let names: Vec<_> = rows.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, ["gpu.lz4.compress.kernel", "gpu.lz4.compress.e2e"]);
    assert_eq!(rows[1].timing, Timing::WallE2e);
    for m in &rows {
        assert!(m.gbps.is_finite() && m.gbps > 0.0, "{m:?}");
        assert!(m.ratio.unwrap() > 1.5, "{m:?}");
    }
}

#[test]
fn glz_suite_measures_both_directions_with_and_without_groups() {
    let Some(ctx) = context() else { return };
    let inputs = vec![(
        "text".to_string(),
        bench::suite::synthetic_inputs(SMALL.bytes).remove(2).1,
    )];
    let rows = bench::suite::glz(Some(&ctx), &inputs, &SMALL).unwrap();
    let names: Vec<_> = rows.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "cpu.glz.compress.mt",
            "cpu.glz.decompress.mt",
            "gpu.glz.compress.kernel",
            "gpu.glz.compress.e2e",
            "gpu.glz.decompress.kernel",
            "gpu.glz.decompress.e2e",
            "cpu.glz-g64.compress.mt",
            "gpu.glz-g64.compress.kernel",
            "gpu.glz-g64.compress.e2e",
            "gpu.glz-g64.decompress.kernel",
            "gpu.glz-g64.decompress.e2e",
        ]
    );
    for m in &rows {
        assert!(m.gbps.is_finite() && m.gbps > 0.0, "{m:?}");
        assert!(m.ratio.unwrap() > 1.3, "{m:?}");
    }
}

#[test]
fn glz_suite_without_gpu_measures_the_cpu_paths() {
    let inputs = vec![(
        "text".to_string(),
        bench::suite::synthetic_inputs(1 << 16).remove(2).1,
    )];
    let names: Vec<_> = bench::suite::glz(None, &inputs, &SMALL)
        .unwrap()
        .into_iter()
        .map(|m| m.name)
        .collect();
    assert_eq!(
        names,
        [
            "cpu.glz.compress.mt",
            "cpu.glz.decompress.mt",
            "cpu.glz-g64.compress.mt"
        ]
    );
}

const FILTER_CPU_ROWS: [&str; 6] = [
    "compress.none.mt",
    "compress.auto.mt",
    "auto.wins.none",
    "auto.wins.shuffle-4",
    "auto.wins.delta-4",
    "auto.wins.stored",
];

#[test]
fn numeric_inputs_are_named_deterministic_and_exact_length() {
    let inputs = bench::suite::numeric_inputs(100_003);
    let names: Vec<_> = inputs.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        ["f32-points", "sorted-u32", "i16-audio", "u64-timestamps"]
    );
    for (name, data) in &inputs {
        assert_eq!(data.len(), 100_003, "{name}");
    }
    assert_eq!(inputs, bench::suite::numeric_inputs(100_003));
}

#[test]
fn filters_suite_without_gpu_compares_none_and_auto_and_counts_wins() {
    let inputs = vec![bench::suite::numeric_inputs(1 << 18).remove(1)];
    let rows = bench::suite::filters(None, &inputs, &SMALL).unwrap();
    let names: Vec<_> = rows.iter().map(|m| m.name.as_str()).collect();
    let expected: Vec<String> = ["lz4", "glz"]
        .iter()
        .flat_map(|codec| {
            FILTER_CPU_ROWS
                .iter()
                .chain(&["compress.exhaustive.mt"])
                .map(|r| format!("cpu.{codec}.{r}"))
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(names, expected);
    for codec in ["lz4", "glz"] {
        let get = |r: &str| {
            rows.iter()
                .find(|m| m.name == format!("cpu.{codec}.{r}"))
                .unwrap()
        };
        let (none, auto) = (get("compress.none.mt"), get("compress.auto.mt"));
        assert!(none.gbps > 0.0 && auto.gbps > 0.0);
        // Sorted integers: filters must pay off.
        assert!(
            auto.ratio.unwrap() > 1.2 * none.ratio.unwrap(),
            "{codec}: {auto:?} vs {none:?}"
        );
        let shares: f64 = FILTER_CPU_ROWS[2..].iter().map(|r| get(r).gbps).sum();
        assert!(
            (shares - 1.0).abs() < 1e-9,
            "{codec}: shares sum to {shares}"
        );
        assert!(get("auto.wins.none").gbps < 0.5, "{codec}");
    }
}

#[test]
fn filters_suite_with_gpu_times_filtered_decode() {
    let Some(ctx) = context() else { return };
    let inputs = vec![bench::suite::numeric_inputs(SMALL.bytes).remove(0)];
    let rows = bench::suite::filters(Some(&ctx), &inputs, &SMALL).unwrap();
    let names: Vec<_> = rows.iter().map(|m| m.name.as_str()).collect();
    let mut expected = Vec::new();
    for codec in ["lz4", "glz"] {
        expected.extend(
            FILTER_CPU_ROWS[..2]
                .iter()
                .map(|r| format!("cpu.{codec}.{r}")),
        );
        for r in [
            "decompress.none.kernel",
            "decompress.auto.kernel",
            "decompress.auto.e2e",
            "compress.auto.e2e",
        ] {
            expected.push(format!("gpu.{codec}.{r}"));
        }
        expected.extend(
            FILTER_CPU_ROWS[2..]
                .iter()
                .map(|r| format!("cpu.{codec}.{r}")),
        );
        expected.push(format!("cpu.{codec}.compress.exhaustive.mt"));
        expected.push(format!("gpu.{codec}.compress.exhaustive.e2e"));
    }
    assert_eq!(names, expected);
    for m in rows.iter().filter(|m| m.name.starts_with("gpu.")) {
        assert!(m.gbps.is_finite() && m.gbps > 0.0, "{m:?}");
    }
}
