//! The CPU codec suite on small inputs.

use bench::record::Timing;
use bench::suite::{codecs, synthetic_inputs, SuiteConfig};

const SMALL: SuiteConfig = SuiteConfig {
    bytes: 1 << 20,
    warmup: 0,
    runs: 1,
};

#[test]
fn synthetic_inputs_are_named_and_sized() {
    let inputs = synthetic_inputs(100_000);
    let names: Vec<_> = inputs.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["zeros", "random", "text", "mixed"]);
    assert!(inputs.iter().all(|(_, data)| data.len() == 100_000));
}

#[test]
fn codec_suite_measures_every_cpu_path_per_input() {
    let inputs = vec![(
        "text".to_string(),
        synthetic_inputs(SMALL.bytes).remove(2).1,
    )];
    let rows = codecs(&inputs, &SMALL).unwrap();
    let names: Vec<_> = rows.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "cpu.lz4_flex.compress.1t",
            "cpu.lz4_flex.compress.mt",
            "cpu.greedy.compress.mt",
            "cpu.lz4_flex.decompress.1t",
            "cpu.lz4_flex.decompress.mt",
            "cpu.handwritten.decompress.mt",
        ]
    );
    for m in &rows {
        assert_eq!(m.input, "text");
        assert_eq!(m.timing, Timing::Cpu);
        assert!(m.gbps.is_finite() && m.gbps > 0.0, "{m:?}");
        let ratio = m.ratio.expect("codec rows carry a ratio");
        assert!(ratio > 1.5, "{m:?}");
    }
}

#[test]
fn incompressible_input_has_ratio_just_under_one() {
    let inputs = vec![(
        "random".to_string(),
        synthetic_inputs(SMALL.bytes).remove(1).1,
    )];
    for m in codecs(&inputs, &SMALL).unwrap() {
        let ratio = m.ratio.unwrap();
        assert!((0.99..=1.0).contains(&ratio), "{m:?}");
    }
}

#[test]
fn zstd_suite_measures_64k_chunks_and_whole_files_at_levels_1_and_3() {
    let inputs = vec![(
        "text".to_string(),
        synthetic_inputs(SMALL.bytes).remove(2).1,
    )];
    let rows = bench::suite::zstd(&inputs, &SMALL).unwrap();
    let names: Vec<_> = rows.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "cpu.zstd-1.compress.mt",
            "cpu.zstd-1.decompress.mt",
            "cpu.zstd-1-file.compress.mt",
            "cpu.zstd-1-file.decompress.1t",
            "cpu.zstd-3.compress.mt",
            "cpu.zstd-3.decompress.mt",
            "cpu.zstd-3-file.compress.mt",
            "cpu.zstd-3-file.decompress.1t",
        ]
    );
    for m in &rows {
        assert_eq!((m.input.as_str(), m.timing), ("text", Timing::Cpu));
        assert!(m.gbps.is_finite() && m.gbps > 0.0, "{m:?}");
        assert!(m.ratio.expect("zstd rows carry a ratio") > 1.5, "{m:?}");
    }
    let ratio = |name: &str| rows.iter().find(|m| m.name == name).unwrap().ratio.unwrap();
    // A whole file also matches across chunk boundaries; level 3 searches harder.
    assert!(ratio("cpu.zstd-1-file.compress.mt") > ratio("cpu.zstd-1.compress.mt"));
    assert!(ratio("cpu.zstd-3.compress.mt") > ratio("cpu.zstd-1.compress.mt"));
}
