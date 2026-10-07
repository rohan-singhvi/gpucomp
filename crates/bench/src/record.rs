//! One benchmark run, as stored in `bench/results/*.json`.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Timing {
    /// GPU timestamp queries around the compute pass(es) only.
    GpuTimestamp,
    /// Wall clock around upload + GPU work + readback.
    WallE2e,
    /// Wall clock around submit + wait, data already resident on the GPU.
    /// Used for kernels when timestamp queries are unavailable.
    WallGpu,
    /// Wall clock of CPU-only work.
    Cpu,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    /// Dotted name, e.g. `gpu.upload` or `cpu.lz4_flex.compress`.
    pub name: String,
    /// Input description, e.g. `random` or `silesia/dickens`.
    pub input: String,
    /// Uncompressed bytes processed per iteration.
    pub bytes: u64,
    /// Median throughput in GB/s of uncompressed data.
    pub gbps: f64,
    /// Original / compressed size, for codec measurements.
    pub ratio: Option<f64>,
    pub timing: Timing,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Run {
    /// UTC timestamp, RFC 3339 (sorts chronologically as a string).
    pub date: String,
    /// Short git commit, with `-dirty` if the tree had changes.
    pub commit: String,
    /// Milestone or change label, e.g. `M0` or `M5-atomicOr`.
    pub label: String,
    /// What changed since the previous run, in a few words.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
    pub os: String,
    pub adapter: String,
    pub backend: String,
    pub measurements: Vec<Measurement>,
}

impl Run {
    /// File name under `bench/results/`: `<date>-<label>-<adapter>.json`,
    /// with everything outside `[A-Za-z0-9.-]` replaced by `-`.
    pub fn file_name(&self) -> String {
        let stem: String = format!("{}-{}-{}", self.date, self.label, self.adapter)
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                    c
                } else {
                    '-'
                }
            })
            .collect();
        format!("{stem}.json")
    }
}

#[cfg(test)]
pub(crate) fn sample_run() -> Run {
    Run {
        date: "2026-10-07T20:15:03Z".into(),
        commit: "bd4fbb7".into(),
        label: "M0".into(),
        note: String::new(),
        os: "macos".into(),
        adapter: "Apple M4 Pro".into(),
        backend: "metal".into(),
        measurements: vec![Measurement {
            name: "gpu.upload".into(),
            input: "random".into(),
            bytes: 256 << 20,
            gbps: 12.5,
            ratio: None,
            timing: Timing::WallE2e,
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_round_trips_through_json() {
        let run = sample_run();
        let json = serde_json::to_string_pretty(&run).unwrap();
        assert_eq!(serde_json::from_str::<Run>(&json).unwrap(), run);
    }

    #[test]
    fn timing_serializes_as_kebab_case() {
        assert_eq!(
            serde_json::to_string(&Timing::GpuTimestamp).unwrap(),
            "\"gpu-timestamp\""
        );
    }

    #[test]
    fn file_name_is_date_label_adapter_with_safe_characters() {
        assert_eq!(
            sample_run().file_name(),
            "2026-10-07T20-15-03Z-M0-Apple-M4-Pro.json"
        );
    }
}
