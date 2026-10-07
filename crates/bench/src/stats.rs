//! Timing statistics.

use std::time::Duration;

/// Runs `sample` `warmup` times (discarded), then `runs` times, and returns
/// the median of the measured durations. `runs` must be nonzero.
pub fn median_of<F: FnMut() -> anyhow::Result<Duration>>(
    warmup: usize,
    runs: usize,
    mut sample: F,
) -> anyhow::Result<Duration> {
    for _ in 0..warmup {
        sample()?;
    }
    let mut samples = (0..runs)
        .map(|_| sample())
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(median(&mut samples))
}

/// Median of `samples`; for an even count, the mean of the two middle values.
pub fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    let mid = samples.len() / 2;
    if samples.len() % 2 == 1 {
        samples[mid]
    } else {
        (samples[mid - 1] + samples[mid]) / 2
    }
}

/// Throughput in GB/s (10^9 bytes per second).
pub fn gbps(bytes: u64, elapsed: Duration) -> f64 {
    bytes as f64 / elapsed.as_secs_f64() / 1e9
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn median_of_odd_count_is_middle_value() {
        assert_eq!(median(&mut [ms(9), ms(1), ms(5)]), ms(5));
    }

    #[test]
    fn median_of_even_count_is_mean_of_middle_pair() {
        assert_eq!(median(&mut [ms(4), ms(1), ms(2), ms(100)]), ms(3));
    }

    #[test]
    fn gbps_is_decimal_gigabytes_per_second() {
        assert_eq!(gbps(2_000_000_000, Duration::from_secs(1)), 2.0);
        assert_eq!(gbps(1_000_000, ms(1)), 1.0);
    }

    #[test]
    fn median_of_discards_warmup_samples() {
        let mut calls = 0u64;
        // Warmup samples are huge; they must not move the median.
        let m = median_of(2, 3, || {
            calls += 1;
            Ok(if calls <= 2 { ms(1000) } else { ms(calls) })
        })
        .unwrap();
        assert_eq!(calls, 5);
        assert_eq!(m, ms(4));
    }
}
