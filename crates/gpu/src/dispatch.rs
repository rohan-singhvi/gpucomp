//! Dispatch-size helpers.

/// Splits `workgroups` into an `(x, y)` grid with `x <= max_per_dim`, so large
/// inputs stay within `max_compute_workgroups_per_dimension`. Shaders recover
/// the linear workgroup index as `wg.x + wg.y * num_workgroups.x` and must
/// bounds-check, because `x * y` can exceed `workgroups`.
pub fn dispatch_grid(workgroups: u32, max_per_dim: u32) -> (u32, u32) {
    if workgroups <= max_per_dim {
        return (workgroups, 1);
    }
    (max_per_dim, workgroups.div_ceil(max_per_dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_counts_stay_one_dimensional() {
        assert_eq!(dispatch_grid(100, 65535), (100, 1));
    }

    #[test]
    fn large_counts_wrap_into_rows_within_the_limit() {
        let (x, y) = dispatch_grid(200_000, 65535);
        assert!(x <= 65535 && y <= 65535, "({x}, {y})");
        assert!(u64::from(x) * u64::from(y) >= 200_000, "({x}, {y})");
    }

    #[test]
    fn zero_workgroups_is_an_empty_grid() {
        assert_eq!(dispatch_grid(0, 65535), (0, 1));
    }
}
