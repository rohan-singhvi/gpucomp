//! M7: GPU forward and inverse filters must match the CPU transforms exactly.

mod common;

use common::{context, random};
use format::Filter;
use gpu::filter::{Direction, FilterError, FilterJob, FilterKernels};
use proptest::prelude::*;

const WIDTHS: [u8; 4] = [1, 2, 4, 8];

fn filters() -> Vec<Filter> {
    let mut out = Vec::new();
    out.extend(WIDTHS.map(|width| Filter::Shuffle { width }));
    out.extend(WIDTHS.map(|width| Filter::Delta { width }));
    out
}

fn cpu(direction: Direction, filter: Filter, data: &[u8]) -> Vec<u8> {
    match direction {
        Direction::Forward => cpu::filter::forward(filter, data),
        Direction::Inverse => cpu::filter::inverse(filter, data),
    }
}

/// Smooth u16 samples: plenty of structure, so wrong bytes show.
fn samples(n: usize) -> Vec<u8> {
    (0..n.div_ceil(2))
        .flat_map(|i| (((i as f32 * 0.01).sin() * 3000.0) as i16).to_le_bytes())
        .take(n)
        .collect()
}

const LENGTHS: [u32; 14] = [0, 1, 3, 4, 7, 8, 9, 15, 17, 100, 1023, 4096, 4099, 70_001];

#[test]
fn single_jobs_match_the_cpu_for_every_filter_length_and_direction() {
    let Some(ctx) = context() else { return };
    let kernels = FilterKernels::new(&ctx);
    for direction in [Direction::Forward, Direction::Inverse] {
        for filter in filters() {
            for len in LENGTHS {
                for (kind, data) in [
                    ("random", random(len as usize, u64::from(len) + 1)),
                    ("samples", samples(len as usize)),
                ] {
                    let job = FilterJob {
                        filter,
                        src_offset: 0,
                        dst_offset: 0,
                        len,
                    };
                    let out = kernels
                        .run(&ctx, direction, &data, &[job], &vec![0; len as usize])
                        .unwrap();
                    assert!(
                        out == cpu(direction, filter, &data),
                        "{direction:?} {filter:?} len {len} {kind}"
                    );
                }
            }
        }
    }
}

#[test]
fn many_jobs_in_one_dispatch_leave_other_bytes_untouched() {
    let Some(ctx) = context() else { return };
    let kernels = FilterKernels::new(&ctx);
    let src = random(200_000, 7);
    // Chunks of a 4 KiB-chunked buffer, filtered into a shifted, larger output.
    let mut jobs = Vec::new();
    let all = filters();
    let mut src_at = 0u32;
    let mut dst_at = 8u32;
    for k in 0..30u32 {
        let len = if k == 29 { 4093 } else { 4096 };
        let filter = if k % 5 == 4 {
            Filter::None
        } else {
            all[k as usize % all.len()]
        };
        jobs.push(FilterJob {
            filter,
            src_offset: src_at,
            dst_offset: dst_at,
            len,
        });
        src_at += len;
        dst_at += len + 12; // leave a gap of untouched bytes between jobs
    }
    let initial = random(dst_at as usize + 3, 99);
    for direction in [Direction::Forward, Direction::Inverse] {
        let out = kernels.run(&ctx, direction, &src, &jobs, &initial).unwrap();
        let mut expected = initial.clone();
        for job in &jobs {
            if job.filter == Filter::None {
                continue; // `None` jobs are skipped
            }
            let s = job.src_offset as usize;
            let d = job.dst_offset as usize;
            let n = job.len as usize;
            expected[d..d + n].copy_from_slice(&cpu(direction, job.filter, &src[s..s + n]));
        }
        assert!(out == expected, "{direction:?}");
    }
}

#[test]
fn invalid_jobs_are_rejected() {
    let Some(ctx) = context() else { return };
    let kernels = FilterKernels::new(&ctx);
    let src = vec![1u8; 64];
    let dst = vec![0u8; 64];
    let job = |src_offset, dst_offset, len| FilterJob {
        filter: Filter::Delta { width: 4 },
        src_offset,
        dst_offset,
        len,
    };
    let run = |jobs: &[FilterJob]| kernels.run(&ctx, Direction::Forward, &src, jobs, &dst);
    assert!(matches!(
        run(&[job(2, 0, 8)]),
        Err(FilterError::Misaligned { job: 0 })
    ));
    assert!(matches!(
        run(&[job(0, 0, 8), job(0, 6, 8)]),
        Err(FilterError::Misaligned { job: 1 })
    ));
    assert!(matches!(
        run(&[job(60, 0, 8)]),
        Err(FilterError::OutOfBounds { job: 0 })
    ));
    assert!(matches!(
        run(&[job(0, 60, 8)]),
        Err(FilterError::OutOfBounds { job: 0 })
    ));
    assert!(matches!(
        run(&[job(0, 0, 8), job(0, 4, 8)]),
        Err(FilterError::Overlap { job: 1 })
    ));
    // Width checks come from the format: only 1, 2, 4 and 8.
    let bad = FilterJob {
        filter: Filter::Shuffle { width: 3 },
        ..job(0, 0, 8)
    };
    assert!(matches!(run(&[bad]), Err(FilterError::BadWidth { job: 0 })));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn gpu_forward_matches_cpu_and_gpu_inverse_undoes_it(
        data in proptest::collection::vec(any::<u8>(), 0..3000),
        which in 0usize..8,
    ) {
        let Some(ctx) = context() else { return Ok(()) };
        let kernels = FilterKernels::new(&ctx);
        let filter = filters()[which];
        let job = FilterJob {
            filter,
            src_offset: 0,
            dst_offset: 0,
            len: data.len() as u32,
        };
        let zero = vec![0; data.len()];
        let forward = kernels.run(&ctx, Direction::Forward, &data, &[job], &zero).unwrap();
        prop_assert_eq!(&forward, &cpu::filter::forward(filter, &data));
        let back = kernels.run(&ctx, Direction::Inverse, &forward, &[job], &zero).unwrap();
        prop_assert_eq!(&back, &data);
    }
}
