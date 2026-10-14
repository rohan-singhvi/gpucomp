//! An ordered pipeline for streaming: a reader fills buffers in order on the
//! calling thread, rayon workers process several of them at once, and a
//! writer thread consumes the results in order. Buffers are recycled, and at
//! most `depth` items are in flight, so memory stays bounded.

use std::collections::BTreeMap;
use std::sync::mpsc;

/// `rayon::in_place_scope` on `pool`, or on the global pool.
fn in_place_scope<'scope, R>(
    pool: Option<&rayon::ThreadPool>,
    op: impl FnOnce(&rayon::Scope<'scope>) -> R,
) -> R {
    match pool {
        Some(pool) => pool.in_place_scope(op),
        None => rayon::in_place_scope(op),
    }
}

/// An item's buffers and its result, tagged with its position.
type Done<M, E> = (usize, Vec<u8>, Vec<u8>, Result<M, E>);

/// Runs `read` until it returns `Ok(None)`, `work` on each item it read and
/// `write` on each result, in read order. `read` fills an input buffer and
/// returns the item's context; `work` fills an output buffer from it. The
/// first error, in item order, stops the pipeline: later items' results are
/// not written.
///
/// `read` runs on its own thread, `work` on `pool` (rayon's global pool if
/// `None`; several items at once) and `write` on the calling thread, so no
/// rayon worker ever waits for I/O.
pub(crate) fn ordered<C, M, E>(
    pool: Option<&rayon::ThreadPool>,
    depth: usize,
    mut read: impl FnMut(&mut Vec<u8>) -> Result<Option<C>, E> + Send,
    work: impl Fn(C, &[u8], &mut Vec<u8>) -> Result<M, E> + Sync,
    mut write: impl FnMut(&[u8], M) -> Result<(), E>,
) -> Result<(), E>
where
    C: Send,
    M: Send,
    E: Send,
{
    // Each in-flight item holds one pair of buffers.
    let (free_tx, free_rx) = mpsc::channel::<(Vec<u8>, Vec<u8>)>();
    for _ in 0..depth.max(1) {
        let _ = free_tx.send((Vec::new(), Vec::new()));
    }
    let (done_tx, done_rx) = mpsc::channel::<Done<M, E>>();
    let work = &work;
    std::thread::scope(|s| {
        let reader = s.spawn(move || {
            in_place_scope(pool, |scope| {
                let mut k = 0;
                // Ends when the writer stops (and drops `free_tx`).
                while let Ok((mut input, mut out)) = free_rx.recv() {
                    input.clear();
                    out.clear();
                    match read(&mut input) {
                        Ok(Some(context)) => {
                            let done_tx = done_tx.clone();
                            scope.spawn(move |_| {
                                let result = work(context, &input, &mut out);
                                // The writer may have stopped.
                                let _ = done_tx.send((k, input, out, result));
                            });
                            k += 1;
                        }
                        Ok(None) => break,
                        Err(e) => {
                            let _ = done_tx.send((k, input, out, Err(e)));
                            break;
                        }
                    }
                }
            });
        });
        let mut written = || {
            let mut pending = BTreeMap::new();
            let mut next = 0;
            for (k, input, out, result) in &done_rx {
                pending.insert(k, (input, out, result));
                while let Some((input, out, result)) = pending.remove(&next) {
                    write(&out, result?)?;
                    next += 1;
                    let _ = free_tx.send((input, out));
                }
            }
            Ok(())
        };
        let result = written();
        // Let the reader and the workers finish early after an error.
        drop(done_rx);
        drop(free_tx);
        if let Err(panic) = reader.join() {
            std::panic::resume_unwind(panic);
        }
        result
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    /// Items 0..n, each a buffer holding its index; work echoes it after a
    /// delay that varies, so later items often finish first.
    fn run(n: usize, depth: usize, fail_at: Option<usize>) -> (Result<(), usize>, Vec<u64>, usize) {
        let mut next = 0usize;
        let in_flight = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let mut written = Vec::new();
        let result = ordered(
            None,
            depth,
            |buf| {
                if next == n {
                    return Ok(None);
                }
                let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                buf.extend_from_slice(&(next as u64).to_le_bytes());
                next += 1;
                Ok(Some(next - 1))
            },
            |k, input, out| {
                std::thread::sleep(Duration::from_micros(((k * 7919) % 5) as u64 * 300));
                if Some(k) == fail_at {
                    return Err(k);
                }
                out.extend_from_slice(input);
                Ok(k)
            },
            |out, k| {
                assert_eq!(u64::from_le_bytes(out.try_into().unwrap()), k as u64);
                written.push(k as u64);
                in_flight.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            },
        );
        (result, written, peak.load(Ordering::SeqCst))
    }

    #[test]
    fn results_are_written_in_read_order() {
        let (result, written, _) = run(200, 4, None);
        assert_eq!(result, Ok(()));
        assert_eq!(written, (0..200).collect::<Vec<_>>());
    }

    #[test]
    fn at_most_depth_items_are_in_flight() {
        let (_, _, peak) = run(100, 3, None);
        assert!((1..=3).contains(&peak), "peak {peak}");
    }

    #[test]
    fn the_first_error_in_item_order_stops_the_writes() {
        let (result, written, _) = run(100, 6, Some(37));
        assert_eq!(result, Err(37));
        assert_eq!(written, (0..37).collect::<Vec<_>>());
    }

    #[test]
    fn a_read_error_is_reported_after_the_items_before_it() {
        let mut next = 0u64;
        let mut written = Vec::new();
        let result = ordered(
            None,
            2,
            |buf| {
                next += 1;
                if next == 5 {
                    return Err("read");
                }
                buf.push(next as u8);
                Ok(Some(()))
            },
            |(), input, out| {
                out.extend_from_slice(input);
                Ok(())
            },
            |out, ()| {
                written.push(out[0]);
                Ok(())
            },
        );
        assert_eq!(result, Err("read"));
        assert_eq!(written, [1, 2, 3, 4]);
    }

    #[test]
    fn nothing_to_read_writes_nothing() {
        let (result, written, _) = run(0, 4, None);
        assert_eq!(result, Ok(()));
        assert!(written.is_empty());
    }
}
