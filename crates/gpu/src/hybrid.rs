//! CPU+GPU compression. Every chunk encodes to the same bytes on the GPU and
//! with its CPU twin (`cpu::container` with the twin encoder; the GPU tests
//! enforce this), so the two devices can share one input: the GPU compresses
//! ranges of chunks from the front while the CPU compresses ranges from the
//! back, and [`format::splice`] joins the parts. The file equals
//! [`GpuEncoder::compress`]'s, whatever the split, so the schedule can adapt
//! to the devices' speeds without making the output nondeterministic.

use std::ops::Range;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use cpu::container::{CompressOptions, Encoder};
use format::Codec;

use crate::decode::{GpuDecodeError, GpuDecoder};
use crate::encode::{EncodeParams, FilterMode, GpuCompressOptions, GpuEncodeError, GpuEncoder};
use crate::Context;

/// Which device compresses a range of chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Gpu,
    Cpu,
}

/// The GPU stops claiming chunks when fewer than this many bytes are left,
/// and claims at least this much otherwise.
const MIN_GPU_BYTES: usize = 4 << 20;
/// The GPU stops claiming when its measured share of the work falls below this:
/// its batches also cost the CPU copies and memory bandwidth.
const MIN_GPU_SHARE: f64 = 0.25;
/// The GPU joins only when the CPU alone would need longer than this for
/// what's left. A GPU pass has a latency floor (a chunk is encoded or decoded
/// by one workgroup, partly serially: ~5 ms for a GLZ-E decode and ~12 ms for
/// LZ4 here), so shorter jobs gain nothing from it.
const GPU_MIN_TIME: Duration = Duration::from_millis(40);
/// Chunks a CPU worker claims at a time (one worker per rayon thread, each
/// compressing its own claims, so no worker waits for another).
const CPU_CLAIM: usize = 4;

/// The CPU twin's compression options for `options` with `params`: the same
/// file, chunk for chunk.
pub fn twin_options(options: &GpuCompressOptions, params: EncodeParams) -> CompressOptions {
    let lz = cpu::lz4::encode::Params {
        block: params.block as usize,
        hash_log: params.hash_log,
        probe_len: params.probe_len as usize,
        lazy: params.lazy,
        depth: params.depth as usize,
    };
    let encoder = match options.codec {
        Codec::Glz | Codec::GlzE => Encoder::Glz(cpu::glz::GlzParams {
            lz,
            independent_groups: options.independent_groups,
        }),
        Codec::Lz4 | Codec::Stored => Encoder::Greedy(lz),
    };
    CompressOptions {
        codec: options.codec,
        chunk_size: options.chunk_size,
        encoder,
        checksums: options.checksums,
        level: options.level,
        filters: match options.filters {
            FilterMode::None => cpu::container::FilterMode::None,
            FilterMode::Auto => cpu::container::FilterMode::Auto,
            FilterMode::Exhaustive => cpu::container::FilterMode::Exhaustive,
        },
    }
}

/// Compresses `input` on the GPU and the CPU together. Every rayon thread but
/// one claims a few chunks at a time from the back; the GPU claims from the
/// front by [`Schedule::gpu_claim`]'s rule (half of what's left once the CPU's
/// rate is known, then its share by the measured rates), so both devices
/// finish at about the same time.
pub fn compress(
    ctx: &Context,
    encoder: &GpuEncoder,
    input: &[u8],
    options: &GpuCompressOptions,
) -> Result<Vec<u8>, GpuEncodeError> {
    if input.is_empty() {
        return encoder.compress(ctx, input, options);
    }
    let chunk = options.chunk_size.max(1) as usize;
    let twin = twin_options(options, encoder.params());
    let schedule = Mutex::new(Schedule::new(input.len().div_ceil(chunk), chunk));
    let lock = || schedule.lock().unwrap_or_else(|e| e.into_inner());
    type Parts = Vec<(usize, Vec<u8>)>;
    let cpu_parts: Mutex<Result<Parts, GpuEncodeError>> = Mutex::new(Ok(Vec::new()));
    let gpu = std::thread::scope(|s| {
        let gpu = s.spawn(|| -> Result<Parts, GpuEncodeError> {
            let mut parts = Vec::new();
            loop {
                // A statement of its own, so the guard drops before the work.
                let claim = lock().gpu_claim();
                let range = match claim {
                    GpuClaim::Range(range) => range,
                    GpuClaim::Wait => {
                        std::thread::sleep(Duration::from_micros(200));
                        continue;
                    }
                    GpuClaim::Done => break,
                };
                let start = Instant::now();
                match encoder.compress(ctx, chunk_range(input, &range, chunk), options) {
                    Ok(file) => parts.push((range.start, file)),
                    Err(e) => {
                        lock().stop();
                        return Err(e);
                    }
                }
                lock().gpu.add(range.len(), start.elapsed());
            }
            Ok(parts)
        });
        // Ending the CPU side (even by a panic) stops the schedule, so the GPU
        // thread never waits for a CPU rate that won't come.
        let stop = StopOnDrop(|| lock().stop());
        rayon::scope(|r| {
            // One worker fewer than threads: the GPU thread needs a core for its
            // copies and submissions.
            for _ in 0..rayon::current_num_threads().saturating_sub(1).max(1) {
                r.spawn(|_| loop {
                    let claim = lock().cpu_claim();
                    let Some(range) = claim else { break };
                    let file = cpu::container::compress(chunk_range(input, &range, chunk), &twin);
                    let mut parts = cpu_parts.lock().unwrap_or_else(|e| e.into_inner());
                    match (file, parts.as_mut()) {
                        (Ok(file), Ok(parts)) => parts.push((range.start, file)),
                        (Err(e), Ok(_)) => {
                            *parts = Err(e.into());
                            lock().stop();
                        }
                        (_, Err(_)) => {}
                    }
                    drop(parts);
                    lock().cpu_done(range.len());
                });
            }
        });
        drop(stop);
        gpu.join()
    });
    let cpu = cpu_parts.into_inner().unwrap_or_else(|e| e.into_inner());
    let mut parts = gpu.expect("the GPU thread panicked")?;
    parts.extend(cpu?);
    parts.sort_by_key(|(first, _)| *first);
    let file = splice(input.len(), options, &parts);
    // Freeing hundreds of part buffers takes milliseconds; not the caller's.
    std::thread::spawn(move || drop(parts));
    file
}

/// [`compress`] with a fixed split: `parts` are consecutive (device, chunk
/// count) ranges from the start, covering every chunk exactly. For tests and
/// experiments.
pub fn compress_parts(
    ctx: &Context,
    encoder: &GpuEncoder,
    input: &[u8],
    options: &GpuCompressOptions,
    parts: &[(Device, u32)],
) -> Result<Vec<u8>, GpuEncodeError> {
    let chunk = options.chunk_size.max(1) as usize;
    let chunks = input.len().div_ceil(chunk) as u64;
    let covered: u64 = parts.iter().map(|&(_, n)| u64::from(n)).sum();
    if covered != chunks {
        return Err(GpuEncodeError::BadSplit { covered, chunks });
    }
    let twin = twin_options(options, encoder.params());
    let mut files = Vec::with_capacity(parts.len());
    let mut first = 0;
    for &(device, n) in parts {
        let range = first..first + n as usize;
        first = range.end;
        if range.is_empty() {
            continue;
        }
        let slice = chunk_range(input, &range, chunk);
        let file = match device {
            Device::Gpu => encoder.compress(ctx, slice, options)?,
            Device::Cpu => cpu::container::compress(slice, &twin)?,
        };
        files.push((range.start, file));
    }
    splice(input.len(), options, &files)
}

/// Decompresses `file` on the GPU and the CPU together, splitting its chunks
/// between them by speed. The output is the original whatever the split.
/// Unlike [`compress`], the CPU side is one claimant making few, large claims,
/// each decoded by every rayon thread: at CPU decode speeds (up to ~14 GB/s for
/// LZ4) a claim's tail, waiting for the slowest thread, is what costs. It takes
/// half its share of what's left each time, and everything once the GPU is
/// done or won't join (when the CPU alone would finish soon).
pub fn decompress(
    ctx: &Context,
    decoder: &GpuDecoder,
    file: &[u8],
    verify: bool,
) -> Result<Vec<u8>, GpuDecodeError> {
    let index = parse_index(file)?;
    let chunk = index.header.chunk_size as usize;
    let total = index.header.total_size as usize;
    let mut out = vec![0u8; total];
    // Bytes of chunks `range`: whole chunks but the last.
    let bytes = |range: &Range<usize>| (range.end * chunk).min(total) - range.start * chunk;
    // The schedule, and the output of the unclaimed chunks: the GPU splits its
    // claims off the front, the CPU off the back.
    let state = Mutex::new((Schedule::new(index.chunks.len(), chunk), out.as_mut_slice()));
    let lock = || state.lock().unwrap_or_else(|e| e.into_inner());
    let claim = |gpu: bool| {
        let mut state = lock();
        let (schedule, unclaimed) = &mut *state;
        let range = if gpu {
            match schedule.gpu_claim() {
                GpuClaim::Range(range) => range,
                GpuClaim::Wait => return Some(None),
                GpuClaim::Done => return None,
            }
        } else {
            schedule.cpu_claim_guided()?
        };
        let rest = std::mem::take(unclaimed);
        let dst = if gpu {
            let (dst, rest) = rest.split_at_mut(bytes(&range));
            *unclaimed = rest;
            dst
        } else {
            let at = rest.len() - bytes(&range);
            let (rest, dst) = rest.split_at_mut(at);
            *unclaimed = rest;
            dst
        };
        Some(Some((range, dst)))
    };
    let options = cpu_decode_options(&index, verify);
    let (gpu, cpu) = std::thread::scope(|s| {
        let gpu = s.spawn(|| -> Result<(), GpuDecodeError> {
            while let Some(next) = claim(true) {
                let Some((range, dst)) = next else {
                    // Waiting for the CPU's first claim to finish.
                    std::thread::sleep(Duration::from_micros(200));
                    continue;
                };
                let start = Instant::now();
                let mut at = 0;
                let len = range.len();
                let result = decoder.decompress_chunks(ctx, file, range, verify, &mut |b| {
                    dst[at..at + b.len()].copy_from_slice(b);
                    at += b.len();
                    Ok(())
                });
                if let Err(e) = result {
                    lock().0.stop();
                    return Err(e);
                }
                lock().0.gpu.add(len, start.elapsed());
            }
            Ok(())
        });
        let mut cpu: Result<(), GpuDecodeError> = Ok(());
        // See compress: the GPU thread never waits on an ended CPU side.
        let stop = StopOnDrop(|| lock().0.stop());
        while let Some(Some((range, dst))) = claim(false) {
            let len = range.len();
            if let Err(e) =
                cpu::container::decompress_chunks_into(file, &index, range, dst, &options)
            {
                lock().0.stop();
                cpu = Err(e.into());
                break;
            }
            lock().0.cpu_done(len);
        }
        drop(stop);
        (gpu.join(), cpu)
    });
    gpu.expect("the GPU thread panicked")?;
    cpu?;
    Ok(out)
}

/// [`decompress`] with a fixed split: `parts` are consecutive (device, chunk
/// count) ranges from the start, covering every chunk exactly. For tests and
/// experiments.
pub fn decompress_parts(
    ctx: &Context,
    decoder: &GpuDecoder,
    file: &[u8],
    verify: bool,
    parts: &[(Device, u32)],
) -> Result<Vec<u8>, GpuDecodeError> {
    let index = parse_index(file)?;
    let chunks = index.chunks.len() as u64;
    let covered: u64 = parts.iter().map(|&(_, n)| u64::from(n)).sum();
    if covered != chunks {
        return Err(GpuDecodeError::Unsupported(format!(
            "the split covers {covered} chunks, the file has {chunks}"
        )));
    }
    let options = cpu_decode_options(&index, verify);
    let mut out = vec![0u8; index.header.total_size as usize];
    let mut rest = out.as_mut_slice();
    let mut first = 0;
    for &(device, n) in parts {
        let range = first..first + n as usize;
        first = range.end;
        let bytes: usize = index.chunks[range.clone()]
            .iter()
            .map(|c| c.uncomp_size as usize)
            .sum();
        let (dst, tail) = rest.split_at_mut(bytes);
        rest = tail;
        match device {
            Device::Gpu => {
                let mut at = 0;
                decoder.decompress_chunks(ctx, file, range, verify, &mut |b| {
                    dst[at..at + b.len()].copy_from_slice(b);
                    at += b.len();
                    Ok(())
                })?;
            }
            Device::Cpu => {
                cpu::container::decompress_chunks_into(file, &index, range, dst, &options)?;
            }
        }
    }
    Ok(out)
}

/// The header and chunk table of `file`, checked against its data section.
fn parse_index(file: &[u8]) -> Result<format::Index, GpuDecodeError> {
    let index = format::Index::parse(file)?;
    let data = file.len() as u64 - index.data_offset().min(file.len() as u64);
    index.validate(data)?;
    Ok(index)
}

/// The CPU decoder for `index`'s codec: lz4_flex for LZ4 (the fastest), else
/// the hand-written GLZ and GLZ-E decoders.
fn cpu_decode_options(index: &format::Index, verify: bool) -> cpu::container::DecompressOptions {
    cpu::container::DecompressOptions {
        decoder: match index.header.codec {
            Codec::Lz4 => cpu::container::Decoder::Lz4Flex,
            _ => cpu::container::Decoder::HandWritten,
        },
        verify,
    }
}

/// The input bytes of chunks `range`.
fn chunk_range<'a>(input: &'a [u8], range: &Range<usize>, chunk: usize) -> &'a [u8] {
    &input[range.start * chunk..(range.end * chunk).min(input.len())]
}

/// Joins the parts (files of consecutive chunk ranges, in order) into the
/// file for the whole `len`-byte input.
fn splice(
    len: usize,
    options: &GpuCompressOptions,
    parts: &[(usize, Vec<u8>)],
) -> Result<Vec<u8>, GpuEncodeError> {
    let chunk_count = format::chunk_count_for(len as u64, options.chunk_size.max(1));
    let header = format::Header {
        codec: options.codec,
        chunk_size: options.chunk_size,
        chunk_count: u32::try_from(chunk_count)
            .map_err(|_| GpuEncodeError::TooManyChunks(chunk_count))?,
        total_size: len as u64,
        checksums: options.checksums,
        level: options.level,
    };
    let files: Vec<&[u8]> = parts.iter().map(|(_, file)| file.as_slice()).collect();
    let plan = format::splice_plan(header, &files)?;
    // The copies in parallel, each into its own slice of the output.
    let mut out = vec![0u8; plan.len];
    let (head, mut rest) = out.split_at_mut(plan.head.len());
    head.copy_from_slice(&plan.head);
    let mut jobs = Vec::with_capacity(plan.copies.len());
    for (part, range) in &plan.copies {
        let (dst, tail) = rest.split_at_mut(range.len());
        jobs.push((dst, &files[*part][range.clone()]));
        rest = tail;
    }
    use rayon::prelude::*;
    jobs.into_par_iter()
        .for_each(|(dst, src)| dst.copy_from_slice(src));
    Ok(out)
}

/// Calls its closure when dropped.
struct StopOnDrop<F: FnMut()>(F);

impl<F: FnMut()> Drop for StopOnDrop<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}

/// What the GPU does next.
#[derive(Clone, Debug, PartialEq, Eq)]
enum GpuClaim {
    Range(Range<usize>),
    /// Ask again shortly: the CPU's rate isn't known yet.
    Wait,
    Done,
}

/// A device's measured throughput.
#[derive(Default)]
struct Rate {
    chunks: usize,
    time: Duration,
}

impl Rate {
    fn add(&mut self, chunks: usize, time: Duration) {
        self.chunks += chunks;
        self.time += time;
    }

    /// Chunks per second, once measured.
    fn per_second(&self) -> Option<f64> {
        (self.chunks > 0 && !self.time.is_zero())
            .then(|| self.chunks as f64 / self.time.as_secs_f64())
    }
}

/// The unclaimed chunks `front..back`: the GPU claims from the front, the CPU
/// from the back.
struct Schedule {
    front: usize,
    back: usize,
    min_gpu: usize,
    cpu_claim: usize,
    /// GPU time spent per chunk.
    gpu: Rate,
    /// Chunks the CPU workers finished, together, since `started`.
    cpu_chunks: usize,
    started: Instant,
    /// The GPU has stopped claiming, or won't start.
    gpu_done: bool,
    /// The GPU has claimed something.
    gpu_started: bool,
}

impl Schedule {
    fn new(chunks: usize, chunk_size: usize) -> Self {
        Schedule {
            front: 0,
            back: chunks,
            min_gpu: MIN_GPU_BYTES.div_ceil(chunk_size).max(1),
            cpu_claim: CPU_CLAIM,
            gpu: Rate::default(),
            cpu_chunks: 0,
            started: Instant::now(),
            gpu_done: false,
            gpu_started: false,
        }
    }

    fn cpu_done(&mut self, chunks: usize) {
        self.cpu_chunks += chunks;
    }

    /// The CPU workers' combined chunks per second, once they've finished some.
    fn cpu_rate(&self) -> Option<f64> {
        let secs = self.started.elapsed().as_secs_f64();
        (self.cpu_chunks > 0 && secs > 0.0).then(|| self.cpu_chunks as f64 / secs)
    }

    /// The GPU's next range. It waits for the CPU's rate, then joins only if
    /// the CPU alone would need longer than `GPU_MIN_TIME` for what's left,
    /// starting with half of it: a GPU pass has a latency floor, so small
    /// probes would only measure that. Then it takes its share of what's left
    /// by the measured rates, and stops when that share is under
    /// `MIN_GPU_SHARE` (it would only add contention) or less than the
    /// minimum is left.
    fn gpu_claim(&mut self) -> GpuClaim {
        let left = self.back - self.front;
        if self.gpu_done || left < self.min_gpu {
            self.gpu_done = true;
            return GpuClaim::Done;
        }
        let Some(cpu) = self.cpu_rate() else {
            return GpuClaim::Wait;
        };
        let share = match self.gpu.per_second() {
            Some(gpu) => gpu / (gpu + cpu),
            None if self.cpu_would_finish_soon(cpu) => {
                self.gpu_done = true;
                return GpuClaim::Done;
            }
            None => 0.5,
        };
        if share < MIN_GPU_SHARE {
            self.gpu_done = true;
            return GpuClaim::Done;
        }
        self.gpu_started = true;
        let take = ((left as f64 * share).ceil() as usize).clamp(self.min_gpu, left);
        self.front += take;
        GpuClaim::Range(self.front - take..self.front)
    }

    /// Whether the CPU, at `cpu` chunks per second, would finish what's left
    /// within `GPU_MIN_TIME`.
    fn cpu_would_finish_soon(&self, cpu: f64) -> bool {
        ((self.back - self.front) as f64 / cpu) < GPU_MIN_TIME.as_secs_f64()
    }

    /// The CPU's next range when each claim is one parallel call whose tail
    /// (the slowest thread) costs ~0.5 ms, as in decompression: few, large
    /// claims. A sixteenth of what's left before its rate is known, a quarter
    /// while the GPU hasn't been measured, then half the CPU's share, and
    /// everything once the GPU is done (or won't join: see
    /// [`gpu_claim`](Self::gpu_claim)).
    fn cpu_claim_guided(&mut self) -> Option<Range<usize>> {
        let left = self.back - self.front;
        if left == 0 {
            return None;
        }
        if !self.gpu_started
            && self
                .cpu_rate()
                .is_some_and(|c| self.cpu_would_finish_soon(c))
        {
            // The GPU won't join (see gpu_claim).
            self.gpu_done = true;
        }
        let take = if self.gpu_done {
            left
        } else {
            let share = match (self.gpu.per_second(), self.cpu_rate()) {
                (Some(g), Some(c)) => c / (g + c) / 2.0,
                // The GPU may take half of what's left.
                (None, Some(_)) => 1.0 / 4.0,
                (_, None) => 1.0 / 16.0,
            };
            ((left as f64 * share).ceil() as usize).clamp(self.cpu_claim.min(left), left)
        };
        self.back -= take;
        Some(self.back..self.back + take)
    }

    fn cpu_claim(&mut self) -> Option<Range<usize>> {
        let take = self.cpu_claim.min(self.back - self.front);
        if take == 0 {
            return None;
        }
        self.back -= take;
        Some(self.back..self.back + take)
    }

    /// Leaves nothing more to claim (after an error).
    fn stop(&mut self) {
        self.back = self.front;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule(chunks: usize) -> Schedule {
        // 64 KiB chunks: the GPU's minimum is 64 chunks.
        Schedule {
            cpu_claim: 48,
            ..Schedule::new(chunks, 64 << 10)
        }
    }

    /// Sets the CPU's measured rate: `chunks` finished in `ms`.
    fn cpu_did(s: &mut Schedule, chunks: usize, ms: u64) {
        s.cpu_chunks = chunks;
        s.started = Instant::now() - Duration::from_millis(ms);
    }

    #[test]
    fn the_gpu_waits_for_the_cpus_rate_and_starts_with_half() {
        let mut s = schedule(4096);
        assert_eq!(s.min_gpu, 64);
        assert_eq!(s.gpu_claim(), GpuClaim::Wait);
        assert_eq!(s.cpu_claim_guided(), Some(4096 - 256..4096));
        // 256 chunks in 10 ms: the other 3840 would take 150 ms, long enough
        // for the GPU, which starts with half.
        cpu_did(&mut s, 256, 10);
        assert_eq!(s.gpu_claim(), GpuClaim::Range(0..1920));
    }

    #[test]
    fn the_gpu_stays_out_of_jobs_the_cpu_finishes_soon() {
        // 3840 chunks left at 256 per ms: 15 ms.
        let mut s = schedule(4096);
        s.cpu_claim_guided();
        cpu_did(&mut s, 256, 1);
        assert_eq!(s.cpu_claim_guided(), Some(0..3840));
        assert_eq!(s.gpu_claim(), GpuClaim::Done);
    }

    #[test]
    fn the_gpu_takes_its_share_by_rate_once_measured() {
        let mut s = schedule(4096);
        cpu_did(&mut s, 100, 100);
        assert_eq!(s.gpu_claim(), GpuClaim::Range(0..2048));
        // GPU 3 chunks per ms, CPU 1: the GPU takes 3/4 of the rest.
        s.gpu.add(300, Duration::from_millis(100));
        let GpuClaim::Range(r) = s.gpu_claim() else {
            panic!("no claim")
        };
        // The CPU rate moves on with the wall clock, and claims round up.
        let share = r.len() as f64 / 2048.0;
        assert!((0.74..=0.76).contains(&share), "{share}");
        // GPU 1 chunk per ms, CPU 12: under a quarter, so the GPU stops.
        s.gpu = Rate::default();
        s.gpu.add(100, Duration::from_millis(100));
        cpu_did(&mut s, 1200, 100);
        assert_eq!(s.gpu_claim(), GpuClaim::Done);
    }

    #[test]
    fn guided_cpu_claims_shrink_with_the_cpus_share_and_take_all_after_the_gpu() {
        let mut s = schedule(4096);
        // No rates yet: a sixteenth.
        assert_eq!(s.cpu_claim_guided(), Some(4096 - 256..4096));
        // CPU measured, GPU not yet: a quarter (the GPU may take half).
        cpu_did(&mut s, 256, 100);
        assert_eq!(s.cpu_claim_guided(), Some(3840 - 960..3840));
        // GPU 1 chunk per ms, CPU 3: the CPU takes half its 3/4 share.
        s.gpu_started = true;
        s.gpu.add(100, Duration::from_millis(100));
        cpu_did(&mut s, 300, 100);
        let claim = s.cpu_claim_guided().unwrap();
        assert_eq!(claim.end, 2880);
        let share = claim.len() as f64 / 2880.0;
        assert!((0.37..=0.38).contains(&share), "{share}");
        // Once the GPU stops claiming, the CPU takes everything left.
        cpu_did(&mut s, 100_000, 100);
        assert_eq!(s.gpu_claim(), GpuClaim::Done);
        assert_eq!(s.cpu_claim_guided(), Some(0..claim.start));
        assert_eq!(s.cpu_claim_guided(), None);
    }

    #[test]
    fn the_cpu_finishes_what_is_too_small_for_the_gpu() {
        // A slow CPU: 1 chunk per second.
        let mut s = schedule(100);
        cpu_did(&mut s, 1, 1000);
        assert_eq!(s.gpu_claim(), GpuClaim::Range(0..64)); // at least the minimum
        assert_eq!(s.gpu_claim(), GpuClaim::Done); // 36 left
        assert_eq!(s.cpu_claim(), Some(64..100));
        assert_eq!(s.cpu_claim(), None);
    }

    #[test]
    fn claims_cover_every_chunk_once() {
        for chunks in [0, 1, 63, 64, 65, 1000, 4097] {
            let mut s = schedule(chunks);
            let mut seen = vec![0u8; chunks];
            for round in 0.. {
                if round == 2 {
                    cpu_did(&mut s, 7, 1000);
                }
                if round == 4 {
                    s.gpu.add(10, Duration::from_millis(1));
                }
                let gpu = match s.gpu_claim() {
                    GpuClaim::Range(r) => Some(r),
                    _ => None,
                };
                let cpu = if round % 2 == 0 {
                    s.cpu_claim()
                } else {
                    s.cpu_claim_guided()
                };
                if gpu.is_none() && cpu.is_none() {
                    break;
                }
                for r in gpu.into_iter().chain(cpu) {
                    r.for_each(|i| seen[i] += 1);
                }
            }
            assert!(seen.iter().all(|&n| n == 1), "{chunks} chunks");
        }
    }

    #[test]
    fn stopping_leaves_nothing_to_claim() {
        let mut s = schedule(1000);
        s.stop();
        assert_eq!(s.gpu_claim(), GpuClaim::Done);
        assert_eq!((s.cpu_claim(), s.cpu_claim_guided()), (None, None));
    }
}
