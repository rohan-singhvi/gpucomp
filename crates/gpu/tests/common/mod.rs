//! Shared helpers for GPU integration tests.
#![allow(dead_code)]

use gpu::{Context, ContextOptions, GpuError};

/// A GPU context, or `None` (with a message) when the machine has no adapter.
pub fn context() -> Option<Context> {
    match Context::new(&ContextOptions::default()) {
        Ok(ctx) => Some(ctx),
        Err(GpuError::NoAdapter(e)) => {
            eprintln!("skipping GPU test: no adapter ({e})");
            None
        }
        Err(e) => panic!("GPU context creation failed: {e}"),
    }
}

pub fn random(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s as u8
        })
        .collect()
}

pub fn text(n: usize) -> Vec<u8> {
    let words = [
        "lorem ", "ipsum ", "dolor ", "sit ", "amet, ", "gpu ", "chunk ", "\n",
    ];
    let mut out = Vec::with_capacity(n + 8);
    let mut i = 0usize;
    while out.len() < n {
        out.extend_from_slice(words[(i * 7 + i / 3) % words.len()].as_bytes());
        i += 1;
    }
    out.truncate(n);
    out
}

pub const CHUNK: u32 = 4096;

/// The M1 round-trip suite's inputs.
pub fn fixtures() -> Vec<(&'static str, Vec<u8>)> {
    let c = CHUNK as usize;
    vec![
        ("empty", vec![]),
        ("one byte", vec![42]),
        ("exactly one chunk", text(c)),
        ("chunk - 1", text(c - 1)),
        ("chunk + 1", text(c + 1)),
        ("random", random(10 * c + 123, 1)),
        ("zeros", vec![0; 10 * c]),
        ("text", text(10 * c + 7)),
        (
            "mixed",
            [text(3 * c), random(2 * c, 2), vec![7; 3 * c]].concat(),
        ),
    ]
}
