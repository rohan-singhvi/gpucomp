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

/// One context per backend (Metal, D3D12, Vulkan) that has an adapter here.
pub fn contexts() -> Vec<(&'static str, Context)> {
    let backends = [
        ("metal", gpu::wgpu::Backends::METAL),
        ("dx12", gpu::wgpu::Backends::DX12),
        ("vulkan", gpu::wgpu::Backends::VULKAN),
    ];
    let found: Vec<_> = backends
        .into_iter()
        .filter_map(|(name, backends)| {
            Context::new(&ContextOptions {
                backends: Some(backends),
            })
            .ok()
            .map(|ctx| (name, ctx))
        })
        .collect();
    if found.is_empty() {
        eprintln!("skipping GPU test: no adapter on any backend");
    }
    found
}

/// Canterbury corpus files, if `scripts/fetch_corpus` has been run.
pub fn canterbury() -> Vec<(String, Vec<u8>)> {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/corpus/canterbury");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!(
            "note: {} missing; run scripts/fetch_corpus for the full matrix",
            dir.display()
        );
        return Vec::new();
    };
    let mut files: Vec<_> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let name = format!("canterbury/{}", p.file_name().unwrap().to_string_lossy());
            (name, std::fs::read(&p).unwrap())
        })
        .collect()
}

/// Numeric inputs that filters help (M7): sorted u32s, smooth f32 xyz points,
/// i16 sine + noise and u64 timestamps, each with a ragged tail.
pub fn numeric() -> Vec<(&'static str, Vec<u8>)> {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut v = 7u32;
    let sorted: Vec<u8> = (0..9000)
        .flat_map(|_| {
            v = v.wrapping_add((next() % 40) as u32);
            v.to_le_bytes()
        })
        .chain([1, 2, 3])
        .collect();
    let points: Vec<u8> = (0..4001)
        .flat_map(|i| {
            let t = i as f32 * 0.002;
            [t.sin() * 50.0, t.cos() * 50.0, t]
                .into_iter()
                .flat_map(f32::to_le_bytes)
        })
        .collect();
    let audio: Vec<u8> = (0..20_001)
        .flat_map(|i| {
            let x = (i as f32 * 0.03).sin() * 8000.0 + (next() % 64) as f32;
            (x as i16).to_le_bytes()
        })
        .collect();
    let mut t = 1_700_000_000_000u64;
    let stamps: Vec<u8> = (0..5000u64)
        .flat_map(|i| {
            t += 1000 + (i * 7919) % 17;
            t.to_le_bytes()
        })
        .chain([9; 5])
        .collect();
    vec![
        ("sorted u32", sorted),
        ("f32 points", points),
        ("i16 audio", audio),
        ("u64 timestamps", stamps),
    ]
}
