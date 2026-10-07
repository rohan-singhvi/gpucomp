//! End-to-end tests of the `gpucomp` binary on temporary files.

use std::path::PathBuf;
use std::process::Command;

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("gpucomp-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn gpucomp(args: &[&str]) -> std::process::Output {
    let out = Command::new(env!("CARGO_BIN_EXE_gpucomp"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "gpucomp {args:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

fn sample() -> Vec<u8> {
    (0..300_000u32)
        .flat_map(|i| format!("line {} of {}\n", i % 977, i / 977).into_bytes())
        .take(300_000)
        .collect()
}

#[test]
fn compress_then_decompress_restores_the_file() {
    let (input, packed, output) = (temp("a.txt"), temp("a.gpcz"), temp("a.out"));
    std::fs::write(&input, sample()).unwrap();
    let s = |p: &PathBuf| p.to_str().unwrap().to_string();
    gpucomp(&[
        "compress",
        &s(&input),
        &s(&packed),
        "--chunk-size",
        "16K",
        "--checksum",
    ]);
    gpucomp(&["decompress", &s(&packed), &s(&output), "--verify"]);
    assert!(std::fs::read(&output).unwrap() == sample());
    assert!(std::fs::metadata(&packed).unwrap().len() < sample().len() as u64 / 2);
}

#[test]
fn greedy_encoder_output_decompresses_with_lz4_flex() {
    let (input, packed, output) = (temp("b.txt"), temp("b.gpcz"), temp("b.out"));
    std::fs::write(&input, sample()).unwrap();
    let s = |p: &PathBuf| p.to_str().unwrap().to_string();
    gpucomp(&["compress", &s(&input), &s(&packed), "--encoder", "greedy"]);
    gpucomp(&[
        "decompress",
        &s(&packed),
        &s(&output),
        "--decoder",
        "lz4-flex",
    ]);
    assert!(std::fs::read(&output).unwrap() == sample());
}

#[test]
fn decompress_range_writes_just_that_slice() {
    let (input, packed, output) = (temp("c.txt"), temp("c.gpcz"), temp("c.out"));
    std::fs::write(&input, sample()).unwrap();
    let s = |p: &PathBuf| p.to_str().unwrap().to_string();
    gpucomp(&["compress", &s(&input), &s(&packed), "--chunk-size", "4K"]);
    gpucomp(&[
        "decompress",
        &s(&packed),
        &s(&output),
        "--offset",
        "123456",
        "--length",
        "10000",
    ]);
    assert!(std::fs::read(&output).unwrap() == sample()[123_456..133_456]);
}

#[test]
fn info_on_a_file_summarises_the_container() {
    let (input, packed) = (temp("d.txt"), temp("d.gpcz"));
    std::fs::write(&input, sample()).unwrap();
    let s = |p: &PathBuf| p.to_str().unwrap().to_string();
    gpucomp(&["compress", &s(&input), &s(&packed), "--chunk-size", "64K"]);
    let out = String::from_utf8(gpucomp(&["info", &s(&packed)]).stdout).unwrap();
    for needle in [
        "codec:",
        "lz4",
        "chunk size:",
        "64 KiB",
        "chunks:",
        "5",
        "ratio:",
    ] {
        assert!(out.contains(needle), "missing {needle:?} in:\n{out}");
    }
}

fn has_gpu() -> bool {
    let ok = gpu::Context::new(&gpu::ContextOptions::default()).is_ok();
    if !ok {
        eprintln!("skipping GPU CLI test: no adapter");
    }
    ok
}

#[test]
fn gpu_decompress_restores_the_file_and_ranges() {
    if !has_gpu() {
        return;
    }
    let (input, packed) = (temp("e.txt"), temp("e.gpcz"));
    let (whole, part) = (temp("e.out"), temp("e.part"));
    std::fs::write(&input, sample()).unwrap();
    let s = |p: &PathBuf| p.to_str().unwrap().to_string();
    gpucomp(&[
        "compress",
        &s(&input),
        &s(&packed),
        "--chunk-size",
        "8K",
        "--checksum",
    ]);
    gpucomp(&["decompress", &s(&packed), &s(&whole), "--gpu", "--verify"]);
    assert!(std::fs::read(&whole).unwrap() == sample());
    gpucomp(&[
        "decompress",
        &s(&packed),
        &s(&part),
        "--gpu",
        "--offset",
        "77777",
        "--length",
        "4321",
    ]);
    assert!(std::fs::read(&part).unwrap() == sample()[77_777..82_098]);
}

#[test]
fn gpu_compress_matches_cpu_greedy_and_round_trips() {
    if !has_gpu() {
        return;
    }
    let (input, gpu_packed, cpu_packed, out) = (
        temp("f.txt"),
        temp("f.gpu.gpcz"),
        temp("f.cpu.gpcz"),
        temp("f.out"),
    );
    std::fs::write(&input, sample()).unwrap();
    let s = |p: &PathBuf| p.to_str().unwrap().to_string();
    gpucomp(&[
        "compress",
        &s(&input),
        &s(&gpu_packed),
        "--gpu",
        "--checksum",
    ]);
    gpucomp(&[
        "compress",
        &s(&input),
        &s(&cpu_packed),
        "--encoder",
        "greedy",
        "--checksum",
    ]);
    assert!(std::fs::read(&gpu_packed).unwrap() == std::fs::read(&cpu_packed).unwrap());
    gpucomp(&["decompress", &s(&gpu_packed), &s(&out), "--verify"]);
    assert!(std::fs::read(&out).unwrap() == sample());
}

#[test]
fn glz_codec_round_trips_on_the_cpu() {
    let (input, packed, out) = (temp("g.txt"), temp("g.gpcz"), temp("g.out"));
    std::fs::write(&input, sample()).unwrap();
    let s = |p: &PathBuf| p.to_str().unwrap().to_string();
    for groups in [None, Some("32")] {
        let mut args = vec![
            "compress".to_string(),
            s(&input),
            s(&packed),
            "--codec".into(),
            "glz".into(),
        ];
        if let Some(g) = groups {
            args.extend(["--independent-groups".into(), g.into()]);
        }
        gpucomp(&args.iter().map(String::as_str).collect::<Vec<_>>());
        let info = String::from_utf8(gpucomp(&["info", &s(&packed)]).stdout).unwrap();
        assert!(info.contains("codec:       glz"), "{info}");
        gpucomp(&["decompress", &s(&packed), &s(&out), "--verify"]);
        assert!(std::fs::read(&out).unwrap() == sample(), "{groups:?}");
    }
}

#[test]
fn gpu_glz_compress_matches_cpu_glz() {
    if !has_gpu() {
        return;
    }
    let (input, gpu_packed, cpu_packed) = (temp("h.txt"), temp("h.gpu.gpcz"), temp("h.cpu.gpcz"));
    std::fs::write(&input, sample()).unwrap();
    let s = |p: &PathBuf| p.to_str().unwrap().to_string();
    for groups in ["0", "16"] {
        let mut gpu_args = vec![
            "compress",
            &*s(&input),
            &*s(&gpu_packed),
            "--gpu",
            "--codec",
            "glz",
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
        let mut cpu_args = vec!["compress", &*s(&input), &*s(&cpu_packed), "--codec", "glz"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        if groups != "0" {
            for args in [&mut gpu_args, &mut cpu_args] {
                args.extend(["--independent-groups".to_string(), groups.to_string()]);
            }
        }
        gpucomp(&gpu_args.iter().map(String::as_str).collect::<Vec<_>>());
        gpucomp(&cpu_args.iter().map(String::as_str).collect::<Vec<_>>());
        assert!(
            std::fs::read(&gpu_packed).unwrap() == std::fs::read(&cpu_packed).unwrap(),
            "groups {groups}"
        );
    }
}
