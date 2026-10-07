use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

mod files;

#[derive(Parser, Debug)]
#[command(
    name = "gpucomp",
    version,
    about = "GPU compression and decompression via wgpu"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Compress a file into the chunked .gpcz container.
    Compress(files::CompressArgs),
    /// Decompress a .gpcz file, or just a byte range of it.
    Decompress(files::DecompressArgs),
    /// With a file: summarise the container. Without: print the GPU adapter, backend and limits.
    Info {
        file: Option<PathBuf>,
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
    /// Run the benchmark suite; optionally record it and regenerate BENCHMARKS.md.
    Bench(BenchArgs),
}

#[derive(clap::Args, Debug)]
struct BenchArgs {
    /// Bytes per iteration, in MiB.
    #[arg(long, default_value_t = 256)]
    size_mib: usize,
    /// Measured iterations (the median is reported).
    #[arg(long, default_value_t = 10)]
    runs: usize,
    /// Discarded warm-up iterations.
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    /// Label for this run in the history, e.g. a milestone ("M1") or change ("M5-atomicOr").
    #[arg(long, default_value = "dev")]
    label: String,
    /// What changed since the previous recorded run (shown in the report's history).
    #[arg(long, default_value = "")]
    note: String,
    /// Save the run as JSON under --results-dir.
    #[arg(long)]
    record: bool,
    /// Regenerate --report-path from every recorded run.
    #[arg(long)]
    report: bool,
    /// Skip measuring; only regenerate the report.
    #[arg(long, requires = "report")]
    report_only: bool,
    /// Measure CPU baselines only.
    #[arg(long)]
    cpu_only: bool,
    /// Also benchmark every file in this directory, plus their concatenation
    /// (e.g. testdata/corpus/silesia after scripts/fetch_corpus).
    #[arg(long)]
    corpus: Option<PathBuf>,
    #[arg(long, value_enum)]
    backend: Option<BackendArg>,
    #[arg(long, default_value = "bench/results")]
    results_dir: PathBuf,
    #[arg(long, default_value = "BENCHMARKS.md")]
    report_path: PathBuf,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum BackendArg {
    Metal,
    Dx12,
    Vulkan,
}

impl BackendArg {
    fn backends(self) -> gpu::wgpu::Backends {
        match self {
            BackendArg::Metal => gpu::wgpu::Backends::METAL,
            BackendArg::Dx12 => gpu::wgpu::Backends::DX12,
            BackendArg::Vulkan => gpu::wgpu::Backends::VULKAN,
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    match Cli::parse().command {
        Command::Compress(args) => files::compress(&args)?,
        Command::Decompress(args) => files::decompress(&args)?,
        Command::Info {
            file: Some(file), ..
        } => files::info(&file)?,
        Command::Info {
            file: None,
            backend,
        } => {
            let ctx = gpu::Context::new(&gpu::ContextOptions {
                backends: backend.map(BackendArg::backends),
            })?;
            print!("{}", ctx.report());
        }
        Command::Bench(args) => bench_command(&args)?,
    }
    Ok(())
}

fn bench_command(args: &BenchArgs) -> anyhow::Result<()> {
    if !args.report_only {
        let ctx = if args.cpu_only {
            None
        } else {
            Some(gpu::Context::new(&gpu::ContextOptions {
                backends: args.backend.map(BackendArg::backends),
            })?)
        };
        let cfg = bench::suite::SuiteConfig {
            bytes: args.size_mib << 20,
            warmup: args.warmup,
            runs: args.runs,
        };
        let (adapter, backend) = ctx.as_ref().map_or(("cpu-only".into(), "-".into()), |c| {
            (c.adapter().name.clone(), c.adapter().backend.to_string())
        });
        let run = bench::record::Run {
            date: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            commit: git_commit(),
            label: args.label.clone(),
            note: args.note.clone(),
            os: std::env::consts::OS.into(),
            adapter,
            backend,
            measurements: bench::suite::platform(ctx.as_ref(), &cfg)?,
        };
        let mut run = run;
        let mut inputs = bench::suite::synthetic_inputs(cfg.bytes);
        if let Some(dir) = &args.corpus {
            inputs.extend(corpus_inputs(dir)?);
        }
        run.measurements
            .extend(bench::suite::codecs(&inputs, &cfg)?);
        if let Some(ctx) = &ctx {
            run.measurements
                .extend(bench::suite::gpu_decode(ctx, &inputs, &cfg)?);
            run.measurements
                .extend(bench::suite::gpu_encode(ctx, &inputs, &cfg)?);
        }
        print!("{}", bench::report::render(std::slice::from_ref(&run)));
        if args.record {
            let path = bench::store::write_run(&args.results_dir, &run)?;
            eprintln!("recorded {}", path.display());
        }
    }
    if args.report {
        let runs = bench::store::load_runs(&args.results_dir)?;
        std::fs::write(&args.report_path, bench::report::render(&runs))?;
        eprintln!(
            "wrote {} from {} run(s)",
            args.report_path.display(),
            runs.len()
        );
    }
    Ok(())
}

/// Every regular file in `dir` (sorted by name) as `<dir name>/<file>`, plus
/// their concatenation as `<dir name>/all`.
fn corpus_inputs(dir: &std::path::Path) -> anyhow::Result<Vec<(String, Vec<u8>)>> {
    let corpus = dir
        .file_name()
        .map_or("corpus".into(), |n| n.to_string_lossy().into_owned());
    let mut paths: Vec<_> = std::fs::read_dir(dir)?
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<_, _>>()?;
    paths.retain(|p| p.is_file());
    paths.sort();
    anyhow::ensure!(!paths.is_empty(), "no files in {}", dir.display());
    let mut inputs = Vec::new();
    for path in paths {
        let name = path.file_name().unwrap().to_string_lossy();
        inputs.push((format!("{corpus}/{name}"), std::fs::read(&path)?));
    }
    let all = inputs.iter().flat_map(|(_, d)| d.iter().copied()).collect();
    inputs.push((format!("{corpus}/all"), all));
    Ok(inputs)
}

/// Short HEAD commit, suffixed `-dirty` when the working tree has changes.
fn git_commit() -> String {
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let Some(head) = git(&["rev-parse", "--short", "HEAD"]) else {
        return "unknown".into();
    };
    let dirty = git(&["status", "--porcelain"]).is_some_and(|s| !s.is_empty());
    if dirty {
        format!("{head}-dirty")
    } else {
        head
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu::wgpu::Backends;

    #[test]
    fn backend_args_map_to_single_wgpu_backends() {
        assert_eq!(BackendArg::Metal.backends(), Backends::METAL);
        assert_eq!(BackendArg::Dx12.backends(), Backends::DX12);
        assert_eq!(BackendArg::Vulkan.backends(), Backends::VULKAN);
    }

    #[test]
    fn info_accepts_a_backend() {
        let cli = Cli::try_parse_from(["gpucomp", "info", "--backend", "dx12"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Info {
                file: None,
                backend: Some(BackendArg::Dx12)
            }
        ));
    }

    fn bench_args(extra: &[&str]) -> BenchArgs {
        let argv = ["gpucomp", "bench"].iter().chain(extra).copied();
        match Cli::try_parse_from(argv).unwrap().command {
            Command::Bench(args) => args,
            other => panic!("parsed as {other:?}"),
        }
    }

    #[test]
    fn bench_defaults_measure_256_mib_without_recording() {
        let args = bench_args(&[]);
        assert_eq!(args.size_mib, 256);
        assert_eq!((args.warmup, args.runs), (2, 10));
        assert!(!args.record && !args.report && !args.report_only);
        assert_eq!(args.results_dir, PathBuf::from("bench/results"));
        assert_eq!(args.report_path, PathBuf::from("BENCHMARKS.md"));
    }

    #[test]
    fn corpus_inputs_are_sorted_files_plus_their_concatenation() {
        let dir = std::env::temp_dir().join(format!("gpucomp-corpus-{}", std::process::id()));
        let corpus = dir.join("mini");
        std::fs::create_dir_all(corpus.join("subdir")).unwrap();
        std::fs::write(corpus.join("b.txt"), b"bbb").unwrap();
        std::fs::write(corpus.join("a.txt"), b"aa").unwrap();
        let inputs = corpus_inputs(&corpus).unwrap();
        let summary: Vec<_> = inputs
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        assert_eq!(
            summary,
            [
                ("mini/a.txt", &b"aa"[..]),
                ("mini/b.txt", b"bbb"),
                ("mini/all", b"aabbb")
            ]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn bench_report_only_requires_report() {
        let argv = ["gpucomp", "bench", "--report-only"];
        assert!(Cli::try_parse_from(argv).is_err());
        assert!(bench_args(&["--report-only", "--report"]).report_only);
    }
}
