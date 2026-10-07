use clap::{Parser, Subcommand, ValueEnum};

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
    /// Print the selected GPU adapter, backend and device limits.
    Info {
        #[arg(long, value_enum)]
        backend: Option<BackendArg>,
    },
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
        Command::Info { backend } => {
            let ctx = gpu::Context::new(&gpu::ContextOptions {
                backends: backend.map(BackendArg::backends),
            })?;
            print!("{}", ctx.report());
        }
    }
    Ok(())
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
                backend: Some(BackendArg::Dx12)
            }
        ));
    }
}
