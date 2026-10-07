//! `compress`, `decompress` and `info <file>`.

use std::path::PathBuf;

use clap::ValueEnum;
use cpu::container::{CompressOptions, DecompressOptions};
use format::{Codec, Filter, Index};

#[derive(clap::Args, Debug)]
pub struct CompressArgs {
    pub input: PathBuf,
    pub output: PathBuf,
    #[arg(long, value_enum, default_value_t = CodecArg::Lz4)]
    pub codec: CodecArg,
    /// Uncompressed chunk size: a power of two from 4K to 1M (suffixes K, M).
    #[arg(long, default_value = "64K", value_parser = parse_size)]
    pub chunk_size: u32,
    /// Store a checksum of every chunk.
    #[arg(long)]
    pub checksum: bool,
    /// CPU block encoder for LZ4: `lz4-flex` (baseline) or `greedy` (the GPU
    /// encoder's twin). GLZ always uses its own greedy encoder.
    #[arg(long, value_enum, default_value_t = EncoderArg::Lz4Flex)]
    pub encoder: EncoderArg,
    /// GLZ only: no match may copy from another match's output within each
    /// group of this many sequences (dependency elimination).
    #[arg(long)]
    pub independent_groups: Option<u32>,
    /// Compress on the GPU (LZ4 or GLZ). Output is identical to the CPU
    /// twin: `--encoder greedy` for LZ4, the GLZ encoder for GLZ.
    #[arg(long, conflicts_with = "encoder")]
    pub gpu: bool,
    /// GPU backend (with --gpu).
    #[arg(long, value_enum, requires = "gpu")]
    pub backend: Option<crate::BackendArg>,
}

#[derive(clap::Args, Debug)]
pub struct DecompressArgs {
    pub input: PathBuf,
    pub output: PathBuf,
    /// Start of the range to extract, in bytes of the original.
    #[arg(long, requires = "length")]
    pub offset: Option<u64>,
    /// Length of the range to extract.
    #[arg(long, requires = "offset")]
    pub length: Option<u64>,
    /// Check per-chunk checksums (when the file has them).
    #[arg(long)]
    pub verify: bool,
    /// CPU block decoder (ignored with --gpu).
    #[arg(long, value_enum, default_value_t = DecoderArg::HandWritten)]
    pub decoder: DecoderArg,
    /// Decompress on the GPU.
    #[arg(long)]
    pub gpu: bool,
    /// GPU backend (with --gpu).
    #[arg(long, value_enum, requires = "gpu")]
    pub backend: Option<crate::BackendArg>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum CodecArg {
    Stored,
    Lz4,
    /// GPU-friendly LZ77 with separate fixed-width streams.
    Glz,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum EncoderArg {
    Lz4Flex,
    Greedy,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum DecoderArg {
    HandWritten,
    Lz4Flex,
}

/// Parses `4096`, `64K` or `1M` (binary units).
pub fn parse_size(text: &str) -> Result<u32, String> {
    let (digits, shift) = match text.chars().last() {
        Some('k' | 'K') => (&text[..text.len() - 1], 10),
        Some('m' | 'M') => (&text[..text.len() - 1], 20),
        _ => (text, 0),
    };
    let n: u64 = digits
        .parse()
        .map_err(|_| format!("invalid size {text:?}"))?;
    u32::try_from(n << shift).map_err(|_| format!("size {text:?} is too large"))
}

/// Human-readable summary of a container for `gpucomp info <file>`.
pub fn describe(index: &Index, file_len: u64) -> String {
    use bench::report::human_bytes;
    let h = &index.header;
    let stored = index.chunks.iter().filter(|c| c.stored).count();
    let mut filters: Vec<(String, usize)> = Vec::new();
    for c in &index.chunks {
        let name = match c.filter {
            Filter::None => "none".to_string(),
            Filter::Shuffle { width } => format!("shuffle-{width}"),
            Filter::Delta { width } => format!("delta-{width}"),
        };
        match filters.iter_mut().find(|(n, _)| *n == name) {
            Some((_, count)) => *count += 1,
            None => filters.push((name, 1)),
        }
    }
    let codec = match h.codec {
        Codec::Stored => "stored",
        Codec::Lz4 => "lz4",
        Codec::Glz => "glz",
    };
    let ratio = if file_len == 0 {
        0.0
    } else {
        h.total_size as f64 / file_len as f64
    };
    let filters = filters
        .iter()
        .map(|(n, c)| format!("{n} {c}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "codec:       {codec}\n\
         level:       {}\n\
         chunk size:  {}\n\
         chunks:      {} ({stored} stored)\n\
         original:    {}\n\
         compressed:  {}\n\
         ratio:       {ratio:.2}\n\
         checksums:   {}\n\
         filters:     {filters}\n",
        h.level,
        human_bytes(u64::from(h.chunk_size)),
        h.chunk_count,
        human_bytes(h.total_size),
        human_bytes(file_len),
        if h.checksums { "yes" } else { "no" },
    )
}

pub fn compress(args: &CompressArgs) -> anyhow::Result<()> {
    let input = std::fs::read(&args.input)?;
    if args.gpu {
        let ctx = gpu::Context::new(&gpu::ContextOptions {
            backends: args.backend.map(crate::BackendArg::backends),
        })?;
        let encoder = gpu::encode::GpuEncoder::new(&ctx, Default::default())?;
        let options = gpu::encode::GpuCompressOptions {
            codec: match args.codec {
                CodecArg::Lz4 => Codec::Lz4,
                CodecArg::Glz => Codec::Glz,
                CodecArg::Stored => anyhow::bail!("--gpu compresses with lz4 or glz"),
            },
            chunk_size: args.chunk_size,
            checksums: args.checksum,
            level: 1,
            independent_groups: args.independent_groups,
        };
        std::fs::write(&args.output, encoder.compress(&ctx, &input, &options)?)?;
        return Ok(());
    }
    let options = CompressOptions {
        codec: match args.codec {
            CodecArg::Stored => Codec::Stored,
            CodecArg::Lz4 => Codec::Lz4,
            CodecArg::Glz => Codec::Glz,
        },
        chunk_size: args.chunk_size,
        encoder: match (args.codec, args.encoder) {
            (CodecArg::Glz, _) => cpu::container::Encoder::Glz(cpu::glz::GlzParams {
                independent_groups: args.independent_groups,
                ..Default::default()
            }),
            (_, EncoderArg::Lz4Flex) => cpu::container::Encoder::Lz4Flex,
            (_, EncoderArg::Greedy) => cpu::container::Encoder::Greedy(Default::default()),
        },
        checksums: args.checksum,
        level: 1,
    };
    std::fs::write(&args.output, cpu::container::compress(&input, &options)?)?;
    Ok(())
}

pub fn decompress(args: &DecompressArgs) -> anyhow::Result<()> {
    let options = DecompressOptions {
        decoder: match args.decoder {
            DecoderArg::HandWritten => cpu::container::Decoder::HandWritten,
            DecoderArg::Lz4Flex => cpu::container::Decoder::Lz4Flex,
        },
        verify: args.verify,
    };
    if args.gpu {
        let ctx = gpu::Context::new(&gpu::ContextOptions {
            backends: args.backend.map(crate::BackendArg::backends),
        })?;
        let decoder = gpu::decode::GpuDecoder::new(&ctx);
        let output = match (args.offset, args.length) {
            (Some(offset), Some(length)) => {
                let mut file = std::io::BufReader::new(std::fs::File::open(&args.input)?);
                decoder.decompress_range(&ctx, &mut file, offset, length, args.verify)?
            }
            _ => decoder.decompress(&ctx, &std::fs::read(&args.input)?, args.verify)?,
        };
        std::fs::write(&args.output, output)?;
        return Ok(());
    }
    let output = match (args.offset, args.length) {
        (Some(offset), Some(length)) => {
            let mut file = std::io::BufReader::new(std::fs::File::open(&args.input)?);
            cpu::container::decompress_range(&mut file, offset, length, &options)?
        }
        _ => cpu::container::decompress(&std::fs::read(&args.input)?, &options)?,
    };
    std::fs::write(&args.output, output)?;
    Ok(())
}

pub fn info(path: &std::path::Path) -> anyhow::Result<()> {
    let mut file = std::fs::File::open(path)?;
    let index = cpu::container::read_index(&mut file)?;
    print!("{}", describe(&index, file.metadata()?.len()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use format::{ChunkEntry, Header};

    use super::*;

    #[test]
    fn sizes_accept_plain_bytes_and_binary_suffixes() {
        assert_eq!(parse_size("4096"), Ok(4096));
        assert_eq!(parse_size("64K"), Ok(65_536));
        assert_eq!(parse_size("64k"), Ok(65_536));
        assert_eq!(parse_size("1M"), Ok(1 << 20));
    }

    #[test]
    fn sizes_reject_garbage() {
        for bad in ["", "K", "12Q", "-4K", "99999999999"] {
            assert!(parse_size(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn describe_reports_codec_sizes_ratio_and_chunk_kinds() {
        let entry = |stored, filter| ChunkEntry {
            comp_offset: 0,
            comp_size: 1000,
            stored,
            uncomp_size: 4096,
            checksum: 0,
            filter,
        };
        let index = Index {
            header: Header {
                codec: Codec::Lz4,
                chunk_size: 4096,
                chunk_count: 3,
                total_size: 12_288,
                checksums: true,
                level: 1,
            },
            chunks: vec![
                entry(false, Filter::None),
                entry(true, Filter::None),
                entry(false, Filter::Delta { width: 4 }),
            ],
        };
        let text = describe(&index, 6144);
        for needle in [
            "codec:       lz4",
            "chunk size:  4 KiB",
            "chunks:      3 (1 stored)",
            "original:    12 KiB",
            "compressed:  6 KiB",
            "ratio:       2.00",
            "checksums:   yes",
            "filters:     none 2, delta-4 1",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
        }
    }
}
