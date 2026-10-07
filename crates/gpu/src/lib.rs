//! GPU compression and decompression through wgpu compute shaders.

mod context;
mod smoke;

pub use context::{device_limits, report, AdapterSummary, Context, ContextOptions, GpuError};
pub use smoke::xor_u32_cpu;
pub use wgpu;
