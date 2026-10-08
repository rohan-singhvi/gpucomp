//! GPU compression and decompression through wgpu compute shaders.

mod context;
pub mod decode;
mod dispatch;
pub mod encode;
pub mod filter;
mod smoke;
mod timer;

pub use context::{
    device_features, device_limits, report, AdapterSummary, Context, ContextOptions, GpuError,
};
pub use dispatch::dispatch_grid;
pub use smoke::{xor_u32_cpu, XorKernel};
pub use timer::GpuTimer;
pub use wgpu;
