//! M0 GPU smoke tests. Skipped (not failed) when no adapter is available.

use gpu::{Context, ContextOptions, GpuError};

fn context() -> Option<Context> {
    match Context::new(&ContextOptions::default()) {
        Ok(ctx) => Some(ctx),
        Err(GpuError::NoAdapter(e)) => {
            eprintln!("skipping GPU test: no adapter ({e})");
            None
        }
        Err(e) => panic!("GPU context creation failed: {e}"),
    }
}

#[test]
fn gpu_xor_matches_cpu_xor() {
    let Some(ctx) = context() else { return };
    // Not a multiple of the workgroup size, so the bounds check is exercised.
    let input: Vec<u32> = (0..10_007u32)
        .map(|i| i.wrapping_mul(2_654_435_761))
        .collect();
    let key = 0xDEAD_BEEF;
    assert_eq!(
        ctx.xor_u32(&input, key).unwrap(),
        gpu::xor_u32_cpu(&input, key)
    );
}

#[test]
fn gpu_xor_of_empty_input_is_empty() {
    let Some(ctx) = context() else { return };
    assert_eq!(ctx.xor_u32(&[], 7).unwrap(), Vec::<u32>::new());
}

#[test]
fn device_gets_adapters_storage_binding_size() {
    let Some(ctx) = context() else { return };
    assert_eq!(
        ctx.device_limits().max_storage_buffer_binding_size,
        ctx.adapter_limits().max_storage_buffer_binding_size
    );
}
