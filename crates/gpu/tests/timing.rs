//! Upload/readback helpers and GPU timestamp timing. Skipped without an adapter.

use gpu::{Context, ContextOptions, GpuError, GpuTimer, XorKernel};

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
fn upload_then_read_back_returns_the_same_bytes() {
    let Some(ctx) = context() else { return };
    let bytes: Vec<u8> = (0..4096u32).map(|i| (i * 7 + 3) as u8).collect();
    let buffer = ctx.upload(&bytes, gpu::wgpu::BufferUsages::COPY_SRC);
    assert_eq!(ctx.read_buffer(&buffer, bytes.len() as u64).unwrap(), bytes);
}

#[test]
fn device_has_timestamp_queries_exactly_when_adapter_does() {
    let Some(ctx) = context() else { return };
    let ts = gpu::wgpu::Features::TIMESTAMP_QUERY;
    assert_eq!(
        ctx.device.features().contains(ts),
        ctx.adapter_features().contains(ts)
    );
}

#[test]
fn timer_requires_timestamp_queries() {
    let Some(ctx) = context() else { return };
    let has_ts = ctx
        .device
        .features()
        .contains(gpu::wgpu::Features::TIMESTAMP_QUERY);
    // The timer may also be unavailable *with* the feature, if the probe pass
    // finds the driver writes no samples.
    assert!(GpuTimer::new(&ctx).is_none() || has_ts);
}

#[test]
fn timer_measures_a_nonzero_kernel_duration() {
    let Some(ctx) = context() else { return };
    let Some(timer) = GpuTimer::new(&ctx) else {
        eprintln!("skipping: no usable GPU timestamps on this adapter");
        return;
    };
    let words = 1 << 20;
    let data = ctx.upload(
        &vec![0u8; words * 4],
        gpu::wgpu::BufferUsages::STORAGE | gpu::wgpu::BufferUsages::COPY_SRC,
    );
    XorKernel::new(&ctx).dispatch(&ctx, &data, 0xFF, Some(&timer));
    let elapsed = timer.read(&ctx).unwrap();
    assert!(elapsed.as_nanos() > 0, "elapsed = {elapsed:?}");
    assert!(elapsed.as_secs() < 1, "elapsed = {elapsed:?}");
}

#[test]
fn xor_kernel_dispatch_modifies_buffer_in_place() {
    let Some(ctx) = context() else { return };
    let data = ctx.upload(
        bytemuck::cast_slice(&[1u32, 2, 3]),
        gpu::wgpu::BufferUsages::STORAGE | gpu::wgpu::BufferUsages::COPY_SRC,
    );
    XorKernel::new(&ctx).dispatch(&ctx, &data, 1, None);
    let out: Vec<u32> = bytemuck::pod_collect_to_vec(&ctx.read_buffer(&data, 12).unwrap());
    assert_eq!(out, [0, 3, 2]);
}
