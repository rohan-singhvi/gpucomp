# Decisions

Deviations from `plan.md` and non-obvious choices, newest last.

## M0 — wgpu 30.0.1 API notes
`wgpu::InstanceDescriptor` no longer implements `Default`; we build it from
`InstanceDescriptor::new_without_display_handle_from_env()` so the standard `WGPU_*`
environment variables still work, overriding `backends` (default: `Backends::PRIMARY`,
i.e. Metal/D3D12/Vulkan). Device limits start from WebGPU defaults (downlevel defaults
if the adapter can't meet them) and raise only `max_storage_buffer_binding_size`,
`max_buffer_size` and `max_compute_workgroup_storage_size` to the adapter's values, so
we never request a limit the adapter lacks. On Apple M4 Pro / Metal wgpu reports
`max_buffer_size` = 4 GiB − 1, which M7's batching must respect.

## M0 — License
The repository ships under the "MIT NON-AI" license already committed upstream; this
answers open question 4 in the plan.
