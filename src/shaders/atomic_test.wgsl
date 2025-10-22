@group(0) @binding(0) var<storage, read_write> output : array<u32, 64>;

var<workgroup> local_histogram : array<atomic<u32>, 2>;

@compute
@workgroup_size(64, 1, 1)
fn compute(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    if local_id.x <= 2 {
        atomicStore(&local_histogram[local_id.x], 0u);
    }
    workgroupBarrier();

    let local_offset = atomicAdd(&local_histogram[local_id.x & 1u], 1u);
    workgroupBarrier();

    output[global_id.x] = local_offset;
}
