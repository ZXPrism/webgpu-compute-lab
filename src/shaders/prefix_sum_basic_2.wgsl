override SEGMENT_LENGTH: u32;

@group(0) @binding(0) var<uniform> array_length : u32;
@group(0) @binding(1) var<storage, read> input_array : array<u32>;
@group(0) @binding(2) var<storage, read_write> prefix_sum : array<u32>;
@group(0) @binding(3) var<storage, read_write> segment_sum : array<u32>;

@compute
@workgroup_size(1, 1, 1)
fn compute(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) segment_id : vec3<u32>
) {
    let base_addr = segment_id.x * SEGMENT_LENGTH;
    prefix_sum[base_addr] = input_array[base_addr];
    for(var i = 1u; i < SEGMENT_LENGTH; i++) {
        let curr_addr = base_addr + i;
        prefix_sum[curr_addr] = prefix_sum[curr_addr - 1u] + input_array[curr_addr];
    }
    segment_sum[segment_id.x] = prefix_sum[base_addr + SEGMENT_LENGTH - 1u];
}
