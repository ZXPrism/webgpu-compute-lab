override SEGMENT_LENGTH: u32;

@group(0) @binding(0) var<uniform> array_length : u32;
@group(0) @binding(1) var<storage, read> input_array : array<u32>;
@group(0) @binding(2) var<storage, read_write> prefix_sum : array<u32>;
@group(0) @binding(3) var<storage, read_write> segment_sum : array<u32>;

var<workgroup> workgroup_data: array<u32, SEGMENT_LENGTH>;

@compute
@workgroup_size(SEGMENT_LENGTH, 1, 1)
fn compute(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) segment_id : vec3<u32>
) {
    if global_id.x < array_length {
        workgroup_data[local_id.x] = input_array[global_id.x];
    } else {
        workgroup_data[local_id.x] = 0u;
    }

    workgroupBarrier();

    if local_id.x == 0u {
        for(var i = 1u; i < SEGMENT_LENGTH; i++) {
            workgroup_data[i] += workgroup_data[i - 1u];
        }
        segment_sum[segment_id.x] = workgroup_data[SEGMENT_LENGTH - 1u];
    }

    workgroupBarrier();

    prefix_sum[global_id.x] = workgroup_data[local_id.x];
}
