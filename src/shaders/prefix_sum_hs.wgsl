override SEGMENT_LENGTH: u32;

@group(0) @binding(0) var<uniform> array_length : u32;
@group(0) @binding(1) var<storage, read> input_array : array<u32>;
@group(0) @binding(2) var<storage, read_write> prefix_sum : array<u32>;
@group(0) @binding(3) var<storage, read_write> segment_sum : array<u32>;

var<workgroup> workgroup_data: array<u32, SEGMENT_LENGTH * 2u>;

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

    var front = 0u;
    var back = 1u;
    for (var d = 1u; d < SEGMENT_LENGTH; d <<= 1u) {
        let front_base_addr = front * SEGMENT_LENGTH;
        let back_base_addr = back * SEGMENT_LENGTH;

        if local_id.x >= d {
            workgroup_data[back_base_addr + local_id.x] =
                workgroup_data[front_base_addr + local_id.x] + workgroup_data[front_base_addr + local_id.x - d];
        } else {
            workgroup_data[back_base_addr + local_id.x] = workgroup_data[front_base_addr + local_id.x];
        }

        front ^= 1u;
        back ^= 1u;

        workgroupBarrier();
    }

    let base_addr = front * SEGMENT_LENGTH;
    prefix_sum[global_id.x] = workgroup_data[base_addr + local_id.x];

    if local_id.x == SEGMENT_LENGTH - 1 {
        segment_sum[segment_id.x] = workgroup_data[base_addr + SEGMENT_LENGTH - 1];
    }
}
