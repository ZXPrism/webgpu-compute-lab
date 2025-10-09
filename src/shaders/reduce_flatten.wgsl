override SEGMENT_LENGTH: u32;

var<workgroup> workgroup_data: array<u32, SEGMENT_LENGTH>;

@group(0) @binding(0) var<uniform> array_length: u32;
@group(0) @binding(1) var<storage, read> input_array: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_sum_per_segment: array<u32>;

@compute @workgroup_size(SEGMENT_LENGTH, 1, 1)
fn compute(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) segment_id : vec3<u32>
) {
    if global_id.x < array_length {
        workgroup_data[local_id.x] = input_array[global_id.x];
    } else{
        workgroup_data[local_id.x] = 0;
    }
    workgroupBarrier();

    var offset = SEGMENT_LENGTH >> 1u;
    while (offset > 0u) {
        if (local_id.x < offset) {
            workgroup_data[local_id.x] += workgroup_data[local_id.x + offset];
        }
        workgroupBarrier();
        offset >>= 1u;
    }

    if (local_id.x == 0u) {
        output_sum_per_segment[segment_id.x] = workgroup_data[0u];
    }
}
