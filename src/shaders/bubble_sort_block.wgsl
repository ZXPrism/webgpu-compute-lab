override SEGMENT_LENGTH: u32;

@group(0) @binding(0) var<uniform> array_length : u32;
@group(0) @binding(1) var<storage, read> input_array : array<u32>;
@group(0) @binding(2) var<storage, read_write> sorted_array : array<u32>;

var<workgroup> workgroup_data: array<u32, SEGMENT_LENGTH>;

@compute
@workgroup_size(SEGMENT_LENGTH, 1, 1)
fn compute(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) segment_id: vec3<u32>
) {
    // here we don't require one segment to be a power of 2 for the sort algo, so just leave padding elements uninit-d
    if global_id.x < array_length {
        workgroup_data[local_id.x] = input_array[global_id.x];
    }

    workgroupBarrier();

    if local_id.x == 0u {
        let n = min(SEGMENT_LENGTH, array_length - segment_id.x * SEGMENT_LENGTH);

        for(var i = 0u; i < n; i++) {
            for(var j = n - 1u; j > i; j--) {
                if workgroup_data[j - 1u] > workgroup_data[j] {
                    let temp = workgroup_data[j];
                    workgroup_data[j] = workgroup_data[j - 1u];
                    workgroup_data[j - 1u] = temp;
                }
            }
        }
    }

    workgroupBarrier();

    if global_id.x < array_length {
        sorted_array[global_id.x] = workgroup_data[local_id.x];
    }
}
