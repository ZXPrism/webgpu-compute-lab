// a slightly modified version of reduce_basic.wgsl

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
        workgroup_data[local_id.x] = 0u;
    }
    workgroupBarrier();

    // for (var current_size = 2u; current_size <= SEGMENT_LENGTH; current_size <<= 1) {
    //     if((local_id.x + 1) % current_size == 0) {
    //         workgroup_data[local_id.x] += workgroup_data[local_id.x - (current_size >> 1)];
    //     }
    //     workgroupBarrier();
    // }

    // the impl above has 2 problems
    // 1. perf: uses an expensive modulo operation
    // 2. style: literals lack suffix

    // optimized version:
    var d = 1u;
    var offset = 1u << d;
    while offset <= SEGMENT_LENGTH {
        let idx = ((local_id.x + 1u) << d) - 1u;

        // don't mis-write as `idx < array_length`! `array_length` is only useful when populating the shared memory
        // here we focus on each workgroup i.e. segment being processed
        if idx < SEGMENT_LENGTH {
            workgroup_data[idx] += workgroup_data[idx - (offset >> 1u)];
        }

        workgroupBarrier();

        d++;
        offset <<= 1u;
    }

    if local_id.x == SEGMENT_LENGTH - 1u {
        output_sum_per_segment[segment_id.x] = workgroup_data[SEGMENT_LENGTH - 1u];
    }
}
