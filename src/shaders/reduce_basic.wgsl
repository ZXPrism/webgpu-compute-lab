// adapted from https://google.github.io/tour-of-wgsl/variables/var-workgroup/

override SEGMENT_LENGTH: u32;

// Create zero-initialized workgroup shared data
var<workgroup> workgroup_data: array<u32, SEGMENT_LENGTH>;

@group(0) @binding(0) var<uniform> array_length: u32;
@group(0) @binding(1) var<storage, read> input_array: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_sum_per_segment: array<u32>;

// Our workgroup will execute SEGMENT_LENGTH invocations of the shader
@compute @workgroup_size(SEGMENT_LENGTH, 1, 1)
fn compute(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) segment_id : vec3<u32>
) {
    // Each invocation will populate the shared workgroup data from the input data
    if global_id.x < array_length { // NOTE: we can't simply return here, or it will cause inconsistency for the barrier at L23
        workgroup_data[local_id.x] = input_array[global_id.x];
    } else{
        workgroup_data[local_id.x] = 0; // init out-of-bounds data to ZERO
    }
    // Wait for each invocation to populate their region of local data
    workgroupBarrier();

    // Get the sum of the elements in the array
    // Input Data:    [0,  1,  2,  3,   4,  5,  6,  7]
    // Loop Pass 1:   [1,  5,  9,  13,  4,  5,  6,  7]
    // Loop Pass 2:   [6,  22, 9,  13,  4,  5,  6,  7]
    for (var current_size = SEGMENT_LENGTH >> 1; current_size >= 1; current_size >>= 1) {
        var sum: u32 = 0;

        if local_id.x < current_size {
            // Read current values from workgroup_data
            sum = workgroup_data[local_id.x * 2] + workgroup_data[local_id.x * 2 + 1];
        }
        // Wait until all invocations have finished reading from workgroup_data, and have calculated their respective sums
        workgroupBarrier();

        if local_id.x < current_size {
            workgroup_data[local_id.x] = sum;
        }

        // Wait for each invocation to finish one iteration of the loop, and to have finished writing to workgroup_data
        workgroupBarrier();
    }

    // Write the sum to the output
    if local_id.x == 0 {
        output_sum_per_segment[segment_id.x] = workgroup_data[0];
    }
}
