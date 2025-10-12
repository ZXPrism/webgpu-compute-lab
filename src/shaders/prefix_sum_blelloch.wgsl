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

    // pass 1: upsweep
    {
        var d = 1u;
        var offset = 1u << d;
        while offset <= SEGMENT_LENGTH {
            let idx = ((local_id.x + 1u) << d) - 1u;

            if idx < SEGMENT_LENGTH {
                workgroup_data[idx] += workgroup_data[idx - (offset >> 1u)];
            }

            d++;
            offset <<= 1u;

            workgroupBarrier();
        }
    }

    if local_id.x == SEGMENT_LENGTH - 1u {
        segment_sum[segment_id.x] = workgroup_data[SEGMENT_LENGTH - 1u];
        workgroup_data[SEGMENT_LENGTH - 1u] = 0u;
    }

    workgroupBarrier();

    // pass 2: downsweep
    {
        var d = SEGMENT_LENGTH;
        while d >= 2u {
            let curr_idx = ((local_id.x + 1u) * d) - 1u;
            if curr_idx < SEGMENT_LENGTH {
                let t = workgroup_data[curr_idx - (d >> 1u)];
                workgroup_data[curr_idx - (d >> 1u)] = workgroup_data[curr_idx];
                workgroup_data[curr_idx] += t;
            }

            d >>= 1u;

            workgroupBarrier();
        }
    }

    // note that blelloch is an exclusive scan, so when storing result we need to calculate the final element manually
    // if local_id.x > 0u {
    //     prefix_sum[global_id.x - 1u] = workgroup_data[local_id.x];
    // } else {
    //     let final_idx = global_id.x + SEGMENT_LENGTH - 1u;
    //     prefix_sum[final_idx] = workgroup_data[SEGMENT_LENGTH - 1u] + input_array[final_idx];
    // }

    prefix_sum[global_id.x] = workgroup_data[local_id.x]; // exclusive version
}
