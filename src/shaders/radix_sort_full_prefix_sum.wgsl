override SEGMENT_LENGTH: u32;
override RADIX_BITS: u32; // 4 by default
override RIGHT_SHIFT_BITS: u32; // set by outer codes, starting from zero, each time += RADIX_BITS

@group(0) @binding(0) var<uniform> array_length : u32;
@group(0) @binding(1) var<storage, read> input_array : array<u32>;
@group(0) @binding(2) var<storage, read_write> slot_size : array<u32>; // size = (1u << RADIX_BITS) * WG_CNT
@group(0) @binding(3) var<storage, read_write> local_prefix_sum : array<u32>; // size = array_length
@group(0) @binding(4) var<uniform> wg_cnt : u32;

var<workgroup> input_data: array<u32, SEGMENT_LENGTH>;
var<workgroup> psum : array<u32, SEGMENT_LENGTH>;

@compute
@workgroup_size(SEGMENT_LENGTH, 1, 1)
fn compute(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) segment_id: vec3<u32>
) {
    if global_id.x < array_length {
        input_data[local_id.x] = input_array[global_id.x];
    } else {
        input_data[local_id.x] = 0xffffffffu; // set to a special value so that they are always at last
    }

    workgroupBarrier();

    let psum_way_cnt = 1u << RADIX_BITS;
    let mask = psum_way_cnt - 1u;
    let curr_slot = (input_data[local_id.x] >> RIGHT_SHIFT_BITS) & mask;

    // 1. (1 << RADIX_BITS)-way prefix sum
    var psum_curr = 0u;
    for (var slot = 0u; slot < psum_way_cnt; slot++) {
        // "split" to psum buffer
        // why split first? because in prefix sums, we will repeated access elements, but they need to be computed first
        // and we just want to compute them once
        psum[local_id.x] = select(0u, 1u, ((input_data[local_id.x] >> RIGHT_SHIFT_BITS) & mask) == slot);

        workgroupBarrier();

        // pass 1: upsweep
        {
            var d = 1u;
            var offset = 1u << d;
            while offset <= SEGMENT_LENGTH {
                let idx = ((local_id.x + 1u) << d) - 1u;

                if idx < SEGMENT_LENGTH {
                    psum[idx] += psum[idx - (offset >> 1u)];
                }

                d++;
                offset <<= 1u;

                workgroupBarrier();
            }
        }

        if local_id.x == SEGMENT_LENGTH - 1u {
            psum[SEGMENT_LENGTH - 1u] = 0u;
        }

        workgroupBarrier();

        // pass 2: downsweep
        {
            var d = SEGMENT_LENGTH;
            while d >= 2u {
                let new_d = d >> 1u;
                let curr_idx = ((local_id.x + 1u) * d) - 1u;
                if curr_idx < SEGMENT_LENGTH {
                    let t = psum[curr_idx - new_d];
                    psum[curr_idx - new_d] = psum[curr_idx];
                    psum[curr_idx] += t;
                }

                d = new_d;

                workgroupBarrier();
            }
        }

        if local_id.x == SEGMENT_LENGTH - 1 {
            slot_size[slot * wg_cnt + segment_id.x] = psum[SEGMENT_LENGTH - 1];
        }

        if slot == curr_slot {
            psum_curr = psum[local_id.x];
        }

        workgroupBarrier();
    }

    // set final psum -- interleaving psums
    if global_id.x < array_length {
        local_prefix_sum[global_id.x] = psum_curr;
    }

    // add the final element to its slot
    if local_id.x == SEGMENT_LENGTH - 1 {
        slot_size[curr_slot * wg_cnt + segment_id.x]++;
    }
}
