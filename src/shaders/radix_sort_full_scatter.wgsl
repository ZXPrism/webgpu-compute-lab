override SEGMENT_LENGTH: u32;
override RADIX_BITS: u32; // 4 by default
override RIGHT_SHIFT_BITS: u32; // set by outer codes, starting from zero, each time += RADIX_BITS

@group(0) @binding(0) var<uniform> array_length : u32;
@group(0) @binding(1) var<storage, read> input_array : array<u32>;
@group(0) @binding(2) var<storage, read> slot_size_prefix_sum : array<u32>; // size = (1u << RADIX_BITS) * WG_CNT
@group(0) @binding(3) var<storage, read> local_prefix_sum : array<u32>; // size = array_length
@group(0) @binding(4) var<storage, read_write> sorted_array : array<u32>;

@compute
@workgroup_size(SEGMENT_LENGTH, 1, 1)
fn compute(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) segment_id: vec3<u32>
) {
    if global_id.x >= array_length {
        return;
    }

    let wg_cnt: u32 = u32(ceil(f32(array_length) / f32(SEGMENT_LENGTH)));
    let value = input_array[local_id.x];
    let local_prefix = local_prefix_sum[global_id.x];
    let mask = (1u << RADIX_BITS) - 1u;
    let curr_slot = (value >> RIGHT_SHIFT_BITS) & mask;
    let scatter_pos = slot_size_prefix_sum[curr_slot * wg_cnt + segment_id.x] + local_prefix;

    sorted_array[scatter_pos] = value;
}
