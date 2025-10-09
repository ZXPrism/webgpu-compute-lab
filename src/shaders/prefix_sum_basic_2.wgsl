// pre_hs = copy.copy(arr) + [0] * N
// front = 0
// back = 1
// for d in range(1, log2n + 1):
//     prev_step = 2 ** (d - 1)
//     for k in range(N):  # in parallel
//         if k < prev_step:
//             pre_hs[back * N + k] = pre_hs[front * N + k]
//         else:
//             pre_hs[back * N + k] = (
//                 pre_hs[front * N + k] + pre_hs[front * N + k - prev_step]
//             )
//     front, back = back, front
// print(
//     f"prefix sum (Hillis & Steele, double-buffered, concatenated): {pre_hs[:N] if front == 0 else pre_hs[N:]}"
// )

// implementation notes:
// 1. two pass approach (applicable for any prefix sum algorithms)
// 2. divide input array input segments, one workgroup -> one segment
// 3. pass 1: perform prefix sum kernel
// 3.1 load segment data into the shared memory of current workgroup (remember to allocate double space for double buffering)
// 3.2 add workgroup barrier to ensure all data is loaded for current workgroup (note: sync between workgroups are not required
// Q: how it works in for-loops?
// 3.3 for each layer, add workgroup barrier to ensure every thread within one workgroup is working on the same layer
// 3.4 perform additions for current layer (one thread one element; Q: one thread one segment??)
// 4. pass 2: add sum of previous segment to each element of the current segment

override SEGMENT_LENGTH: u32;

@group(0) @binding(0) var<uniform> array_length : u32;
@group(0) @binding(1) var<storage, read> input_array : array<u32>;
@group(0) @binding(2) var<storage, read_write> prefix_sum_intermediate : array<u32>;
@group(0) @binding(3) var<storage, read_write> segment_sum : array<u32>;

@compute
@workgroup_size(1, 1, 1)
fn compute(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) segment_id : vec3<u32>
) {
    let base_addr = segment_id.x * SEGMENT_LENGTH;
    prefix_sum_intermediate[base_addr] = input_array[base_addr];
    for(var i = 1u; i < SEGMENT_LENGTH; i++) {
        let curr_addr = base_addr + i;
        prefix_sum_intermediate[curr_addr] = prefix_sum_intermediate[curr_addr - 1u] + input_array[curr_addr];
    }
    segment_sum[segment_id.x] = prefix_sum_intermediate[base_addr + SEGMENT_LENGTH - 1u];
}
