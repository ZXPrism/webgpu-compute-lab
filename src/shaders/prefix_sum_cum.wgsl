@group(0) @binding(0) var<uniform> array_length : u32;
@group(0) @binding(1) var<storage, read> input_array : array<u32>;
@group(0) @binding(2) var<storage, read_write> prefix_sum : array<u32>;

@compute
@workgroup_size(1, 1, 1)
fn compute(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {

}
