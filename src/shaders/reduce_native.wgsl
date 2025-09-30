@group(0) @binding(0) var<uniform> array_length: u32;
@group(0) @binding(1) var<storage, read> input_array: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_sum: u32;

@compute
@workgroup_size(1, 1, 1)
fn compute(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let n = array_length;
    var sum = 0u;
    for(var i = 0u; i < n; i++) {
        sum += input_array[i];
    }
    output_sum = sum;
}
