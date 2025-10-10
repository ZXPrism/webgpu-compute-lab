@group(0) @binding(0) var<uniform> array_length : u32;
@group(0) @binding(1) var<storage, read> input_array : array<u32>;
@group(0) @binding(2) var<storage, read_write> sorted_array : array<u32>;

@compute
@workgroup_size(1, 1, 1)
fn compute(@builtin(global_invocation_id) global_invocation_id : vec3<u32>) {
  for(var i = 0u; i < array_length; i++) {
    sorted_array[i] = input_array[i];
  }

  for(var i = 0u; i < array_length; i++) {
    for(var j = array_length - 1u; j > i; j--) {
      if sorted_array[j - 1u] > sorted_array[j] {
          let temp = sorted_array[j];
          sorted_array[j] = sorted_array[j - 1u];
          sorted_array[j - 1u] = temp;
      }
    }
  }
}
