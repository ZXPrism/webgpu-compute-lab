export let c_array_length = 100000;
export let c_reduce_kernel_segment_length = 256;
export let c_prefix_sum_kernel_segment_length = 256;
export let c_sort_kernel_segment_length = 256;
export let c_radix_sort_bits = 2;
export let c_reduce_mode = -1; // <0: skip; 0: native; 1: basic; 2: upsweep; 3: flatten
export let c_prefix_sum_mode = 3; // <0: skip; 0: native; 1: basic; 2: hs; 3: blelloch
export let c_sort_mode = 3; // <0: skip; 0: bubble sort; 1: bubble sort block; 2: radix sort block
