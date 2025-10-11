# radix_sort.py is not close to GPU version, since it requires usage of atomic variables to store count
# here is way I found (seemingly) to avoid atomic by directly constructing the output index of each element for each pass
# from book https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

# for now (25-10-10) I don't know what a real radix sort on GPU look like
# since the book above use radix=2, I am not sure if this approach is applicable for any radix
# but I don't see atomic variables in one radix sort impl, so it either uses this method, or other methods
# nevertheless, I will focus on implmenting the approach proposed in the book

# well, today morning (25-10-11) I totally understood how the approach in the book above worked!
# and it can be easily genearlized to any radix!! All dots suddenly connect!
# it seemed to be exactly the common approach that those computing libs are using..!
# let me test it..calm down...

import random
from rich import print


# cfg
N = 100
MAX_VAL = int(1e2)
radix_bits = 8  # must divide 32
bucket_size = 1 << radix_bits
mask = bucket_size - 1

# prepare input array, suppose element is all u32
input_array = [random.randint(0, MAX_VAL) for _ in range(N)]
print(f"input_array: {input_array}")
print(f"sorted reference: {sorted(input_array)}")

# perform radix sort
sorted_array = [0 for _ in range(N)]
for i in range(0, 32, radix_bits):
    # don't forget to do exclusive scan
    psum = [[0] * N for _ in range(bucket_size)]

    # many scans tho, can be done parallelly and efficiently on GPU!
    for j in range(bucket_size):
        for k in range(1, N):
            psum[j][k] = psum[j][k - 1]
            if (input_array[k - 1] >> i & mask) == j:
                psum[j][k] += 1

    # psum of each slot sizes, exclusive too
    psum_slot_size = [0] * bucket_size
    for j in range(1, bucket_size):
        psum_slot_size[j] = psum_slot_size[j - 1] + psum[j - 1][-1]
        if (input_array[-1] >> i & mask) == j - 1:
            psum_slot_size[j] += 1

    # scatter
    for j in range(N):
        slot = (input_array[j] >> i) & mask
        sorted_array[psum_slot_size[slot] + psum[slot][j]] = input_array[j]

    input_array, sorted_array = sorted_array, input_array  # ping....pong!

# display sorted array
print(f"sorted_array: {input_array}")
