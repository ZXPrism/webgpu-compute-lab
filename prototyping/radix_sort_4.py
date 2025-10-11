# based on radix_sort_3.py, I think I can optimize it even further..

import random
from rich import print


# cfg
N = 10
MAX_VAL = int(1e6)
radix_bits = 8  # better divide 32
bucket_size = 1 << radix_bits
mask = bucket_size - 1

# prepare input array, suppose element is all u32
input_array = [random.randint(0, MAX_VAL) for _ in range(N)]
print(f"input_array: {input_array}")
print(f"sorted reference: {sorted(input_array)}")

# perform radix sort
sorted_array = [0 for _ in range(N)]

for i in range(0, 32, radix_bits):
    psum_final = [0] * N  # global memory
    tot = [0] * bucket_size

    for j in range(bucket_size):
        # in parallel:

        # shared memory, no need to zero init actually, since we can overwrite old values
        psum = [0] * N
        for k in range(1, N):
            psum[k] = psum[k - 1]
            if (input_array[k - 1] >> i & mask) == j:
                psum[k] += 1

        tot[j] = psum[-1]

        for k in range(N):
            if (input_array[k] >> i & mask) == j:
                psum_final[k] = psum[k]

    final_slot = input_array[-1] >> i & mask
    tot[final_slot] += 1

    psum_slot_size = [0] * bucket_size

    for j in range(1, bucket_size):
        psum_slot_size[j] = psum_slot_size[j - 1] + tot[j - 1]

    # scatter
    for j in range(N):
        slot = (input_array[j] >> i) & mask
        sorted_array[psum_slot_size[slot] + psum_final[j]] = input_array[j]

    input_array, sorted_array = sorted_array, input_array  # ping....pong!

# display sorted array
print(f"sorted_array: {input_array}")
