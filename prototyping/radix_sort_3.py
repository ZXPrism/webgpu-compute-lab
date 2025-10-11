# based on radix_sort_2.py, I think I can optimize it even further

import random
from rich import print


# cfg
N = 10
MAX_VAL = int(1e6)
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
    # in radix_sort_2.py, we use psum = [[0] * bucket_size for _ in range(N)]
    # but many elements in this array are wasted, again, since every element only belongs to one slot
    # is it possible, to construct a single psum array with length N, where the psum for each slot is interleaving?
    # maybe a bit confusing, I mean, in essence, we can pack bucket_size prefix sum arrays (each with length N) into one single length N array

    # an intuitive way would be tracking the last index of the same-slot element
    last = [-1] * bucket_size
    psum = [0] * N
    tot = [0] * bucket_size

    for j in range(N):  # in parallel
        slot = (input_array[j] >> i) & mask
        if last[slot] != -1:
            psum[j] = psum[last[slot]] + 1
        last[slot] = j
        tot[slot] += 1

    psum_slot_size = [0] * bucket_size
    for j in range(1, bucket_size):
        psum_slot_size[j] = psum_slot_size[j - 1] + tot[j - 1]

    # scatter
    for j in range(N):
        slot = (input_array[j] >> i) & mask
        sorted_array[psum_slot_size[slot] + psum[j]] = input_array[j]

    input_array, sorted_array = sorted_array, input_array  # ping....pong!

# display sorted array
print(f"sorted_array: {input_array}")
