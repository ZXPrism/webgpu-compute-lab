import random
from rich import print


# cfg
N = 100
MAX_VAL = int(1e9)
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
    # count
    cnt = [0] * bucket_size
    for elem in input_array:
        cnt[elem >> i & mask] += 1

    # prefix sum (scan)
    # note that we need to do an exclusive scan
    # so that psum[i] points to the start index
    psum = [0] * bucket_size
    for j in range(1, bucket_size):
        psum[j] = psum[j - 1] + cnt[j - 1]

    # scatter
    for elem in input_array:
        label = elem >> i & mask
        sorted_array[psum[label]] = elem
        psum[label] += 1

    input_array, sorted_array = sorted_array, input_array

# display sorted array
print(f"sorted_array: {input_array}")
