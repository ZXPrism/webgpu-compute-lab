# based on radix_sort_4.py
# segmented prefix calc + global scatter

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
    pass

    input_array, sorted_array = sorted_array, input_array  # ping....pong!

# display sorted array
print(f"sorted_array: {input_array}")
