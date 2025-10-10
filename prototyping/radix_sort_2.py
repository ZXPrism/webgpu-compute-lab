# radix_sort.py is not close to GPU version, since it requires usage of atomic variables to store count
# here is way to avoid atomic by directly constructing the output index of each element for each pass
# from book https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

# for now (25-10-10) I don't know what a real radix sort on GPU look like
# since the book above use radix=2, I am not sure if this approach is applicable for any radix
# but I don't see atomic variables in one radix sort impl, so it either uses this method, or other methods
# nevertheless, I will focus on implmenting the approach proposed in the book
