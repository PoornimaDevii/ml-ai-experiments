import numpy as np
from timeit import default_timer as timer
from numba import vectorize

# This program allows you to compare the speeds of computation
# on CPU with that on GPU on your local system

def VectorAdd_cpu(a,b,c):
    for i in range(len(a)):
        c[i] = a[i] + b[i]

@vectorize(['float32(float32, float32)'], target = "cuda")
def VectorAdd_gpu(a,b):
    return a + b

def main():

    # Number of elements per numpy array
    N = 32000000

    # Initialise two arrays with "all-ones" as inputs to add
    A = np.ones(N, dtype = np.float32)
    B = np.ones(N, dtype= np.float32)

    # Initialise one array with "all-zeroes" as output placeholder
    C = np.zeros(N, dtype=np.float32)

    # Run vector addition on CPU
    start_on_cpu = timer()
    VectorAdd_cpu(A,B,C)
    vectoradd_cpu_time = timer() - start_on_cpu

    print("C[:5] = " + str(C[:5]))
    print("C[-5:] = " + str(C[-5:]))
    print("VectorAdd running on CPU took %f seconds" % vectoradd_cpu_time)

    # Run vector addition on GPU
    start_on_gpu = timer()
    C = VectorAdd_gpu(A,B)
    vectoradd_gpu_time = timer() - start_on_gpu

    print("C[:5] = " + str(C[:5]))
    print("C[-5:] = " + str(C[-5:]))

    print("VectorAdd running on GPU took %f seconds" % vectoradd_gpu_time)


if __name__ == '__main__':
    main()
