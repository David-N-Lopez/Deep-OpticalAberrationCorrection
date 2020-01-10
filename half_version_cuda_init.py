from numba import cuda, int32, float32, void
import numpy as np
from string import Template
import time

#defining n,m index to sequential ordering conversion function
def nmToterm(n, m):
    if (n == 0):
        return 0
    else:
        return (m + n) // 2 + 1 + nmToterm(n - 1, n - 1)

#add function for cuda kernel to execute
@cuda.jit#(void(int32, int32, int32, float32, float32, float32, float32))
def add(num_of_terms, a, x, y):
    #index of individual threads in each block
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    #number of threads working in parallel
    stride = cuda.blockDim.x * cuda.gridDim.x
    for i in range(index, sizeX * sizeY, stride):
        for j in range(0, num_of_terms):
            m = i % sizeX
            n = i // sizeX
            y[n][m] = (y[n][m] + a[j] * x[j * sizeX * sizeY + i])
        y[n][m] = y[n][m] % 1

#defining string templates for importingzernike polynomial npy files
first_template = Template("D:\\zernike\\zernike_polynomials\\high_res-n=$n")
second_template = Template("&m=$m")
third_string = "_half.npy"

#width of zernike polynomial map
sizeX = 500
#height of zernike polynomial map
sizeY = 500
#number of order of parameter n of zernike polynomials to use
num_of_orders = 21
#total number of zernike terms used determined by n parameter used
num_of_terms = nmToterm(num_of_orders, num_of_orders)
#copying weights to device

#initializing host polynomial memory array
h_polynomial = np.zeros((num_of_terms, sizeX, sizeY), np.float32)
#initializing host addition result array
h_result = np.zeros((sizeX, sizeY), np.float32)

#importing polynomials
for n in range(1, num_of_orders + 1):
    for m in range(-n , n + 2, 2):
        term = nmToterm(n, m) - 1
        first_string = first_template.substitute(n = str(n))
        second_string = second_template.substitute(m = str(m))
        filename = first_string + second_string + third_string
        h_polynomial[term] = np.load(filename)
        print(term)

#copying polynomials from host to device
h_polynomial = h_polynomial.ravel()
d_polynomial = cuda.device_array(num_of_terms * sizeX * sizeY, dtype=np.float32)
d_polynomial = cuda.to_device(h_polynomial)
#releasing device memory from polynomials imported
del h_polynomial
#creating device memory for addition result
d_result = cuda.device_array((sizeX, sizeY), dtype=np.float32)
d_result = cuda.to_device(h_result)

#defining number of threads in the block
blockSize = 256
#calculating total number of blocks needed for one addition operation
numBlocks = (sizeX * sizeY + blockSize - 1) // blockSize
