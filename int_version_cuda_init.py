from numba import cuda
import numpy as np
from string import Template
import time

#defining n,m index to sequential ordering conversion function
def nmToterm(n, m):
    if (n == 0):
        return 0
    else:
        return (m + n) // 2 + 1 + nmToterm(n - 1, n - 1)

#defining add function for cpu to execute
def cpuadd(j, row, col, a, x, y):
    y += a * x[j * row * col : (j + 1) * row * col]

#add function for cuda kernel to execute
@cuda.jit
def add(j, row, col, a, x, y):
    #index of individual threads in each block
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    #number of threads working in parallel
    stride = cuda.blockDim.x * cuda.gridDim.x
    for i in range(index, row * col, stride):
        y[i] = (y[i] + a * x[j * row * col + i])

#defining string templates for importingzernike polynomial npy files
first_template = Template("D:\\zernike\\zernike_polynomials\\high_res-n=$n")
second_template = Template("&m=$m")
third_string = "_int.npy"

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
h_polynomial = np.zeros((num_of_terms, sizeX, sizeY))
#initializing host addition result array
h_result = np.zeros((sizeX * sizeY))

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
d_polynomial = cuda.device_array(num_of_terms * sizeX * sizeY, dtype=np.int)
d_polynomial = cuda.to_device(h_polynomial)
#releasing device memory from polynomials imported
#del h_polynomial
#creating device memory for addition result
d_result = cuda.device_array(sizeX * sizeY, dtype=np.int)
d_result = cuda.to_device(h_result)

#defining number of threads in the block
blockSize = 32
#calculating total number of blocks needed for one addition operation
numBlocks = (sizeX * sizeY + blockSize - 1) // blockSize
