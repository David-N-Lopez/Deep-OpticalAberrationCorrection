import numpy as np
from string import Template
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

#defining n,m index to sequential ordering conversion function
def nmToterm(n, m):
    if (n == 0):
        return 0
    else:
        return (m + n) // 2 + 1 + nmToterm(n - 1, n - 1)

#defining and compile add gpu code with C++ syntax
mod = SourceModule("""
    __global__
    void add(int j, int row, int col, float a, float *x, float *y)
    {
        int index = blockIdx.x*blockDim.x + threadIdx.x;
        int stride = blockDim.x*gridDim.x;
        for (int i = index; i < row * col; i += stride)
            y[i] = y[i] + a * x[j * row * col + i];
    }
""")

add = mod.get_function("add")

#defining string templates for importingzernike polynomial npy files
first_template = Template("D:\\zernike\\zernike_polynomials\\high_res-n=$n")
second_template = Template("&m=$m")
third_string = ".npy"

#width of zernike polynomial map
sizeX = 1000
#height of zernike polynomial map
sizeY = 1000
#number of order of parameter n of zernike polynomials to use
num_of_orders = 21
#total number of zernike terms used determined by n parameter used
num_of_terms = nmToterm(num_of_orders, num_of_orders)
#copying weights to device

#initializing host polynomial memory array
h_polynomial = np.zeros((num_of_terms, sizeX, sizeY))
h_polynomial = h_polynomial.astype(np.float32)
#initializing host addition result array
h_result = np.zeros((sizeX * sizeY))
h_result = h_result.astype(np.float32)
#creating device memory for addition result
d_result = cuda.mem_alloc(h_result.nbytes)
cuda.memcpy_htod(d_result, h_result)

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
d_polynomial = cuda.mem_alloc(h_polynomial.nbytes)
cuda.memcpy_htod(d_polynomial, h_polynomial)
#releasing device memory from polynomials imported
del h_polynomial

#defining number of threads in the block
blockSize = 256
#calculating total number of blocks needed for one addition operation
numBlocks = (sizeX * sizeY + blockSize - 1) // blockSize