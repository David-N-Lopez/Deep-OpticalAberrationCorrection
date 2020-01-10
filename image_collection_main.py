from numba import cuda
import numpy as np
import cv2
from string import Template
import time

from cuda_init import *
import camera

cv2.namedWindow("SLM")

for batch in range(4000):
    filename = 'D:\\zernike\\zernike_coefficients\\Zernike_Coefficients_batch_' + str(batch) + '.npy'
    #initialize weights in polynomial addition
    weights = np.load(filename)
    #addition of polynomials for multiple times
    gpu_start = time.time()
    for i in range(0, 200):
        start = time.time()
        #clearing device addition result memory
        d_result = cuda.device_array((sizeX * sizeY), np.float)
        print("time of initializing memory: ", time.time()-start)
        for j in range(0, num_of_terms):
            #actual addition operation on device
            add[numBlocks, blockSize](j, sizeY, sizeX, weights[i, j], d_polynomial, d_result)
        print("time of addition: ", time.time()-start)
        #copying device addition result memory to host
        h_result = d_result.copy_to_host()
        print("time of copying memory: ", time.time()-start)
        h_result = (np.reshape(h_result,(sizeX, sizeY))) % 1
        print("time of wrapping: ", time.time()-start)
        cv2.moveWindow("SLM", 2140, -25)
        cv2.imshow("SLM", h_result)
        cv2.waitKey(1)
        print("time of display: ", time.time()-start)
        display = camera.capture()
        print("time of image acquisition:", time.time()-start)
        #np.save('D:\\zernike\\captured_images_for_training\\image_' + str(batch) + '_' + str(i) + '.npy', display)
        #cv2.imshow('display',display)
        #cv2.waitKey(1)
	
    gpu_time = time.time() - gpu_start
    print("Time elapsed for GPU is "+str(gpu_time)+"seconds")

cv2.destroyAllWindows()
camera.close()

#cpu_start = time.time()
#for i in range(0, 200):
#    h_result = np.zeros((sizeX * sizeY))
#    for j in range(0, num_of_terms):
#        cpuadd(j, sizeY, sizeX, weights[i, j], h_polynomial, h_result)
#
#cpu_time = time.time() - cpu_start
#print("Time elapsed for CPU is "+str(cpu_time)+"seconds")