import numpy as np
import cv2
import time
import camera
import matplotlib.pyplot as plt

display = np.ndarray((100,224,224))

start = time.time()
for i in range(100):
   tmp = camera.capture()
   display[i] = tmp
   cv2.imshow("camera", tmp)
   cv2.waitKey(1)

end = time.time()

print("time passed: ", end - start)

display = np.asarray(display)

correlation = np.ndarray(100)

average = np.average(display, axis = 0)

for i in range(0, 100):
   first = display[0:100-i]
   second = display[i:100]
   first_ave = np.average(first, axis = 0)
   second_ave = np.average(second, axis = 0)
   first_std = np.std(first, axis = 0)
   second_std = np.std(second, axis = 0)
   cor = (np.average(first * second, axis = 0) - first_ave * second_ave)#/(first_std * second_std)
   correlation[i] = np.average(cor)

plt.plot(correlation)
plt.show()

cv2.destroyAllWindows()
camera.close()