import camera
import fittest
import scanning
import cv2
import numpy as np
from scipy import ndimage
import serial
import struct
import time

n=10

X=fittest.X
Y=fittest.Y

scanning.init_show()
cv2.moveWindow("SLM", 1360,0)
time.sleep(1)
display=camera.capture()
#cv2.imshow('display',display)
#cv2.waitKey(1)
(x_center,y_center)=ndimage.measurements.center_of_mass(display)
input()

measurement=np.zeros((95,53,10))
phase=np.zeros((95,53))
amplitude=np.zeros((95,53))

phase_file = open("phase.txt","a")
amplitude_file = open("amplitude.txt","a")

for i in range(-47,48):
    for j in range(-26,27):
        sinusoidal=np.zeros(n)
        for k in range(0,n):
            scanning.sops_show(i,j,25*k)
            display=camera.capture()
            #cv2.imshow('display',display)
            #cv2.waitKey(1)
            sinusoidal[k]=display[np.int(x_center),np.int(y_center)]
            measurement[i+47,j+26,k]=sinusoidal[k]
        popt=fittest.fit_cos(sinusoidal,n)
        phase[i+47,j+26]=popt[1]
        amplitude[i+47,j+26]=popt[0]
        print(popt[0],popt[1])
        phase_file.write(str(popt[1]))
        phase_file.write('  ')
        amplitude_file.write(str(popt[0]))
        amplitude_file.write('  ')

arduino=serial.Serial('COM3',9600)
time.sleep(1)
arduino.write(struct.pack('>B', 0))
time.sleep(1)
arduino.write(struct.pack('>B', 0))

np.savetxt(r'c:\Users\Physics Lab\Desktop\SOPS\measurement.csv',measurement,delimiter=",")
np.savetxt(r'c:\Users\Physics Lab\Desktop\SOPS\amplitude.csv',amplitude,delimiter=",")
np.savetxt(r'c:\Users\Physics Lab\Desktop\SOPS\phase.csv',phase,delimiter=",")
