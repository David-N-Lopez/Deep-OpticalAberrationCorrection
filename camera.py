import ctypes
import numpy as np
from pyueye import ueye
import copy
import time

uEyeDll = ctypes.cdll.LoadLibrary(r"C:\Windows\System32\uEye_api_64.dll")

cam = ctypes.c_uint32(0)
hWnd = ctypes.c_voidp()
msg=uEyeDll.is_InitCamera(ctypes.byref(cam),hWnd)
uEyeDll.is_EnableAutoExit (cam, ctypes.c_uint(1))
col = ueye.c_int(0)
mod = ueye.c_int(0)
ueye.is_GetColorDepth(cam, col, mod)
ueye.is_SetColorMode (cam, ueye.IS_CM_MONO8)
imageSize = ueye.IS_SIZE_2D()
imageSize.s32Width=ueye.c_int(224)
imageSize.s32Height=ueye.c_int(224)
ueye.is_AOI(cam, ueye.IS_AOI_IMAGE_SET_SIZE, imageSize, ueye.sizeof(imageSize))
width_py = imageSize.s32Width
height_py = imageSize.s32Height
pixels_py = 8
width = width_py
height = height_py
bitspixel=ctypes.c_int(pixels_py)
pcImgMem = ctypes.c_char_p()
pid=ctypes.c_int()
uEyeDll.is_AllocImageMem(cam, width, height,  bitspixel, ctypes.byref(pcImgMem), ctypes.byref(pid))
uEyeDll.is_SetImageMem(cam, pcImgMem, pid)
ImageData = np.ones((224,224),dtype=np.uint8)
uEyeDll.is_SetExternalTrigger(cam, ueye.IS_SET_TRIGGER_OFF)
#exp = ctypes.c_double()
#print("Setting exposure time.")
#uEyeDll.is_SetExposureTime(cam, ctypes.c_double(1), exp)
print("Optimizing camera clock.")
uEyeDll.is_SetOptimalCameraTiming(cam, ueye.IS_BEST_PCLK_RUN_ONCE, ueye.c_int(4000))
actual_fr = ctypes.c_double()
print("Setting frame rate.")
uEyeDll.is_SetFrameRate(cam, ctypes.c_double(100), actual_fr)
uEyeDll.is_CaptureVideo(cam, ueye.IS_WAIT)

def capture():
    uEyeDll.is_FreezeVideo(cam, ueye.IS_WAIT)
    ueye.is_CopyImageMem(cam, pcImgMem, pid, ImageData.ctypes.data)
    return ImageData

def close():
    ueye.is_ExitCamera(cam)
