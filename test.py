import cv2 as cv
import numpy as np
import math

def fftcon2(I,K):
#    img = cv.imread("./images/eval_2000.jpg")
    [h,w] = I.shape
    [kh,_] = K.shape
    rad = math.floor(0.5*kh)
    

