import numpy as np
import cv2 as cv
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
from Kernel_Estimation import kernel_estimation
from Deconvolution.Deconv import deconv
 
def kernel_write(kernel, name, savedir):
    plt.imshow(kernel)
    plt.colorbar()
    fig = plt.gcf()
    plt.margins(0,0)
    fig.savefig(savedir+"/Kernel_" + name , dpi = 500, bbox_inches = 'tight')
    plt.close()

def motion_blur(image, degree=12, angle=45):
    image = np.array(image)
 
    M = cv.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv.warpAffine(motion_blur_kernel, M, (degree, degree))
 
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv.filter2D(image, -1, motion_blur_kernel)
 
    # cv.normalize(blurred, blurred, 0, 255, cv.NORM_MINMAX)
    # blurred = np.array(blurred, dtype=np.float)
    # blurred = blurred/255.0
    blurred = np.clip(blurred,0,1)

    return blurred, motion_blur_kernel

def addNoise(image,mean = 0, var = 0.001):
    """
    Add Gaussian Noise to the image

    """

    image = np.array(image)
    noise = np.random.normal(mean,var,image.shape)
    out = image + noise
    out = np.clip(out,0,1.0)

    return out

    
def deblur(Nd, B, unikernel = True, deconvmode = 'lucy', verbose = True):

    I = np.zeros(B.shape,B.dtype)
    d = B.ndim
    if unikernel:
        if d == 3:
            gNd = cv.cvtColor(Nd,cv.COLOR_BGR2GRAY)
            gB = cv.cvtColor(B,cv.COLOR_BGR2GRAY)
        else:
            gNd = Nd
            gB = B

        ksize = 36   # Kernel Size
        lamb = 5    # lambda used in the L1 least square method
        method_to_estima = 'l1ls'

        K = kernel_estimation(gNd,gB,lens=ksize, lam=lamb**2, method=method_to_estima, verbose = verbose)

        I = deconv(Nd,B,K,mode = deconvmode ,verbose=verbose)

    # deconvolution to each channel separately
    else:
        K = np.zeros((ksize,ksize,d))
        I = np.zeros(Nd.shape)
        for i in range(d):
            K[:,:,i] = kernel_estimation(Nd[:,:,i],B[:,:,i],lens=ksize,lam=lamb,method=method_to_estima,verbose=verbose)
            I[:,:,i] = deconv(Nd[:,:,i],B[:,:,i],K[:,:,i],mode =deconvmode,verbose=verbose)

    return I
