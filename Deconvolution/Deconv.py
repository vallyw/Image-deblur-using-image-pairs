import math
import cv2 as cv
import numpy as np
from numpy import linalg 
from skimage import restoration
from scipy.signal import convolve2d, fftconvolve

def Igain_map(Nd,alpha):
    def Gscale(I,level,gsize,sig):
        # gaussian_I = cv.GaussianBlur(I,gsize,sig)
        pyramid_I = []
        for i in range(level):
            if i == 0:
                gaussian_I = cv.GaussianBlur(I,gsize,sig)
                pyramid_I.append(gaussian_I)
            else:
                gaussian_I = cv.GaussianBlur(pyramid_I[i-1],gsize,sig)
                im1 = gaussian_I[:,0:gaussian_I.shape[1]:2]
                im2 = im1[0:gaussian_I.shape[0]:2,:]
                pyramid_I.append(im2)
        return pyramid_I


    def imgradientxy(imgI, method = 'CentralDifference'):
        if method =='sobel':
            Gx = cv.Sobel(imgI,-1,1,0,ksize=3)
            Gy = cv.Sobel(imgI,-1,0,1,ksize=3)
        
        elif method == 'prewitt':
            h = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
            ht = np.transpose(h)
            Gx = cv.filter2D(imgI,-1,ht)
            Gy = cv.filter2D(imgI,-1,h)
        elif method == 'CentralDifference':
            [h,w] = imgI.shape
            if h == 1 and w == 1:
                Gx = 0
                Gy = 0
            elif w == 1 and w != 1:
                Gx = np.zeros(imgI.shape,imgI.dtype)
                Gy = np.gradient(imgI)
            elif h==1 and w != 1:
                Gx = np.zeros(imgI.shape,imgI.dtype)
                Gy = np.gradient(imgI)
            else:
                [Gx,Gy] = np.gradient(imgI)
        elif method == 'intermediatedifference':
            Gx = np.zeros(imgI.shape,imgI.dtype)
            if imgI.shape[1]>1:
                Gx[:,0:-1] = np.diff(imgI,1,1)
            
            Gy = np.zeros(imgI.shape,imgI.dtype)
            if imgI.shape[0]>1:
                Gy[:,0:-1] = np.diff(imgI,1,0)
        else:
            return None
        return [Gx,Gy]
            

    def norm(x,n):
        """
        Calculate the form like ||x|| in certain ord
        """
        return linalg.norm(x,n)


    lmax = 10
    gsize = (5,5)
    sig = 0.5
    [h,w] = Nd.shape

    Nd1 = Gscale(Nd,lmax,gsize,sig)

    grad_img = []
    # temp = {}
    tempgx = 0
    tempgy = 0 
    
    for i in range(lmax):
        # [temp['Gx'],temp['Gy']] = imgradientxy(Nd1[i],'CentralDifference')
        # temp['Gx'] = temp['Gx']/ (2**(i))
        # temp['Gy'] = temp['Gy']/ (2**(i))
        # temp['Gx'] = cv.resize(temp['Gx'],(w,h),interpolation=cv.INTER_LINEAR)
        # temp['Gy'] = cv.resize(temp['Gy'],(w,h),interpolation=cv.INTER_LINEAR)
        # grad_img.append(temp)
        [tempgx,tempgy] = imgradientxy(Nd1[i],'CentralDifference')
        tempgx = tempgx/ (2**(i))
        tempgy = tempgy/ (2**(i))
        tempgx = cv.resize(tempgx,(w,h),interpolation=cv.INTER_LINEAR)
        tempgy = cv.resize(tempgy,(w,h),interpolation=cv.INTER_LINEAR)
        grad_img.append({"Gx":tempgx,"Gy":tempgy})

    Igain = np.zeros((h,w),dtype = np.float)
    
    for i in range(h):
        for j in range(w):
            Sum = 0
            for l in range(lmax):
                Sum += norm([grad_img[l]['Gx'][i,j],grad_img[l]['Gy'][i,j]],2)
            
            Igain[i,j] = (1-alpha) + alpha*Sum
    
    Igain = Igain/Igain.max()

    # Igain = maploop(h,w,lmax,grad_img,alpha)
    # print(Igain)
    return Igain

def deconvgclucy(Nd,dB,K,niter,alpha = 0.2,verbose=True):


    if alpha > 0:
        if verbose:
            print("Calculating I_gain map ...")
        Igain = Igain_map(Nd,alpha)
        if verbose:
            print("Igain map ready...")
    else:
        [h,w] = Nd.shape
        Igain = np.ones((h,w))

    K_mirror = K[::-1,::-1]
    dI = np.zeros(dB.shape)
    dnom = np.zeros(dB.shape)

    if verbose:
        print("Start iterating...")
    
    
    for i in range(niter):
        dI = dI + 1
        dI[dI<0] = 0

        dnom = fftconv2(dI,K)

        tmp = dB/dnom

        dI = fftconv2(tmp,K_mirror)*dI - 1

        if i != niter-1:
            dI = Igain*dI

        dI[dI>1] = 1

    return dI
    

def fftconv2(Nd,K):
    return convolve2d(Nd,K,mode='same',boundary = 'symm')

def jbfilter(Ir,Ig,winr,sig_d,sig_r):

    def gaussian(x,sigma):
        return math.exp((-1*x**2)/(2*sigma**2))
    
    def gaussian_mask(mask,sigma):
        divisor = -2*sigma**2
        mask = np.square(mask)
        mask = np.divide(mask,divisor)
        mask = np.exp(mask)
        return mask
    
    def get_neighbours(img, y, x, rad, height, width):
        diam = rad*2+1
        top = rad if y - rad >=0 else y
        bottom = rad + 1 if y+rad < height else height -y
        left = rad if x - rad >=0 else x
        right = rad + 1 if x + rad < width else width - x
        neighbours = np.zeros((diam,diam))
        neighbours[rad-top:rad+bottom,rad-left:rad+right] = img[y-top:y+bottom,x-left:x+right]
        return neighbours

    def create_mask(rad,sigma):
        diam = 2*rad+1
        mask = np.zeros((diam,diam))
        for i in range(-rad,rad+1):
            for j in range(-rad,rad+1):
                mask[i+rad,j+rad] = gaussian((i**2+j**2)**0.5,sigma)
        Sum = np.sum(mask)
        mask = np.divide(mask,Sum)

        return mask    

    def filter(y,x,A,F,mask,rad,diam,sig_r,height,width):
        F_neighbours = get_neighbours(F,y,x,rad,height,width)
        pix = F_neighbours[rad][rad]
        A_neighbours = get_neighbours(A,y,x,rad,height,width)
        result = np.copy(mask)
        temp1 = np.subtract(F_neighbours,pix)
        temp = gaussian_mask(temp1,sig_r)
        result = np.multiply(result,temp)
        k = np.sum(result)
        result = np.multiply(result,A_neighbours)
        result = np.divide(result,k)
        return np.sum(result)

    def joint_bilateral(A,F,img,rad,sig_d,sig_r,height,width):
        diam = 2*rad+1
        mask = create_mask(rad,sig_d)
        for y in range(height-1):
            for x in range(width-1):
                img[y][x] = filter(y,x,A,F,mask,rad,diam,sig_r,height,width)
        return img
    
    height = Ir.shape[0]
    width = Ir.shape[1]
    img = np.copy(Ir)
    return joint_bilateral(Ir,Ig,img,winr,sig_d,sig_r,height,width)



def deconv(Nd, B, K, mode = 'lucy', verbose = True):

    if verbose:
        print("Start deconvolution")
    
    niter = 20
    alpha = 0.2    # alpha controls the influence of the gain map
    winr = 5
    sig_d = 1.6
    sig_r = 0.08


    if mode =='lucy': # Conventional Richardson Lucy algorithm
        if B.ndim == 2:
            I = restoration.richardson_lucy(B,K,niter)
            # I = restoration.wiener(B,K,11)
        else:
            B = cv.cvtColor(B,cv.COLOR_BGR2GRAY)
            I = restoration.richardson_lucy(B,K,niter)
            # I = np.zeros((B.shape))
            # for i in range(B.ndim):
            #     I[:,:,i] = restoration.richardson_lucy(B[:,:,i],K,niter)
                
    elif mode == 'resRL': # Residual RL algorithm
        NdK = np.zeros(B.shape,B.dtype)
        # [_,_,d ] = B.shape
        # for i in range(d):
        #     NdK[:,:,i] = fftconv2(Nd[:,:,i],K)
        NdK = fftconv2(Nd,K) 
        dB = B - NdK + 1
        dI = deconvgclucy(Nd,dB,K,niter,0,verbose)
        I = Nd + dI

    elif mode == 'gcRL':
        NdK = np.zeros(B.shape)
        # [_,_,d ] = B.shape
        # for i in range(d):
        #     NdK[:,:,i] = fftconv2(Nd[:,:,i],K)
        NdK = fftconv2(Nd,K)
        dB = B - NdK + 1
        dI = deconvgclucy(Nd,dB,K,niter,alpha,verbose)

        I = Nd + dI

    elif mode == 'detailedRL':
        if verbose:
            print("Calculaing residual-RL result Ir ....")
        
        Ir = deconv(Nd,B,K,'resRL',True)

        if verbose:
            print("Calculating gain-controlled RL result Ig ...")
        
        Ig = deconv(Nd,B,K,'gcRL',True)
        if verbose:
            print("Calculating Ibar with joint/cross bilateral filter...")
        Ibar = jbfilter(Ir,Ig,winr,sig_d,sig_r)

        Id = Ir - Ibar
        if verbose:
            # cv.imwrite("./images/detail_layer.jpg",Id+0.8)
            print("Detailed RL completed!!!")
        
        I = Ig + Id     # Final Result
    else:
        print("Unimplemented deconvolution method: {} ".format(mode))
        return []
    

    if verbose:
        print("Deconvolution Complete...")
    
    # Normolize the range of output image I

    I[I<0] = 0
    I[I>1] = 1

    return I
    