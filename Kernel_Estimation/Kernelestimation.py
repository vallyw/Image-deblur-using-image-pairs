import math
import numpy as np
from scipy.sparse import csr_matrix
from numba import jit
from .l1ls import l1_ls_nonneg as L

@jit
def img2mat(I, lens):

    [h,w] = I.shape
    km = np.zeros((w*h,lens**2))
    ks = math.floor((lens-1)/2)
    kt = math.floor(lens/2)


    for i in range(0,h):
        for j in range(0,w):
            for p in range(-ks,kt+1):
                for q in range(-ks,kt+1):
                    if i+p >= 0 and j+q>=0 and i+p <= h-1 and j+q <=w-1:
                        km[i*w + j, q + ks + (p+ks)*lens] = I[i+p,j+q]
                    # else:
                    #     km[i*w + j, q+ks+(p+ks)*lens ] = 0
    return km

def mat2vec(M):

    return M.reshape((M.size,1))


def kernel_estimation(I, B, lens = 5, lam = 1, method = 'l1ls', verbose = True):
#    
#    [h,w] = I.shape
#    if h<100 and w <100:
#        # row = range(0,h)
#        # coloum = range(0,w)
#        I = I
#        B = B
#    elif h<100 and w > 100:
#        # row = range(0,h)
#        # coloum = range(w//2-50,w//2+50)
#        I =I[0:h,w//2-50:w//2+50]
#        B =B[0:h,w//2-50:w//2+50]
#    elif h>100 and w < 100:
#        # row = range(h//2-50,h//2+50)
#        # coloum = range(0,w)
#        I = I[h//2-50:h//2+50,0:w]
#        B = B[h//2-50:h//2+50,0:w]
#    else:
#        # row = range(h//2-50,h//2+50)
#        # coloum = range(w//2-50,w//2+50)
#        I = I[h//2-50:h//2+50,w//2-50:w//2+50]
#        B = B[h//2-50:h//2+50,w//2-50:w//2+50]
    
    if verbose:
        print("Starting kernel estimation...")
    
    niter = 50
    if method == 'l1ls':        
        A = img2mat(I,lens)
        b = mat2vec(B)
#        A_sparse = csr_matrix(A)
        rel_tol = 0.001
        [K, status, history] = L.l1ls_nonneg(A,b,lam**2,tar_gap=rel_tol,quiet=not verbose)

    elif method == 'landweber':

        K = np.zeros([lens**2,1])
        K[int((lens**2+1)/2)] = 1

        A = img2mat(I, lens)
        b = mat2vec(B)

        At = A.transpose()
        AtA = At.dot(A)
        Atb = At.dot(b)
        lambd2 = lam**2
        E = np.identity(lens**2)
        beta = 1.0  # The beta parameter in the paper

        # t_low = 0.03
        # t_high = 0.05
        # M_high = np.zeros(K.shape)
        # M_low = np.zeros(K.shape)
        # M_high[K>t_high*K.max()] = 1
        # M_low[K>t_low*K.max()] = 1
        # K = K*M_high

        for i in range(1,niter+1):
            # K = K*M_low
            K = K + beta * (Atb - (AtA + lambd2*E).dot(K))
            K[K<0] = 0
            K = K/K.sum()
            ## Hysteresis thresholding
            # M_high = np.zeros(K.shape)
            # M_low = np.zeros(K.shape)
            # M_high[K>t_high*K.max()] = 1
            # M_low[K>t_low*K.max()] = 1
            # K = K*M_high
    else:
        K = []
        print('Unimplemented method')
        return K

    K = np.reshape(K,[lens,lens])
    K = K/K.sum()
    
    if verbose:
        print("Kernel estimation complete")
    return K


