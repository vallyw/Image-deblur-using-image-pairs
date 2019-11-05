import numpy as np
import cv2
import math
from scipy import signal


def denoise_preprocess(N):
    Avg = []
    Std = []
    Var = []
    N = np.array(N,'uint16')

    for i in range(0,10):
        I = N * (i*0.2 + 0.1)
        Avg.append(np.mean(I))
        Std.append(np.std(I))
        var = (np.std(I))*(np.std(I))
        Var.append(var)
    slope,Yint = np.polyfit(Avg,Var,1)


    # DV = A*B+Yint(拟合曲线)
    A = slope
    DV = Var[1]
    B = (DV - Yint) / A
    return [A,DV,B]

def denoise_channel( N, A, DV, B, method):
    imch = np.array(N,'uint16')

    if A == 0:
        sig = DV
        [Ny,Nx] = imch.shape
    else:
        gaussian = [
            [0.0001, 0.0006, 0.0022, 0.0054, 0.0093, 0.0111, 0.0093, 0.0054, 0.0022, 0.0006, 0.0001],
            [0.0006, 0.0032, 0.0111, 0.0273, 0.0469, 0.0561, 0.0469, 0.0273, 0.0111, 0.0032, 0.0006],
            [0.0022, 0.0111, 0.0392, 0.0963, 0.1653, 0.1979, 0.1653, 0.0963, 0.0392, 0.0111, 0.0022],
            [0.0054, 0.0273, 0.0963, 0.2369, 0.4066, 0.4868, 0.4066, 0.2369, 0.0963, 0.0273, 0.0054],
            [0.0093, 0.0469, 0.1653, 0.4066, 0.6977, 0.8353, 0.6977, 0.4066, 0.1653, 0.0469, 0.0093],
            [0.0111, 0.0561, 0.1979, 0.4868, 0.8353, 1.0000, 0.8353, 0.4868, 0.1979, 0.0561, 0.0111],
            [0.0093, 0.0469, 0.1653, 0.4066, 0.6977, 0.8353, 0.6977, 0.4066, 0.1653, 0.0469, 0.0093],
            [0.0054, 0.0273, 0.0963, 0.2369, 0.4066, 0.4868, 0.4066, 0.2369, 0.0963, 0.0273, 0.0054],
            [0.0022, 0.0111, 0.0392, 0.0963, 0.1653, 0.1979, 0.1653, 0.0963, 0.0392, 0.0111, 0.0022],
            [0.0006, 0.0032, 0.0111, 0.0273, 0.0469, 0.0561, 0.0469, 0.0273, 0.0111, 0.0032, 0.0006],
            [0.0001, 0.0006, 0.0022, 0.0054, 0.0093, 0.0111, 0.0093, 0.0054, 0.0022, 0.0006, 0.0001]
        ]
        gaussian = gaussian/np.sum(gaussian)
        #sig = math.fftconv2(imch, gaussian) - B
        sig = signal.convolve2d(imch, gaussian,boundary='symm',mode='same')- B
        sig[sig < 0] = 0
        sig = sig * A + DV
        sig = np.sqrt(sig)
        [Nys,Nxs] = imch.shape
        [Ny,Nx] = sig.shape
        Nys = (Nys - Ny) / 2 + 1
        Nxs = (Nxs - Nx) / 2 + 1
        #imch = imch[Nys:Nys + Ny - 1, Nxs:Nxs + Nx - 1]
    if (method == 'fastnlm'):
        im_d = fast_nlm_II(imch, 7, 15, sig)
    else:
        print("No other method available")
        return []
    # elif(method == 'blsgsm'):
    #     im_d = call_denoi_bls_gsm(imch, Nx, Ny, sig)
    return im_d

def fast_nlm_II(NoisyImg,PatchSizeHalf,WindowSizeHalf,Sigma):
     [Height,Width] = NoisyImg.shape
     u = np.zeros((Height,Width))
     M = u  # weight max
     Z = M  # Initialize the accumlated weights
     PaddedImg = np.lib.pad(NoisyImg,[PatchSizeHalf,PatchSizeHalf],'symmetric')
     PaddedV = np.lib.pad(NoisyImg,[WindowSizeHalf,WindowSizeHalf],'symmetric')
     for dx in range(-WindowSizeHalf,WindowSizeHalf+1):
         for dy in range(-WindowSizeHalf,WindowSizeHalf+1):
             if (dx != 0 or dy != 0):
                 # Compute the Integral Image
                 Sd = integralImgSqDiff(PaddedImg,dx,dy)
                 # Obtaine the Square difference for every pair of pixels
                 SqDist_rows,SqDist_cols  = Sd.shape
                 my =  Sd[PatchSizeHalf:SqDist_rows-PatchSizeHalf,PatchSizeHalf:SqDist_cols-PatchSizeHalf]
                 SqDist = Sd[PatchSizeHalf:SqDist_rows-PatchSizeHalf,PatchSizeHalf:SqDist_cols-PatchSizeHalf]+Sd[0:SqDist_rows-2*PatchSizeHalf,0:SqDist_cols-2*PatchSizeHalf]-Sd[0:SqDist_rows-2*PatchSizeHalf,PatchSizeHalf:SqDist_cols-PatchSizeHalf]-Sd[PatchSizeHalf:SqDist_rows-PatchSizeHalf,0:SqDist_cols-2*PatchSizeHalf]
                 # Compute the weights for every pixels
                 w = np.exp(-SqDist/(2*Sigma*Sigma))
                 # Obtaine the corresponding noisy pixels
                 v = PaddedV[(WindowSizeHalf+dx):(WindowSizeHalf+dx+Height),(WindowSizeHalf+dy):(WindowSizeHalf+dy+Width)]
                 # Compute and accumalate denoised pixels
                 u = u+w*v
                 # Update weight max
                 M = np.maximum(M,w)
                 # Update accumlated weighgs
                 Z = Z+w
     f = 1
     u = u+f*M*NoisyImg
     u = u/(Z+f*M)
     # Output denoised image
     DenoisedImg = u
     return DenoisedImg

def integralImgSqDiff(v,dx,dy):
    t = img2DShift(v,dx,dy)
    diff = (v-t)*(v-t)
    Sd = np.cumsum(diff,0)
    Sd = np.cumsum(Sd,1)
    return Sd

def img2DShift(v,dx,dy):
    t = np.zeros(v.shape)
    Type = (dx>0)*2+(dy>0)
    t_rows,t_cols = t.shape
    if (Type == 0):
        t[-dx:t_rows,-dy:t_cols] = v[0:t_rows+dx,0:t_cols+dy]
    elif (Type == 1):
        t[-dx:t_rows,0:t_cols-dy] = v[0:t_rows+dx,dy:t_cols]
    elif (Type == 2):
        t[0:t_rows-dx,-dy:t_cols] = v[dx:t_rows,0:t_cols+dy]
    elif (Type == 3):
        t[0:t_rows-dx,0:t_cols-dy] = v[dx:t_rows,dy:t_cols]
    return t

def denoise(N):
    """
    Return "uint16" type
    """
    Nd = np.zeros(N.shape,type(N))
    Nd = np.array(Nd,'uint16')
    try:
        rows,cols,channels = N.shape
        for i in range(0,channels):
            print(i)
            A,DV,B = denoise_preprocess(N[:,:,i]*255)
            # let A be 0, simply use DV as sigma
            #A = 0
            Nd[:,:,i] = denoise_channel(N[:,:,i]*255, A, DV, B, 'fastnlm')
    except:
        rows,cols = N.shape
        channels = 1
        A,DV,B = denoise_preprocess(N[:,:]*255)
        # let A be 0, simply use DV as sigma
        #A = 0
        Nd[:,:] = denoise_channel(N[:,:]*255, A, DV, B, 'fastnlm')
    Nd = Nd/255
    return Nd