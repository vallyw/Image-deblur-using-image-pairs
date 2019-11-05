import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.measure import compare_ssim,compare_psnr
from Affiliated import denoise, deconv, motion_blur, addNoise
from Affiliated import utils, auxiliary
from Kernel_Estimation import kernel_estimation

file_path = glob.glob('./images/*.jpg') # Read out all the test image in the /image/ file directory

num_to_cal = 1  # This parameter is used to control the num of the image you want to read out from the ./images datafile
num = 0
is_random_kernel = False     # Decide wheather to generate the random kernel
size_of_kernel = 30     # Decide the size of the kernel generated and estimate

for fname in file_path[0:num_to_cal]:
    print("--"*10)
    print("Starting the {} round".format(num+1))
#    savename = 'eval_' + str(2000+num)
    savedir = './results/eval_' + str(2000+num)
    os.mkdir(savedir)
    
    I = io.imread(fname,as_gray=True)
    I = img_as_float(I)
    
    if is_random_kernel:
        blur_kernel = utils.kernel_generator(size_of_kernel)
        B = utils.blur(I,blur_kernel)
    else:
        motion_degree = np.random.randint(0,360)    # Generate one specific motion deblur
        B , blur_kernel = motion_blur(I,size_of_kernel,motion_degree)
        
    N = addNoise(I,0,0.1)
    Nd = denoise(N)
    
    K_estimated = kernel_estimation(Nd,B,lens=size_of_kernel,lam=5,method='l1ls')
#    K_estimated = blur_kernel
    
    auxiliary.kernel_write(K_estimated,"estimated",savedir)
    auxiliary.kernel_write(blur_kernel,'true',savedir)

    plt.imsave(savedir+"/original.jpg",I,cmap = 'gray')
    plt.imsave(savedir+"/blurred.jpg",B,cmap = 'gray')
    plt.imsave(savedir+"/denoised.jpg",Nd,cmap = 'gray')


    csvname = savedir + "/IQE.csv"   # Used to estimate the quality of the image, Image quality estimation
    with open(csvname,mode = 'w') as f:
        f_csv = csv.writer(f, delimiter = ',')
        f_csv.writerow(['Method','SSIM','PSNR'])
        
    deconvmode = ['detailedRL','lucy','resRL','gcRL']
## deconvolution, can be 'lucy', 'resRL', 'gcRL' and 'detailedRL'.
    for demode in deconvmode:
        # deBlur = deblur(Nd,B,unikernel = True,deconvmode=demode)
        deBlur = deconv(Nd,B,K_estimated,mode=demode)
        
        plt.imsave(savedir+"/deblurred_"+demode+".jpg",deBlur,cmap = 'gray')
        
        ssim = compare_ssim(I,deBlur)
        psnr = compare_psnr(I,deBlur)

        with open(csvname,mode = 'a+') as f:
            f_csv = csv.writer(f,delimiter = ',')
            f_csv.writerow([demode + ":", ssim, psnr])
            
    num += 1
    print("Complete the {} round".format(num))
print("__"*10)
print("Complete all cycles")
print("__"*10)


