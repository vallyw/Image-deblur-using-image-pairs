import skimage.io as io
import  skimage.color as color
import numpy as np
import  cv2


direction_dict = {0:(1,0),
                  1:(1,1),
                  2:(0,1),
                  3:(-1,1),
                  4:(-1,0),
                  5:(-1,-1),
                  6:(0,-1),
                  7:(1,-1)}

def blur(img, kernel):
    img_blur = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    return np.clip(img_blur,0,1)
def kernel_generator(kernel_size):
    kernel = np.zeros(shape=(kernel_size,kernel_size), dtype=np.float32)
    center_x = np.random.randint(int(0.4*kernel_size),int(0.6*kernel_size))
    center_y = np.random.randint(int(0.4*kernel_size),int(0.6*kernel_size))

    movements = np.random.randint(0.05*kernel_size*kernel_size, 0.15*kernel_size*kernel_size)
    current_x = center_x
    current_y = center_y
    for i in range(movements):
        kernel[current_y][current_x] += 1
        direction = np.random.randint(0,8)
        move = direction_dict[direction]
        current_x += move[1]
        current_y += move[0]
        if current_y>=kernel_size or current_y<0:
            current_y -= move[0]
        if current_x>=kernel_size or current_x<0:
            current_x -= move[1]
    #kernel = kernel/np.max(kernel)
    # for visualization only
    kernel /= movements
    return kernel

def noise(img, sigma):
    h = img.shape[0]
    w = img.shape[1]
    noise = np.random.randn(h, w) * sigma /255.0
    img_noise = np.clip(img+noise,0,1)
    return img_noise
'''test code
kernel = kernel_generator(48)
img = io.imread('F://university/course/4th_1/acquisition/prj/KEP-Net/francis.JPG',as_grey=True)
img_blur = blur(img, kernel)
img_noise = noise(img, 30)
io.imshow(img)
io.show()
io.imshow(kernel)
io.show()
io.imshow(img_blur)
io.show()
io.imshow(img_noise)
io.show()
'''


def Img2patch(img, patch_size, flatten=True):
    # directly crop but not pad
    new_shape = np.zeros_like(img.shape)
    new_shape[0] = np.floor(img.shape[0]/patch_size)*patch_size
    new_shape[1] = np.floor(img.shape[1]/patch_size)*patch_size

    patches = np.zeros(shape=(new_shape[0]*new_shape[1]//(patch_size**2),patch_size,patch_size))
    for i in range(new_shape[0]//patch_size):
        for j in range(new_shape[1]//patch_size):
            patches[i*(new_shape[1]//patch_size)+j] = \
                img[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
    if flatten:
        patches = np.reshape(patches, (new_shape[0]*new_shape[1]//(patch_size**2),patch_size**2))
    else:
        patches = np.reshape(patches, (new_shape[0] * new_shape[1] // (patch_size ** 2), patch_size, patch_size))
    return patches,new_shape

def Patch2img(patches, patch_size, new_shape, flattened=True):
    if flattened == False:
        patches = np.reshape(patches, (new_shape[0] * new_shape[1] // (patch_size ** 2), patch_size ** 2))
    img = np.zeros(shape=new_shape, dtype=np.float32)
    for i in range(new_shape[0]//patch_size):
        for j in range(new_shape[1]//patch_size):
            img[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size] = \
                np.reshape(patches[i*(new_shape[1]//patch_size)+j],(patch_size,patch_size))
    return img
