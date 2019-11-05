# Image deblurring using noisy/blurred image pairs

Image deblurring was basically achieved by _non-blind method_, that is, we first estimate our blur kernel and then deconvolution using our obtained blurred kernel. So the project can be roughly divided into two parts, **kernel estimation** and **deconvolution**.

## Introduction to each part

> ### Kernel Estimation
>
> Basically, the process of Image blurring can be actually be described as $B = I \bigotimes K$ . As a result, to restore the image, we first have to figure out the $K$ . First, we transform the convolution form $B = I\bigotimes K$ as $ b=Ak$ , where $b$ and $k$ are the vector forms of $B$ and $K$ , and $A$ is the matrix form of $I$ .    
>
> Here, we transform the problem into matrix problem, which can be described as linear language and be solved using the knowledge of Linear algebra. That is we describe this problem as following optimization:
> $$
> min||Ak-b||^{2}+\lambda||k||_{1}, subject\quad to\quad k_{i} \geq 0, \quad and\quad\sum_i k_{i} =1
> $$
> This is called $L_{1}$ _Least_ _Square_ _Optimization_ Problem, so to estimate the kernel is to solve the optimization problem mentioned above. 
>
> In the program, the kernel estimation is achieved by calling the function `kernel_estimation()`  , then it will return the estimated kernel. 

>## Deconvolution
>
>After solving out the kernel, here comes to the _deconvolution_ problem. we have carried out four methods to deconvolve the blurring image, and they are called _Richardson Lucy Algorithm, (RL)_ , _Residual RL method_ , _Gain-Control Residual RL_ , and _Detail layer added RL_ . Actually, the latter three methods are based on the direct _RL_ method, which are the improvement of the _RL method_ , as you can see from the result of each restoration image in the **./result/** directory.

## Illustration to the program

> ## File & Directory Description
>
> |       Name        |                         Description                          |
> | :---------------: | :----------------------------------------------------------: |
> |      main.py      |           __File__ The entry of the whole program            |
> |      results      | __Directory__, Used to save the output of the program, such as the _deblurring image_, _estimated kernel_, _Image Quality Evaluation_, etc |
> |      images       |                **Directory** , the test image                |
> | Kernel_Estimation | **Package** It contains the function used to estimate the kernel, like _kernel_estimation_ , and the _l1ls_ used to solve the *$L_{1}$ least square problem*. |
> |    Affiliated     | __Package__ It contains many auxiliary function, like _denoise_ , _motion_blur_ , _blur kernel generator_ , _addnoise_ etc, which all play great role in testing our result with the ground truth. |
> |   Deconvolution   | **Package** It involves many function and its dependency to deconvolve the image. |
>
> ## How to implement
>
> In the `main.py` file, there are several **key variables**, they are *`num_to_cal`* , *`is_random_kernel`*, *`size_of_kernel`* .
>
> First you have to choose how many images you want to calculate at this time, which is controlled by the variable *`num_to_cal`* , so you can modify the value in this code line
>
> > num_to_cal = 1
>
> Then, you choose to whether to generate the random kernel given specific size, which are controlled by  *`is_random_kernel`*, *`size_of_kernel`* 
>
> > is_random_kernel = False 
> > size_of_kernel = 30  
>
> As a matter of fact, this algorithm, after testing we find, will get a more satisfying result when the direction of the blur is along just one direction
>
> The kernel estimation function is called like:
>
> > K_estimated = kernel_estimation(Nd,B,lens=size_of_kernel,lam=5,method='l1ls')
>
> The deconvolution function is called like:
>
> > deBlur = deconv(Nd,B,K_estimated,mode=demode)
>
> ​	`Nd`, is the image denoised; 
> ​	`B`, is the blurred image; 
> ​	`K_estimated`, is the blur kernel; 
> ​	`mode`, is the variable controlling which method to choose, such as 'detailedRL','lucy','resRL','gcRL', which are corresponding to  the method I have mentioned above; 
> ​	`method` , used to choose which algorithm to estimate, up to now, though, we have carried out another method, but it seems to not performing very well, but you can still have a try, that's change the `method='l1ls'` to `method='landweber'`

## Summary

The result of using this method is not satisfying, mainly resulting from the inaccuracy estimation of the kernel, and iteration algorithm to deconvolution is not so good, leaving "ringing" effect after deconvolution. Besides, the denoising effect is also not satisfying sometimes, resulting in the bad quality of the restoration image, and maybe it's also one of the reason contributed to the inaccuracy of the kernel estimation. And that is the reason we refer to the *neutral network based joint image denoising and deblurring* .







