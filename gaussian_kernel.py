"""
Copyright (c) [2025] [Ye Zhiwei]  Dalian University of Technology
All rights reserved.

This file is part of Deep recongnition of Moleculer fluorescence.

This code is licensed under the [MIT]
You may not use this file except in compliance with the License.

"""
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

class gauss_kernel:
    def corekernel(sigma, size):
        center = [(size[0]-1)/2, (size[0]-1)/2]
        kernel = np.fromfunction(
            lambda x, y:(1/(2 * np.pi * sigma**2)) *
                np.exp(-((x-center[0])**2 + (y-center[1])**2) / (2 * sigma**2)),
            size)
        return kernel/np.sum(kernel)

    def convlayer(gaussiankernel_tensor,x):
        x = F.conv2d(x,gaussiankernel_tensor,
                          padding=gaussiankernel_tensor.shape[2]//2)
        return x

    def fft_convolution(image, kernel):
        kernel_size = kernel.shape
        padded_image = np.pad(image, [
            (0, kernel_size[0]-1), (0, kernel_size[1]-1)],
            mode = 'symmetric')
        #print(f"the size of image is {image.shape}")
        #plt.imshow(padded_image)
        #plt.axis('off')  # Hide the axes
        #plt.show()
        #print(f"the size of paddedimage is {padded_image.shape}")

        #compute the FFT transform
        imagefft = np.fft.fft2(padded_image)
        kernelfft = np.fft.fft2(kernel, s=padded_image.shape)


        #multiply in frequency domain
        convolvedfft = imagefft * kernelfft

        #compute the inverse fft
        convolved = np.fft.ifft2(convolvedfft).real
        #print(f"the size of {convolved.shape}")

        return convolved[(kernel_size[0]//2+1):image.shape[0],(kernel_size[1]//2+1):image.shape[1]]

if __name__ == "__main__":
    # Example usage
    size = (5, 5)  # Size of Gaussian kernel
    sigma = 1.0    # Standard deviation
    kernel = gauss_kernel
    kernelcore = kernel.corekernel(sigma, size)
    image = np.random.rand(100, 100)  # Example image

    plt.imshow(image)
    plt.axis('off')  # Hide the axes
    plt.show()

    result = kernel.fft_convolution(image, kernelcore)

    plt.imshow(result)
    plt.axis('off')  # Hide the axes
    plt.show()