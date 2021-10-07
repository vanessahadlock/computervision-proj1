from PIL import Image 
import numpy as np

numImages = 10
imageSize = 256
g_color = 128

# Generate grey image
greyImage_arr = np.full((imageSize, imageSize), g_color)

# ------- USE TO DISPLAY ARRAY AS GREYSCALE IMAGE ------
Image.fromarray(np.uint8(greyImage_arr)).show()
# ------------------------------------------------------

# Generate 10 noisy images corrupt array with additive zero-mean Gaussian noise with standard deviation 2.0
# 10 times and save
noisyImages = []
for i in range(numImages):
    mu = 0
    sigma = 2
    gauss_arr = np.random.normal(mu,sigma,(imageSize,imageSize))
    noisyImages.append(greyImage_arr + gauss_arr)

# Use EST NOISE procedures to estimate the noise in the images
pixel_avg_arr = np.zeros((256,256))
for i in range(256):
    for j in range(256):
        
        pixel_sum = 0
        for image in range(numImages):
            pixel_sum += noisyImages[image][i][j]
        
        pixel_avg_arr[i][j] = pixel_sum / numImages

sigma_arr = np.zeros((256,256))
for i in range(256):
    for j in range(256):
        
        sum = 0
        for image in range(numImages):
            sum += (pixel_avg_arr[i][j] - noisyImages[image][i][j]) ** 2

        sigma_arr[i][j] = (sum / numImages) ** 1/2

# Filter with a 3x3 box filter

# Use EST NOISE procedures to estimate the noise of filtered images