from PIL import Image 
import cv2
import numpy as np

# Estimate the amount of noise given a list of ndarrays of equal size
def EST_NOISE(img_arr: list):

    numImages = len(img_arr)
    w = len(img_arr[0])
    h = len(img_arr[0][0])

    pixel_avg_arr = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            
            pixel_sum = 0
            for image in img_arr:
                pixel_sum += image[i][j]
            
            pixel_avg_arr[i][j] = pixel_sum / numImages

    sigma_arr = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            
            sum = 0
            for image in img_arr:
                sum += (pixel_avg_arr[i][j] - image[i][j]) ** 2

            sigma_arr[i][j] = (sum / numImages) ** 1/2

    return round(np.average(sigma_arr), 3)

# main
def main():

    numImages = 10
    imageSize = 256
    g_color = 128

    mu = 0
    sigma = 2

    # Generate grey image
    greyImage_arr = np.full((imageSize, imageSize), g_color)

    # Generate 10 noisy images corrupt array with additive zero-mean Gaussian noise with standard deviation 2.0
    noisyImages = []
    for _ in range(numImages):
        gauss_arr = np.random.normal(mu,sigma,(imageSize,imageSize))
        noisyImages.append(greyImage_arr + gauss_arr)

    print(f'Average sigma value for noisy images: {EST_NOISE(noisyImages)}')

    # Filter the 10 noisy images with a 3x3 averaging box filter
    filteredImages = []
    for noisyImage in noisyImages:
        filteredImage = cv2.boxFilter(noisyImage,-1,(3,3))
        filteredImages.append(filteredImage)

    print(f'Average sigma value for filtered images: {EST_NOISE(filteredImages)}')
    
if __name__ == "__main__":
    main()