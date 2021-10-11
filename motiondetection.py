# Python 3.8.2
# Computer Vision, Project 1
# Vanessa Hadlock, Stav Rones

import cv2
import numpy as np


# function that applies a 3x3 box filter to an image and returns the filtered image
# @params   img, the unfiltered image
#           filtered, the filtered image
# @returns  sigma_noise
def smallBoxFilter(img):
    # filtering the image with a 3x3 box filter
    filtered = cv2.boxFilter(img, -1, (3, 3))
    return filtered


# function that applies a 5x5 box filter to an image and returns the filtered image
# @params   img, the unfiltered image
#           filtered, the filtered image
def largeBoxFilter(img):
    # filtering the image with a 5x5 box filter
    filtered = cv2.boxFilter(img, -1, (5, 5))
    return filtered


# function that applies a 2D gaussian filter with sigma = 2.0 and returns the
# filtered image
# @params   img, the unfiltered image
#           filtered, the filtered image
def gaussianFilter(img):
    # filtering the image with a 2D Gaussian filter
    filtered = cv2.GaussianBlur(img, sigmaX=2.0, sigmaY=2.0)
    return filtered


# Function that estimates the noise in the images. It does this by doing the
# math described by the EST_NOISE algorithm to calculate the E_bar(i, j) and
# uses that to find the sigma(i, j) matrices for the first 8 images that are not
# moving in the chair scene
# this is the standard of noise to help determine the threshold for
# when an object is considered moving
# @params   totalnoise, sum of the noise at each pixel over the 10 images
#           n, the number of images for averaging purposes
# @returns  sigma_noise, the sigma(i, j) matrix of the noisy images
def est_noise(totalnoise, n):
    # creates empty matrices to put the sigma(i, j) calculations into
    # the images are 240 x 320 in size
    sigma_noise = np.empty((240, 320))

    # averages the total noise (this is the E_bar(i, j) value)
    average = totalnoise / n

    # estimate sigma(i,j) for noisy images
    for x in range(8):
        # reads in the 10 noisy images
        image_noisy = cv2.imread('images/Office/Greyscale/g_advbgst1_21_000{}.jpg'.format(x + 2))
        # truncates the 3rd column since it just has the RGB info
        image_noisy = image_noisy[:, :, 0]  # (256, 256)
        # subtracts each pixel in noisy image from the average pixel value
        IminusIk_noisy = average - image_noisy
        # squares it
        IminusIk_noisy = IminusIk_noisy ** 2
        # adds this to the running sum of the sigma(i,j) matrix for the noisy images
        sigma_noise = sigma_noise + IminusIk_noisy

    # divides this by 1/n-1 before taking the square root of each summed element
    sigma_noise = (1 / 7) * sigma_noise
    sigma_noise = np.sqrt(sigma_noise)

    return sigma_noise


# function that takes the temporal derivative of n-number of images using 0.5[-1, 0, 1]
# averaging filter. Allows it to be approximated by subtracting the pixels in the t-1 frame
# from the pixels in the t+1 frame to estimate the temporal derivative at time t
# the function than creates a  0 and 1 (binary) mask using the temporal derivative array
# and the inputted threshold
# @params   n, the number of images to take  the temporal derivative of
#           threshold, the determined threshold for the binary mask
# @returns  none
def temporalDerivative(n, threshold):

    columns = 320
    rows = 240

    for i in range(n):
        derivative_arr = np.empty([240, 320], dtype=int)
        # reads in the grayscale chair image, starts at img 10 since the scene does not change before that point
        # this is the t-1 point
        chair_minus1 = cv2.imread('images/Office/Greyscale/g_advbgst1_21_00{}.jpg'.format(i + 10))
        chair_minus1 = chair_minus1[:, :, 0]  # (240, 320)
        print("chair min 1 is: ", len(chair_minus1))

        fo3 = cv2.getGaussianKernel()

        # reads in the grayscale chair image at the t + 1 point
        chair_plus1 = cv2.imread('images/Office/Greyscale/g_advbgst1_21_00{}.jpg'.format(i + 12))
        chair_plus1 = chair_plus1[:, :, 0]  # (240, 320)
        print("chair plus 1 is: ", len(chair_plus1))

        # subtracts each pixel in I(t-1) image from each pixel in the I(t+1) image
        derivative_arr = np.subtract(chair_plus1, chair_minus1)

        print(type(derivative_arr))
        img = np.array(derivative_arr)

        print(type(derivative_arr))

        th, im_th = cv2.threshold(derivative_arr, threshold, 255, type=cv2.THRESH_BINARY_INV)

        cv2.imwrite('images/derivatives/threshold{}.jpg'.format(i+10), im_th)


def mask(n):

    for i in range(n):
        chair = cv2.imread('images/Office/Greyscale/g_advbgst1_21_00{}.jpg'.format(i + 11))
        chair_mask = cv2.imread('images/derivatives/threshold{}.jpg'.format(i + 10))

        masked = cv2.bitwise_and(chair, chair_mask)
        cv2.imwrite('images/motiondetection/masked{}.jpg'.format(i+11), masked)


def main():

    # defines number of frames needed
    n = 50
    threshold = 37
    temporalDerivative(n, threshold)

    mask(n)

    # creating empty array of the noise to keep track of the sum
    totalnoise = np.empty((240, 320))

    for x in range(8):
        # reads in the first 8 grayscale chair images in order to compute the average
        # noise at each pixel to help determine the threshold for the moving images
        chair = cv2.imread('images/Office/Greyscale/g_advbgst1_21_000{}.jpg'.format(x + 2))
        chair = chair[:, :, 0]  # (240, 240)
        # keeping track of the sum of the noise at each pixel value used to
        # calculate noise
        totalnoise = totalnoise + chair

    # runs the EST_NOISE algorithm to return the sigma(i, j) of the first 10
    # images to use as a baseline for the threshold & what is considered noise
    sigma_noise = est_noise(totalnoise, 8)

    # averages the sigma(i, j) matrix to get the estimated noise, sigma
    avg_noise = np.average(sigma_noise)
    # print("the estimation of the average noise of the filtered images is: ", avg_noise)

    # threshold is anything about 3*sigma of the noise in non-moving images
    threshold = avg_noise * 3
    print(threshold)


if __name__ == '__main__':
    print('Start of the program')
    main()
