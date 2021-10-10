# Python 3.8.2
# Computer Vision, Project 1
# Stav Rones, Vanessa Hadlock

from genericpath import isdir
from PIL import Image
import array

import os
import cv2
import numpy as np


def smallBoxFilter(img):
    # filtering the image with a 3x3 box filter
    filtered = cv2.boxFilter(img, -1, (3, 3))
    return filtered


def largeBoxFilter(img):
    # filtering the image with a 5x5 box filter
    filtered = cv2.boxFilter(img, -1, (5, 5))
    return filtered


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


#
# @params
# @returns
def temporalDerivative(n):

    columns = 320
    rows = 240
    derivative_arr = []

    for i in range(n):
        # reads in the grayscale chair image, starts at img 10 since the scene does not change before that point
        # this is the t-1 point
        temporal_der = 0
        chair_minus1: np.ndarray = cv2.imread('images/Office/Greyscale/g_advbgst1_21_00{}.jpg'.format(i + 10))
        chair_minus1 = chair_minus1[:, :, 0]  # (240, 320)
        print("chair min 1 is: ", chair_minus1)
        # reads in the grayscale chair image at the t + 1 point
        chair_plus1: np.ndarray = cv2.imread('images/Office/Greyscale/g_advbgst1_21_00{}.jpg'.format(i + 12))
        chair_plus1 = chair_plus1[:, :, 0]  # (240, 320)
        print("chair plus 1 is: ", chair_plus1)
        # subtracts each pixel in I(t-1) image from each pixel in the I(t+1) image

        derivative_arr.append([])

        for j in range(rows):
            derivative_arr[i].append([])

            for k in range(columns):
                difference = chair_plus1[i][j] - chair_minus1[i][j]
                derivative_arr[i][j].append(difference)

    print(derivative_arr)


def main():

    temporalDerivative(2)

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
    # print(threshold)


if __name__ == '__main__':
    print('Start of the program')
    main()
