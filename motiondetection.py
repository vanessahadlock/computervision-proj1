# Python 3.8.2
# Computer Vision, Project 1
# Stav Rones, Vanessa Hadlock

from genericpath import isdir
from PIL import Image
import os
import cv2
import numpy as np


# function that computes & returns the gradient in the x & y direction of an image
# @params   image, image that needs the gradients computed
# @returns  gradient_x, x-gradient
# @returns  gradient_y, y-gradient
def compute_gradient(image):
    sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    gradient_x: np.ndarray = cv2.filter2D(image, -1, sobel_x, borderType=cv2.BORDER_CONSTANT)
    gradient_y: np.ndarray = cv2.filter2D(image, -1, sobel_y, borderType=cv2.BORDER_CONSTANT)

    return gradient_x, gradient_y


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


def main():

    for x in range(41):
        # reads in the grayscale chair image, starts at img 10 since the scene does not change before that point
        chair = cv2.imread('images/Office/Greyscale/g_advbgst1_21_00{}.jpg'.format(x + 10))

        # truncates the 3rd column since it just has the RGB info & these are grayscale
        chair = chair[:, :, 0]  # (256, 256)

        # computes the x and y gradients at each pixel
        [grad_x, grad_y] = compute_gradient(chair)

    # creating empty array of the noise to keep track of the sum
    totalnoise = np.empty((240, 320))

    for x in range(8):
        # reads in the first 10 grayscale chair images in order to compute the average
        # noise at each pixel to help determine the threshold for the moving images
        chair = cv2.imread('images/Office/Greyscale/g_advbgst1_21_000{}.jpg'.format(x + 2))
        chair = chair[:, :, 0]  # (256, 256)
        # keeping track of the sum of the noise at each pixel value used to
        # calculate noise
        totalnoise = totalnoise + chair

    # runs the EST_NOISE algorithm to return the sigma(i, j) of the first 10
    # images to use as a baseline for the threshold & what is considered noise
    sigma_noise = est_noise(totalnoise, 8)

    # averages the sigma(i, j) matrix to get the estimated noise, sigma
    avg_noise = np.average(sigma_noise)
    print("the estimation of the average noise of the filtered images is: ", avg_noise)

    # threshold is anything about 3*sigma of the noise in non-moving images
    threshold = avg_noise * 3
    print(threshold)


if __name__ == '__main__':
    print('Start of the program')
    main()
