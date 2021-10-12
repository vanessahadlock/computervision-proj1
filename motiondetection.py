# Python 3.8.2
# Computer Vision, Project 1
# Vanessa Hadlock, Stav Rones

from genericpath import isfile
import numpy as np
import ffmpeg
import cv2
import sys
import os

def filterImage(filter, img):
    if (filter == 'box3'):
        return cv2.boxFilter(img, -1, (3, 3))

    if (filter == 'box5'):
        return cv2.boxFilter(img, -1, (5, 5))

    if (filter == 'gauss'):
        sigma = 2
        size = int(np.ceil(5 * sigma))
        if (size % 2 == 0):
            size += 1
        return cv2.GaussianBlur(img,(size,size),sigma)

    return img

def calculateMotionThreshold(path, numImages, filter, debug):

    print(f'Calculating gradient threshold for motion of office image set...\n')

    filenames = os.listdir(path)

    # Get sample image to find its shape
    sample_img: np.ndarray = cv2.imread(f'{path}/{filenames[0]}', cv2.IMREAD_GRAYSCALE)
    sample_img = filterImage(filter, sample_img)

    # Iterate first numImages images to get average value of each pixel
    pixel_sum = np.zeros(sample_img.shape)
    for i in range(numImages):

        # read in image as greyscale
        img: np.ndarray = cv2.imread(f'{path}/{filenames[i]}', cv2.IMREAD_GRAYSCALE) 

        # filter image
        img = filterImage(filter, img)
            
        # add to pixel sum
        pixel_sum += img
    pixel_avg = pixel_sum / numImages

    # Iterate first numImages images to sigma of each pixel
    sigma_sum = np.zeros(sample_img.shape)
    for j in range(numImages):

        # read in image as greyscale
        img2: np.ndarray = cv2.imread(f'{path}/{filenames[j]}', cv2.IMREAD_GRAYSCALE) 

        # filter image
        img2 = filterImage(filter, img2)

        # calculate sigma array
        sigma_sum += (pixel_avg - img2) ** 2

    sigma = (sigma_sum / numImages) ** 0.5

    return round(np.average(sigma * 3), 3)

def simpleTemporalDerivative(path, outpath, threshold, filter, debug):
    
    print(f"Saving masked images to {outpath}...\n")

    if os.path.isdir(outpath):
        for filename in os.listdir(path):
            if(os.path.isfile(f'{outpath}/{filename}')):
                os.remove(f'{outpath}/{filename}')
    else:
        os.mkdir(outpath)

    filenames = sorted(os.listdir(path))
    
    for i in range(1, len(filenames) - 1):

        # read in images as greyscale
        img_minus1 = cv2.imread(f'{path}/{filenames[i - 1]}', cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(f'{path}/{filenames[i]}')
        img_plus1 = cv2.imread(f'{path}/{filenames[i + 1]}', cv2.IMREAD_GRAYSCALE)

        if (debug):
            print(f'img_minus1 from {path}/{filenames[i - 1]}:\n{img_minus1}\n')
            print(f'img_plus1 from {path}/{filenames[i + 1]}:\n{img_plus1}\n')

        # filter images
        img_minus1 = filterImage(filter, img_minus1)
        img_plus1 = filterImage(filter, img_plus1)

        if (debug):
            print(f'img_minus1 filtered:\n{img_minus1}\n')
            print(f'img_plus1 filtered:\n{img_plus1}\n')

        derivative_img = np.subtract(img_plus1, img_minus1) / 2

        if (debug):
            print(f'derivative_img:\n{derivative_img}\n')

        th, mask_img = cv2.threshold(derivative_img, threshold, 255, type=cv2.THRESH_BINARY)

        if (debug):
            print(f'mask_img:\n{mask_img}\n')

        row, col = mask_img.shape
        for r in range(row):
            for c in range(col):
                if (mask_img[r][c] == 255):
                    img[r][c][0] = 0
                    img[r][c][1] = 255
                    img[r][c][2] = 0

        cv2.imwrite(f'{outpath}/{filenames[i]}', mask_img)

    return

def gaussTemporalDerivative(path, outpath, sigma, threshold, filter, debug):
    
    print("Calculating gaussian temporal derivative...\n")

    # remove images from motion directory
    if os.path.isdir(outpath):
        for filename in os.listdir(path):
            if os.path.isfile(f'{outpath}/{filename}'):
                os.remove(f'{outpath}/{filename}')
    else:
        os.mkdir(outpath)

    # compute gaussian temporal filter based on sigma
    gauss_len = int(np.ceil(5 * sigma))
    if (gauss_len % 2 == 0):
        gauss_len += 1

    gauss_filter: np.ndarray = cv2.getGaussianKernel(gauss_len,sigma, cv2.CV_32F)    
    half_gauss_len = int(np.floor(gauss_len/2))

    print(f'Gaussian temporal filter: \n{gauss_filter.transpose()}')

    # get test image for size
    filenames = os.listdir(path)
    w, h = cv2.imread(f'{path}/{filenames[0]}', cv2.IMREAD_GRAYSCALE).shape
    
    # for every image, calculate the gaussian temporal derivative
    for imageIndex in range(len(filenames) - half_gauss_len):

        # create 3D array of images
        imgs = np.empty((w, h, len(gauss_filter)))
        for count in range(len(gauss_filter)):
            img = cv2.imread(f'{path}/{filenames[imageIndex + count]}', cv2.IMREAD_GRAYSCALE)
            imgs[:,:,count] = filterImage(filter, img)

        # calculate gaussian image using temporal filter
        gauss_img = np.zeros((w, h))
        x, y, t = imgs.shape 
        for i in range(x):
            for j in range(y):
                for k in range(t):
                    gauss_img[i][j] += imgs[i][j][k] * gauss_filter[k]

        print('writing image')
        th, mask_img = cv2.threshold(gauss_img, threshold, 255, type=cv2.THRESH_BINARY_INV)

        row, col = mask_img.shape
        out_img = cv2.imread(f'{path}/{filenames[imageIndex + half_gauss_len]}')

        for r in range(row):
            for c in range(col):
                if (mask_img[r][c] == 255):
                    out_img[r][c][0] = 0
                    out_img[r][c][1] = 255
                    out_img[r][c][2] = 0

        cv2.imwrite(f'{outpath}/{filenames[imageIndex]}', out_img)

def main():

    ######################################################################
    ##################### Process command line args #####################
    ######################################################################

    filter = "none"
    gradient = "simple"
    debug = False

    for i in range(1, len(sys.argv)):

        arg = sys.argv[i]

        if (arg == '-filter=box3'):
            filter = 'box3'
        elif (arg == '-filter=box5'):
            filter = 'box5'
        elif (arg == '-filter=gauss'):
            filter = 'gauss'
        elif (arg == '-grad=gauss'):
            gradient = 'gauss'
        elif (arg == '-debug'):
            debug = True
        else:
            print(f"\ninvalid command line arg: {arg}")
            print(f"\nusage: python3 motiondetection.py [\"-debug\"] [\"-grad=gauss\"] [\"-filter=box3\"] [\"-filter=box5\"] [\"-filter=gauss\"]\n")
            return

    print(f"\nStarting program with filter= {filter} and gradient= {gradient}\n")

    ######################################################################
    ### Calculate mask threshold using 3 * avg noise of first 8 images ###
    ######################################################################

    office_threshold = calculateMotionThreshold("./images/Office", 8, filter, debug)
    print(f'\toffice_threshold = {office_threshold}\n')

    chair_threshold = calculateMotionThreshold("./images/RedChair", 8, filter, debug)
    print(f'\tchair_threshold = {chair_threshold}\n')

    ######################################################################
    ### Compute gradient images and map to binary based on threshold ###
    ######################################################################

    if (gradient == 'simple'):
        simpleTemporalDerivative('./images/RedChair', './motion/RedChair', office_threshold, filter, debug)
        simpleTemporalDerivative('./images/Office', './motion/Office', office_threshold, filter, debug)
    else:
        sigma = 1
        gaussTemporalDerivative('./images/Office', './motion/Office', sigma, office_threshold, filter, debug)
        gaussTemporalDerivative('./images/RedChair','./motion/RedChair', sigma, office_threshold, filter, debug)

    ######################################################################
    ##################### Convert image set into video #####################
    ######################################################################

    # path = './images/mask'
    # filenames = os.listdir(path)
    # img = cv2.imread(f'{path}/{filenames[0]}', cv2.IMREAD_GRAYSCALE)

    # out = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15, img.shape)
 
    # for i in range(len(filenames)):
    #     img = cv2.imread(f'{path}/{filenames[i]}')
    #     out.write(img)
    # out.release()

    return

if __name__ == '__main__':
    main()
