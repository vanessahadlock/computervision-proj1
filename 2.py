import numpy as np
import cv2

def main():

    # Generate a 2D Gaussian filter mask with standard deviation 1.4.  
    sigma = 1.4
    size = int(np.ceil(5 * sigma))
    if (size % 2 == 0):
        size += 1

    # Find two equivalent 1-D masks so that they can be used to smooth images applying the separability property.
    dirac_delta = np.zeros((size,size))
    midpt = int(np.floor(size/2))
    dirac_delta[midpt][midpt] = 1

    kernel: np.ndarray = cv2.GaussianBlur(dirac_delta,(size,size),1.4)
    
    print(f'2D Gaussian mask with sd = 1.4: \n{kernel.round(3)}')
    print(f'Equivalent 1D mask: \n{kernel.round(3)[0]}')


if __name__ == "__main__":
    main()