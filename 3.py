import numpy as np
from PIL import Image
import cv2

def main():

    filter1 = np.array([1, 1, 1, 1, 1])/5
    filter2 = np.array([1, 2, 4, 2, 1])/10

    img = np.array([10, 10, 10, 10, 10, 40, 40, 40, 40, 40])
    
    filtered1: np.ndarray = cv2.filter2D(np.uint8(img),-1,filter1,borderType=cv2.BORDER_CONSTANT)

    cv2.filter2D(np.uint8(img),-1,filter1)

    filtered2: np.ndarray  = cv2.filter2D(np.uint8(img),-1,filter2,borderType=cv2.BORDER_CONSTANT)

    print(f'filtered1: \n{filtered1.transpose()[0]}')
    print(f'filtered2: \n{filtered2.transpose()[0]}')

if __name__ == "__main__":
    main()