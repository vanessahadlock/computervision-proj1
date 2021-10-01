# Python 3.8.2
#
# Converts all images to greyscale
# Computer Vision, Project 1
# Stav Rones, Vanessa Hadlock

from genericpath import isdir
from PIL import Image
import os


#########################################################################################################
# Function that takes in the image from the old color directory, converts an image to greyscale, and
# returns it to the new greyscale directory
# params:	oldPath, newPath
# returns:	none
#########################################################################################################
def convert2grey(oldPath, newPath):
    Image.open(oldPath).convert('L').save(newPath)


# Define image locations for Office & ReChair images, both greyscale and color
office_color_dir = './images/Office/Color'
chair_color_dir = './images/RedChair/Color'
office_grey_dir = './images/Office/Greyscale'
chair_grey_dir = './images/Office/Greyscale'

# Empty the existing Grayscale directories
print('Emptying existing greyscale image directories...')

# If the Office greyscale directory exists, delete every image in the directory
if os.path.isdir(office_grey_dir):
    for g_image in os.listdir(office_grey_dir):
        os.remove(f'{office_grey_dir}/{g_image}')

# If the directory doesn't exist, make the Office greyscale directory
else:
    os.mkdir(office_grey_dir)

# If the RedChair greyscale directory exists, delete every image in the directory
if os.path.isdir(chair_grey_dir):
    for g_image in os.listdir(chair_grey_dir):
        os.remove(f'{chair_grey_dir}/{g_image}')

# If the directory doesn't exist, make the RedChair greyscale directory
else:
    os.mkdir(chair_grey_dir)


# Convert office images to greyscale
print('Converting all Office images to greyscale...')

for color_image in os.listdir(office_color_dir):
    oldPath = f'{office_color_dir}/{color_image}'
    newPath = f'{office_grey_dir}/g_{color_image}'
    convert2grey(oldPath, newPath)

# Convert chair images to greyscale
print('Converting all Chair images to greyscale...')

for color_image in os.listdir(chair_color_dir):
    oldPath = f'{chair_color_dir}/{color_image}'
    newPath = f'{chair_grey_dir}/g_{color_image}'
    convert2grey(oldPath, newPath)
