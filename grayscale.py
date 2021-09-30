# Python 3.8.2
#
# Converts all images to greyscale
#
# Stav Rones, Vanessa Hadlock

from genericpath import isdir
from PIL import Image 
import os

# Converts an image to greyscale
def convert2grey(oldPath, newPath):
    Image.open(oldPath).convert('L').save(newPath)

# Define image locations
office_color_dir = './images/Office/Color'
chair_color_dir = './images/RedChair/Color'
office_grey_dir = './images/Office/Greyscale'
chair_grey_dir = './images/Office/Greyscale'

# Empty existing Grayscale directories
print('Emptying existing greyscale image directories...')

if (os.path.isdir(office_grey_dir)):
    for g_image in os.listdir(office_grey_dir):
        os.remove(f'{office_grey_dir}/{g_image}')
else:
    os.mkdir(office_grey_dir)
    
if (os.path.isdir(chair_grey_dir)):
    for g_image in os.listdir(chair_grey_dir):
        os.remove(f'{chair_grey_dir}/{g_image}')
else:
    os.mkdir(chair_grey_dir)


# Convert office images to greyscale
print('Converting all Office images to greyscale...')

for color_image in os.listdir(office_color_dir):
    oldPath = f'{office_color_dir}/{color_image}'
    newPath = f'{office_grey_dir}/g_{color_image}'
    convert2grey(oldPath,newPath)

# Convert chair images to greyscale
print('Converting all Chair images to greyscale...')

for color_image in os.listdir(chair_color_dir):
    oldPath = f'{chair_color_dir}/{color_image}'
    newPath = f'{chair_grey_dir}/g_{color_image}'
    convert2grey(oldPath,newPath)
