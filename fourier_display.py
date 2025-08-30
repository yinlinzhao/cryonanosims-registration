#!/usr/bin/env python

import argparse
import modify_image as mod
import cv2
import numpy as np
import matplotlib.pyplot as plt

# fourier_transform(img, shift=False)
# Input: image array, boolean shift
# Returns: fourier transformed image array
#
# Converts input image to grayscale, then calls OpenCV's dft() function on it and adjusts the values for visualization.
# Optionally uses np.fft.fftshift() to shift the resulting image to the center as well
def fourier_transform(img, shift=False):
    #convert image to grayscale
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Use discrete fourier transform
    img_dft = cv2.dft(np.float32(img_grayscale), flags=cv2.DFT_COMPLEX_OUTPUT)
    img_dft = 20*np.log(cv2.magnitude(img_dft[:,:,0], img_dft[:,:,1]))

    if shift:
        #Use fftshift to center the image
        img_dft_shift = np.fft.fftshift(img_dft)
        return img_dft_shift

    return img_dft

# display_fourier_compare(images)
# Input: nx2 array of image arrays, the first item in each row is the original image, the second item is the translated image
# Returns:
#
# Displays a matplotlib plot showing the original image, fourier transform of the original and translated images
def display_fourier_compare(images):
    row = 1
    for sets in images:
        plt.subplot(len(images), 3, row)
        plt.imshow(sets[0])

        fft_original = fourier_transform(sets[0])
        plt.subplot(len(images), 3, row + 1)
        plt.imshow(fft_original)

        fft_shifted = fourier_transform(sets[1])
        plt.subplot(len(images), 3, row + 2)
        plt.imshow(fft_shifted)

        row += 3
    plt.show()
