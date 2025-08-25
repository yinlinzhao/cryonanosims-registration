#!/usr/bin/env python

import argparse
import modify_image as mod
import cv2
import numpy as np
import matplotlib.pyplot as plt

def fourier_transform_full(img):
    plt.subplot(1, 3, 1)
    plt.imshow(img)

    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_dft = cv2.dft(np.float32(img_grayscale), flags=cv2.DFT_COMPLEX_OUTPUT)
    img_dft = 20*np.log(cv2.magnitude(img_dft[:,:,0], img_dft[:,:,1]))

    plt.subplot(1, 3, 2)
    plt.imshow(img_dft,cmap='gray')

    img_dft_shift = np.fft.fftshift(img_dft)
    plt.subplot(1, 3, 3)
    plt.imshow(img_dft_shift,cmap='gray')

    plt.show()

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

def display_fourier(NanoSIMS_img, TEM_img, final_img):
    fft_NanoSIMS = fourier_transform(NanoSIMS_img)
    fft_TEM = fourier_transform(TEM_img)
    fft_final = fourier_transform(final_img)

    plt.subplot(3, 2, 1)
    plt.imshow(NanoSIMS_img)
    plt.subplot(3, 2, 2)
    plt.imshow(fft_NanoSIMS)

    plt.subplot(3, 2, 3)
    plt.imshow(TEM_img)
    plt.subplot(3, 2, 4)
    plt.imshow(fft_TEM)

    plt.subplot(3, 2, 5)
    plt.imshow(final_img)
    plt.subplot(3, 2, 6)
    plt.imshow(fft_final)

    plt.show()

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

# final_registration = 'images/test_results/synthetic_data_searchmap_halved_3_full.png'
# test_image = cv2.imread(final_registration, cv2.IMREAD_COLOR_BGR)
# fourier_transform(test_image)
#
# path_2 = 'images/xiyao.tif'
# test_image_2 = cv2.imread(path_2, cv2.IMREAD_COLOR_BGR)
# fourier_transform(test_image_2)