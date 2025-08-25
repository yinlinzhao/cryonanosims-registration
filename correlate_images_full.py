#!/usr/bin/env python

import argparse
import modify_image as mod
import img_viewer_napari
import nanosims_registration as nano_reg
import fourier_display
import cv2
import numpy as np
import sys
# IN_DIR = sys.argv[1]
#
# parser = argparse.ArgumentParser()
# parser.add_argument("-o", "--Output", help = "Show Output")
#
# args = parser.parse_args()

file_name = 'C:/Users/ailee/Documents/coding/20250626_Lovric_Correlation/images/test_results/lovric_temp'
NanoSIMS_path = 'images/lovric2_Na.tif'
TEM_path = 'images/lovric2_TEM-3-binned-2.tif'
lamellae_path = 'images/lamellae_lovric2_TEM-3.tif'
flip_vertical = False
show_steps = False

canny_threshold_max = 50
canny_threshold_min = 40
searchmap_blur_intensity = 9
kernel = 0
border_size = 50

blur = False
lamellae_blur_intensity = 11
identity_matrix = np.float32([
    [1, 0, 0],
    [0, 1, 0]
])

########################################################
#run both programs
multiscale_translation = nano_reg.multiscale_cross_correlation(TEM_path, NanoSIMS_path, flip_vertical=flip_vertical, border_size=border_size, select_region=True)
lamellae_transform = nano_reg.highres_correlation(lamellae_path, TEM_path, flip_vertical=flip_vertical)

#Show final result
TEM_img = cv2.imread(TEM_path, cv2.IMREAD_COLOR_RGB)
lamellae_img = cv2.imread(lamellae_path, cv2.IMREAD_COLOR_RGB)
NanoSIMS_img = cv2.imread(NanoSIMS_path, cv2.IMREAD_COLOR_RGB)
if flip_vertical:
    TEM_img = cv2.flip((TEM_img), 0)

# Add padded border to NanoSIMS image using the dimensions returned from multiscale_translation
nanoSIMS_height, nanoSIMS_width = NanoSIMS_img.shape[:2]
NanoSIMS_img_padded = cv2.copyMakeBorder(NanoSIMS_img, multiscale_translation[2][0], multiscale_translation[2][0],
                                multiscale_translation[2][1], multiscale_translation[2][1],
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])

# Calculate transformation matrix (rigid) for lamellae and TEM
tx, ty = multiscale_translation[0][:2]
print("translate TEM image by:", tx, ty)
TEM_translation = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])

#Resize TEM and lamellae to match the size of the NanoSIMS image
resized_lamellae = cv2.resize(lamellae_transform[0], multiscale_translation[1], interpolation=cv2.INTER_CUBIC)
resized_TEM = cv2.resize(TEM_img, multiscale_translation[1], interpolation=cv2.INTER_CUBIC)

# Create translated versions of lamellae and TEM data
# lamellae_to_NanoSIMS = cv2.warpAffine(resized_lamellae, TEM_translation, (nanoSIMS_width, nanoSIMS_height))
# TEM_to_NanoSIMS = cv2.warpAffine(resized_TEM, TEM_translation, (nanoSIMS_width, nanoSIMS_height))
lamellae_to_NanoSIMS = cv2.warpAffine(resized_lamellae, TEM_translation, (nanoSIMS_width, nanoSIMS_height))
TEM_to_NanoSIMS = cv2.warpAffine(resized_TEM, TEM_translation, (nanoSIMS_width, nanoSIMS_height))

#Open all three images in Napari
fourier_display.display_fourier_compare([
    [cv2.warpAffine(NanoSIMS_img, identity_matrix, (nanoSIMS_width, nanoSIMS_height)), NanoSIMS_img_padded],
    [cv2.warpAffine(resized_TEM, identity_matrix, (nanoSIMS_width, nanoSIMS_height)), TEM_to_NanoSIMS]])

img_list = [lamellae_to_NanoSIMS, TEM_to_NanoSIMS, NanoSIMS_img_padded]
img_viewer_napari.overlay_images_napari(img_list)

final_overlay = mod.overlay_images(lamellae_to_NanoSIMS, NanoSIMS_img_padded, (0,0), 0.9)
output_file_name = file_name + "_full.tif"
cv2.imwrite(output_file_name, final_overlay)

information_file_name = file_name + "_information.txt"
with open(information_file_name, 'w') as file:
    # Write content to the file
    # file.write("padded border size (height): ")
    # file.write(str(multiscale_translation[2][0]))
    #
    # file.write("\npadded border size (width): ")
    # file.write(str(multiscale_translation[2][1]))
    file.write("result path:")
    file.write(information_file_name)

    file.write("\ntranslated TEM image by: ")
    file.write(str(multiscale_translation[0][:2]))

    file.write("\nlamellae affine transformation matrix: ")
    file.write(str(lamellae_transform[2]))
