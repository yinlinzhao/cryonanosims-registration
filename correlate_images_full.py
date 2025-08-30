#!/usr/bin/env python

import argparse
import modify_image as mod
import img_viewer_napari
import nanosims_registration as nano_reg
import fourier_display
import cv2
import numpy as np
import os

###---------------------------- COMMAND PROMPT INTERFACE ----------------------------###
##----- COMMENT OUT THIS SECTION IF YOU WANT TO EDIT VALUES FROM PYTHON DIRECTLY -----##
# parser = mod.make_parser()
# args = parser.parse_args()
#
# NanoSIMS_path = args.NanoSIMS
# TEM_path = args.SearchMap
# lamellae_path = args.Tomogram
#
# output_path = "nanosims_registration_results"
# if args.output_path:
#     output_path = args.output_path
#
# file_name = "result"
# if args.file_name:
#     file_name = args.file_name
#
# flip_vertical = False
# if args.flip_vertical:
#     flip_vertical = True
#
# show_steps = False
# if args.show_steps:
#     flip_vertical = True
#
# select_region = False
# if args.select_region:
#     select_region = True
#
# blur_lamellae = False
# if args.blur_lamellae:
#     blur_lamellae = True
#
# canny_threshold_max = 50
# if args.canny_threshold_max:
#     canny_threshold_max = args.canny_threshold_max
#
# canny_threshold_min = 40
# if args.canny_threshold_min:
#     canny_threshold_min = args.canny_threshold_min
#
# searchmap_blur_intensity = 9
# if args.searchmap_blur_intensity:
#     searchmap_blur_intensity = args.search_blur_intensity
#
# kernel = 0
# if args.kernel:
#     kernel = args.kernel
#
# border_size = 50
# if args.border_size:
#     border_size = args.border_size
#
# template_match_steps = 20
# if args.template_match_steps:
#     template_match_steps = args.template_match_steps

###--------------------------- END COMMAND PROMPT SECTION ---------------------------###

###--------- CONSTANTS (UNCOMMENT THIS SECTION TO RUN DIRECTLY FROM PYTHON) ---------###

file_name = "result"
output_path = "nanosims_registration_results"

NanoSIMS_path = 'images/GreenHydra_2_NanoSIMS.tif' # replace these with your own file paths
TEM_path = 'images/GreenHydra_2_SEM_binned.tif'
lamellae_path = 'images/GreenHydra_2_lamellae.tif'
flip_vertical = False
show_steps = False

#multiscale cross correlation variables
select_region = False
canny_threshold_max = 50
canny_threshold_min = 40
searchmap_blur_intensity = 9 # for blurring the searchmap prior to edge isolation
kernel = 0 # kernel used in canny edge search
border_size = 20 # must be positive. smaller values make the border thicker.
template_match_steps = 20

#high res tomogram correlation variables
blur_lamellae = False # Whether or not to apply blurring to high-res cryo-ET image

###----------------------------------- END SECTION -----------------------------------###

identity_matrix = np.float32([
    [1, 0, 0],
    [0, 1, 0]
])

#create output folder
try:
    os.mkdir(output_path)
    print(f"Directory '{output_path}' created successfully.")
except FileExistsError:
    print(f"Directory '{output_path}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{output_path}'.")
except Exception as e:
    print(f"An error occurred: {e}")

#run both programs
multiscale_translation = nano_reg.multiscale_cross_correlation(TEM_path, NanoSIMS_path, flip_vertical=flip_vertical,
                                                               border_size=border_size, select_region=select_region,
                                                               show_steps=show_steps, canny_threshold_max=canny_threshold_max,
                                                               canny_threshold_min=canny_threshold_min,
                                                               blur_intensity=searchmap_blur_intensity, kernel=kernel,
                                                               template_match_steps=template_match_steps)
lamellae_transform = nano_reg.highres_correlation(lamellae_path, TEM_path, flip_vertical=flip_vertical, blur=blur_lamellae,
                                                  show_steps=show_steps)

#Show final result
TEM_img = cv2.imread(TEM_path, cv2.IMREAD_COLOR_RGB)
lamellae_img = cv2.imread(lamellae_path, cv2.IMREAD_COLOR_RGB)
NanoSIMS_img = cv2.imread(NanoSIMS_path, cv2.IMREAD_COLOR_RGB)
if flip_vertical:
    TEM_img = cv2.flip((TEM_img), 0)

# Add padded border to NanoSIMS image using the dimensions returned from multiscale_translation
nanoSIMS_height, nanoSIMS_width = NanoSIMS_img.shape[:2]

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
lamellae_to_NanoSIMS = cv2.warpAffine(resized_lamellae, TEM_translation, (nanoSIMS_width, nanoSIMS_height))
TEM_to_NanoSIMS = cv2.warpAffine(resized_TEM, TEM_translation, (nanoSIMS_width, nanoSIMS_height))

#Open all three images in Napari
fourier_display.display_fourier_compare([
    [cv2.warpAffine(NanoSIMS_img, identity_matrix, (nanoSIMS_width, nanoSIMS_height)), NanoSIMS_img],
    [cv2.warpAffine(resized_TEM, identity_matrix, (nanoSIMS_width, nanoSIMS_height)), TEM_to_NanoSIMS]])

img_list = [lamellae_to_NanoSIMS, TEM_to_NanoSIMS, NanoSIMS_img]
img_viewer_napari.overlay_images_napari(img_list)

# save files
final_overlay = mod.overlay_images(lamellae_to_NanoSIMS, NanoSIMS_img, (0,0), 0.9)
output_file_name = output_path + "/" + file_name + ".tif"

cv2.imwrite(output_file_name, final_overlay)

information_file_name = output_path + "/" + file_name + "_information.txt"
with open(information_file_name, 'w') as file:
    file.write("result path:")
    file.write(information_file_name)

    file.write("\ntranslated TEM image by: ")
    file.write(str(multiscale_translation[0][:2]))

    file.write("\nlamellae affine transformation matrix: ")
    file.write(str(lamellae_transform[2]))
