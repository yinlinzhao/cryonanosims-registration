#!/usr/bin/env python

import modify_image as mod
import cv2
import numpy as np
import img_viewer_napari as open_napari
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

#FUNCTIONS

# runMatchTemplate(TEM, NanoSIMS, matchType=cv2.TM_CCORR)
# Input: template image array, base image array, OpenCV correlation score type
# Returns: (float maximum correlation score, (max x coordinate, max y coordinate))
#
# Calls the OpenCV function matchTemplate() on the TEM and NanoSIMS images
def runMatchTemplate(template_img, base_img, matchType=cv2.TM_CCORR):
    result = cv2.matchTemplate(template_img, base_img, matchType, None)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return (max_val, max_loc)

# multiscale_template_match(base_img, template_img, steps=20, range=(0.1, 1.0))
# Input: base image array, template image array, int number of increments for the scale factor, float range of scale factors
# Returns: ((max x coordinate, max y coordinate), (new height, new width))*****
#
# Iterates through a series of scale factors within the specified range, resizing the template_img
# by this scale factor, running runMatchTemplate() on these images and saving the function returns, then
# returns the largest correlation score and location.
def multiscale_template_match(base_img, template_img, steps=20, range=(0.1, 1.0)):
    max_vals = []
    max_intervals = []
    max_locs = []

    for scale in np.linspace(range[0], range[1], steps):
        template_re_dimension = mod.re_dimension(template_img, base_img.shape[0] * scale)
        template_resized = cv2.resize(template_img, template_re_dimension, interpolation=cv2.INTER_CUBIC)

        max_val, max_l = runMatchTemplate(template_resized, base_img)
        max_vals.append(max_val)
        max_intervals.append(base_img.shape[0] * scale)
        max_locs.append(max_l)

        print("scale:", scale)
        print("maximum value:", max_val)
        print("-------------------")

    max_index = max_vals.index(max(max_vals))
    max_loc = max_locs[max_index]
    template_re_dimension = mod.re_dimension(template_img, max_intervals[max_index])

    print("result: ", max(max_vals))
    return (max_loc, template_re_dimension)

# multiscale_cross_correlation(TEM_path, NanoSIMS_path, flip_vertical=False, canny_threshold_max=50, canny_threshold_min=40, blur_intensity=9, kernel=0, border_size=20, show_steps=False)
# Input: string path to TEM image, string path to NanoSIMS image, boolean flip_vertical, int maximum threshold, int minimum threshold, int blur intensity, int kernel, int border size, boolean show steps
# Returns: ((translation of x coordinate, translation of y coordinate), re_dimension, size of border height, size of border width)
#
# Loads both TEM and NanoSIMS images as grayscale images, normalizes values of both images, and converts them to grayscale.
# Uses isolate_edge_canny() to get the edge isolation and selectively draws contours over the edges to further reduce noise.
# Pads out the borders of the NanoSIMS edge image and runs multiscale_template_match() on it and the TEM edge, saving the
# returned value and visualizing the results for optional display
def multiscale_cross_correlation(TEM_path, NanoSIMS_path, flip_vertical=False, canny_threshold_max=50, canny_threshold_min=40, blur_intensity=9, kernel=0, border_size=20, select_region=False, show_steps=False):
    # LOAD IMAGES, FLIP--------------
    TEM_img = cv2.imread(TEM_path, cv2.IMREAD_GRAYSCALE)  # queryImage
    NanoSIMS_img = cv2.imread(NanoSIMS_path, cv2.IMREAD_GRAYSCALE)  # trainImage
    if flip_vertical:
        TEM_img = cv2.flip(TEM_img, 0)

    # NORMALIZE BOTH IMAGES---------------
    NanoSIMS_img = cv2.normalize(NanoSIMS_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    TEM_img = cv2.normalize(TEM_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # convert numbers back into 8bit grayscale for edge isolation
    NanoSIMS_img = (NanoSIMS_img * 255).clip(0, 255).astype(np.uint8)
    TEM_img = (TEM_img * 255).clip(0, 255).astype(np.uint8)

    # ISOLATE EDGES---------------
    TEM_edge = mod.isolate_edge_canny(TEM_img, blur_intensity, canny_threshold_min, canny_threshold_max, kernel)
    NanoSIMS_edge = mod.isolate_edge_canny(NanoSIMS_img, blur_intensity, canny_threshold_min, canny_threshold_max, kernel)

    # find contours for each image to further reduce noise
    TEM_edge = cv2.adaptiveThreshold(TEM_edge, 10, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 50)
    contours, hierarchy = cv2.findContours(TEM_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if(w/h > 0.5):
            cv2.drawContours(TEM_edge, [cnt], -1, (255,255,255), 2)

    NanoSIMS_edge = cv2.adaptiveThreshold(NanoSIMS_edge, 10, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 50)
    contours2, hierarchy2 = cv2.findContours(NanoSIMS_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours2:
        x,y,w,h = cv2.boundingRect(cnt)
        if(w/h > 0.5):
            cv2.drawContours(NanoSIMS_edge, [cnt], -1, (255,255,255), 2)

    if show_steps:
        # SHOW IMAGE OUTLINES
        mod.show_image('TEM', TEM_edge)
        mod.show_image('NanoSIMS', NanoSIMS_edge)
        cv2.waitKey(0)

    # RUNNING MULTISCALE TEMPLATE MATCHING----------------
    # find size of NanoSIMS image
    NanoSIMS_size = (NanoSIMS_edge.shape[1], NanoSIMS_edge.shape[0]) #width, height
    restricted_NanoSIMS = NanoSIMS_edge
    rv = [[0,0],[0,0],[0,0],[0,0]]

    if select_region:
        # restrict region to search in
        restricted_region = open_napari.select_roi_in_image(NanoSIMS_edge)
        rv = open_napari.get_roi_values_from_selection(restricted_region, NanoSIMS_edge)
        restricted_NanoSIMS = open_napari.create_roi(NanoSIMS_edge, rv[2][1], rv[2][0], rv[0], rv[1])

    border_height = int(NanoSIMS_size[1]/border_size)
    border_width = int(NanoSIMS_size[0]/border_size)
    # add padding to NanoSIMS image for template matching
    padded_NanoSIMS = cv2.copyMakeBorder(restricted_NanoSIMS, border_height, border_height,
                                                              border_width, border_width,
                                                              cv2.BORDER_CONSTANT, value=[0,0,0])

    # run template matching
    match_results = multiscale_template_match(padded_NanoSIMS, TEM_edge)

    TEM_re_dimension = match_results[1]
    max_loc = match_results[0] #(x,y) top left coordinates with padded sides and restricted
    print("max_loc", max_loc)
    print("rv:", rv[2][0], rv[2][1])
    adjusted_x = max_loc[0] + rv[2][0] - border_width
    adjusted_y = max_loc[1] + rv[2][1] - border_height
    rough_translate = (adjusted_x, adjusted_y)
    print("rough translation:", rough_translate)
    translation_matrix = np.float32([[1, 0, rough_translate[0]],
                                     [0, 1, rough_translate[1]]])

    if show_steps:
        # draw rectangle over matched area
        bottom_right = (rough_translate[0] + TEM_re_dimension[0], rough_translate[1] + TEM_re_dimension[1])
        cv2.rectangle(NanoSIMS_edge, rough_translate, bottom_right, 255, 5)

        # Display the result
        mod.show_image('Matched Area', NanoSIMS_edge)

        # draw rectangle over matched area
        bottom_right_padded = (max_loc[0] + TEM_re_dimension[0], max_loc[1] + TEM_re_dimension[1])
        cv2.rectangle(padded_NanoSIMS, max_loc, bottom_right_padded, 255, 5)

        # Display the result
        mod.show_image('Matched Area - with padding', padded_NanoSIMS)
        cv2.waitKey(0)

        overlay = cv2.imread(TEM_path, cv2.IMREAD_COLOR_BGR)  # queryImage
        if flip_vertical:
            overlay = cv2.flip(overlay, 0)
        overlay = cv2.resize(overlay, TEM_re_dimension, interpolation=cv2.INTER_CUBIC)
        background = cv2.imread(NanoSIMS_path, cv2.IMREAD_UNCHANGED)  # trainImage

        new_overlay = cv2.warpAffine(overlay, translation_matrix, (background.shape[1], background.shape[0]))

        image_stack = [new_overlay, background]
        open_napari.overlay_images_napari(image_stack)

    return (rough_translate, TEM_re_dimension, (border_height, border_width), translation_matrix)

# histogram_normalize(img)
# Input: image array
# Returns: normalized image array
#
# Calculates the histogram of the input image and equalizes the values, returning the histogram-equalized image
def histogram_normalize(img):
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()

    #equalize histogram
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    return cdf[img]

# highres_correlation(lamellae_path, SearchMap_path, flip_vertical=False, blur=False, blur_intensity=11, show_steps=False)
# Input: string path to lamellae image, string path to SearchMap/TEM image, boolean flip vertical, boolean blur, int blur intensity, boolean show steps
# Returns: (translated lamellae image array, translated lamellae overlaid over TEM image array, affine transformation matrix used)
#
# Loads both lamellae and TEM images as grayscale, normalizes the values and converts them to 8-bit, and calls
# histogram_normalize() on the TEM image, opens a matplotlib setup for finding regions of interest and selecting
# corresponding keypoints, calculates a 2D (2x3) affine matrix for the lamellae and applies the transformation.
# Optionally visualizes the results.
def highres_correlation(lamellae_path, SearchMap_path, flip_vertical=False, blur=False, blur_intensity=11, show_steps=False):
    # LOAD IMAGES, FLIP IF NEEDED
    lam_img = cv2.imread(lamellae_path, cv2.IMREAD_GRAYSCALE)  # queryImage
    map_img = cv2.imread(SearchMap_path, cv2.IMREAD_GRAYSCALE)  # trainImage
    if flip_vertical:
        map_img = cv2.flip(map_img, 0)

    #NORMALIZE BOTH IMAGES
    lam_img = cv2.normalize(lam_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    map_img = cv2.normalize(map_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    #convert numbers back into 8bit grayscale
    lam_img = (lam_img * 255).clip(0, 255).astype(np.uint8)
    map_img = (map_img * 255).clip(0, 255).astype(np.uint8)

    print("lamellae value range:", (np.min(lam_img), np.max(lam_img)))
    print("map value range:", (np.min(map_img), np.max(map_img)))

    #apply gaussian blur to lamellae image if needed
    if blur:
        lam_img = mod.gaussian_blur(lam_img, blur_intensity)

    #EQUALIZE SEARCHMAP FOR VISIBILITY
    map_img = histogram_normalize(map_img)

    if show_steps:
        mod.show_image('equalized histogram', map_img)
        cv2.waitKey(0)

    # #OPEN POINT SELECTION WINDOW

    # #these set up what happens upon these mouse events
    # map_points = []
    # lam_points = []
    #
    # def on_click(event):
    #     if event.button is MouseButton.LEFT:
    #         if event.inaxes == ax1:
    #             map_points.append([event.xdata, event.ydata])
    #             ax1.plot([event.xdata], [event.ydata], marker='*', markersize=5)
    #             plt.show()
    #
    #             print(f'axis1 pixel coords {event.xdata} {event.ydata}')
    #         elif event.inaxes == ax2:
    #             lam_points.append([event.xdata, event.ydata])
    #             ax2.plot([event.xdata], [event.ydata], marker='*', markersize=5)
    #             plt.show()
    #
    #             print(f'axis2 pixel coords {event.xdata} {event.ydata}')
    #
    #     # if event.button is MouseButton.RIGHT:
    #     #     print('disconnecting callback')
    #     #     print("map points: ", map_points)
    #     #     print("lamellae points: ", lam_points)
    #     #     plt.disconnect(click_id)
    #     #
    #     #     plt.suptitle("close this window")
    #
    # def on_press(event):
    #     print("you pressed: ", event.key)
    #     if event.key == 'enter':
    #         print("hi you pressed enter")
    #         print("okay connected to onclick")
    #         plt.suptitle("select corresponding points, close out of window when done")
    #         plt.show()
    #         click_id = plt.connect('button_press_event', on_click)
    #         plt.show()
    #         plt.disconnect(key_id)
    #         return click_id
    #
    # ax1 = plt.subplot(1,2,1)
    # plt.imshow(map_img)
    #
    # ax2 = plt.subplot(1,2,2)
    # plt.imshow(lam_img)
    #
    # # key_id = plt.connect('key_press_event', on_press)
    # key_id = plt.connect('button_press_event', on_click)
    # plt.suptitle("use magnifying glass to select region, then press enter")
    #
    # plt.show()
    window_results = mod.create_window(map_img, lam_img)
    map_points = window_results[0]
    lam_points = window_results[1]

    # LOAD IMAGES IN COLOR
    lam_img = cv2.imread(lamellae_path, cv2.IMREAD_COLOR_RGB)  # queryImage
    map_img = cv2.imread(SearchMap_path, cv2.IMREAD_COLOR_RGB)  # trainImage
    if flip_vertical:
        map_img = cv2.flip(map_img, 0)
    map_img = histogram_normalize(map_img)

    #AFFINE TRANSFORMATION
    map_points = np.array(map_points)
    lam_points = np.array(lam_points)

    transform_matrix = cv2.estimateAffine2D(lam_points, map_points)
    transform_matrix = transform_matrix[0]
    print("transformation matrix using estimateAffine2D: ", transform_matrix)

    # Get the dimensions of the image
    height, width = map_img.shape[:2]

    # Apply transform
    translated_image = cv2.warpAffine(lam_img, transform_matrix, (width, height))

    # Display the result
    if show_steps:
        mod.show_image('Translated Image', translated_image)
        cv2.waitKey(0)

    lam_corr_overlay = mod.overlay_images(translated_image, map_img, (0,0), 0.9)

    return (translated_image, lam_corr_overlay, transform_matrix) #translated image, overlays, transformation matrix
