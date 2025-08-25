# Adapted from code by

import napari
import numpy as np
import cv2
from napari.settings import get_settings

def select_points_in_two_images(image1, image2, print_output=False):
    viewer = napari.Viewer(title='Select points & close window to apply selection')

    settings = get_settings()
    settings.appearance.theme = 'dark'
    settings.application.grid_stride = 2

    # viewer.window.qt_viewer.dockLayerList.setVisible(False)
    # viewer.window.qt_viewer.dockLayerControls.setVisible(False)
    viewer.add_image(image1)
    points_layer_1 = viewer.add_points(None)
    points_layer_1.mode = 'add'

    viewer.add_image(image2)
    points_layer_2 = viewer.add_points(None)
    points_layer_2.mode = 'add'

    viewer.show(block=True)
    if print_output:
        print(f"Points coordinates are:\n{points_layer_1.data}")
        print(f"Points coordinates are:\n{points_layer_2.data}")

    selected_pts = {}
    for i in np.arange(len(points_layer_1.data), step=2):
        selected_pts[int(i / 2)] = points_layer_1.data[i], points_layer_1.data[i + 1]
    selected_pts_2 = {}
    for j in np.arange(len(points_layer_2.data), step=2):
        selected_pts_2[int(j / 2)] = points_layer_2.data[j], points_layer_2.data[j + 1]

    return selected_pts, selected_pts_2

def overlay_images_napari(image_list):
    viewer = napari.Viewer(title='display images')
    for image in image_list:
        viewer.add_image(image)
    napari.run()

def select_roi_in_image(image, roi_start_size = 200, print_output = False):
    shape_size = roi_start_size
    shape_corner = len(image[1,:])/2 - shape_size/2
    viewer = napari.Viewer(title = 'Select region of interest & close window to apply selection')
    # viewer.window.qt_viewer.dockLayerList.setVisible(False)
    # viewer.window.qt_viewer.dockLayerControls.setVisible(False)
    viewer.add_image(image)
    shapes_layer = viewer.add_shapes([[shape_corner,shape_corner],[shape_corner,shape_corner+shape_size],[shape_corner+shape_size,shape_corner+shape_size],[shape_corner+shape_size,shape_corner]], shape_type= 'rectangle', edge_color = 'red', face_color = '#ffffff00', edge_width = 2)
    shapes_layer.mode = 'SELECT'
    viewer.show(block=True)
    if print_output: print(f"Shape coordinates are:\n{shapes_layer.data}")
    return shapes_layer.data

def get_roi_values_from_selection(selected_roi_array, img):
    y = round(selected_roi_array[0][:][0][0])
    if y < 0:
        y = 0
    x = round(selected_roi_array[0][:][0][1])
    if x < 0:
        x = 0

    y2 = round(selected_roi_array[0][:][2][0])
    if y2 > img.shape[0]:
        y2 = img.shape[0]
    x2 = round(selected_roi_array[0][:][2][1])
    if x2 > img.shape[1]:
        x2 = img.shape[1]

    roi_size_y = abs(y - y2)
    print("sizey:", selected_roi_array[0][:][2][0])
    roi_size_x = abs(x - x2)
    print("sizex:", selected_roi_array[0][:][2][1])

    # roi_size_y = abs(y - round(selected_roi_array[0][:][2][0]))
    # print("sizey:", selected_roi_array[0][:][2][0])
    # roi_size_x = abs(x - round(selected_roi_array[0][:][2][1]))
    # print("sizex:", selected_roi_array[0][:][2][1])

    # roi_size = int(min(roi_size_x, roi_size_y)) if min(roi_size_x, roi_size_y) % 2 else int(
    #     min(roi_size_x, roi_size_y) + 1)

    # roi_center_x = int(x + (roi_size - 1) / 2)
    # roi_center_y = int(y + (roi_size - 1) / 2)

    print("roi size x:", roi_size_x, "roi size y:", roi_size_y)

    # print(f'roi_center_x: {roi_center_x}, roi_center_y: {roi_center_y}, roi_size: {roi_size}')
    return (roi_size_x, roi_size_y, (x, y))
    # roi_center_x, roi_center_y, roi_size)

def create_roi(image, x0_roi, y0_roi, size_roi_x, size_roi_y, type_roi='topleft', test_plot=False):
    roi = np.zeros((size_roi_y, size_roi_x), np.uint16)

    if type_roi == 'topleft':
        roi = image[y0_roi:y0_roi + size_roi_y, x0_roi:x0_roi + size_roi_x]
    else:
        print('ERROR - invalid type of ROI. Choose from ["center","topleft"]')
        return 0

    roi = np.asarray(roi)

    if test_plot:
        viewer = napari.Viewer(title=f'Image [{int(len(roi) / 2)}]')
        viewer.add_image(roi)
        viewer.show(block=True)
    print("worked!")
    return roi

if __name__ == "__main__":
    path_1 = 'images/lovric2_TEM-3-binned-2.tif'
    img1 = cv2.imread(path_1, cv2.IMREAD_COLOR_RGB)

    path_2 = 'images/lamellae_lovric2_TEM-3.tif'
    img2 = cv2.imread(path_2, cv2.IMREAD_COLOR_RGB)

    roi_data = select_roi_in_image(img1, print_output=True)
    rv = get_roi_values_from_selection(roi_data, img1)
    create_roi(img1, rv[2][1], rv[2][0], rv[0], rv[1], test_plot=True)
    # select_points_in_two_images(img1, img2, print_output=True)
    # viewer1 = napari.Viewer()
    # image_layer_1 = viewer1.add_image(img1)
    #
    # # viewer2 = napari.Viewer()
    # image_layer_2 = viewer1.add_image(img2)

    #napari.run()