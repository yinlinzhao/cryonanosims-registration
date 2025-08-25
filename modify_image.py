import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

# blur(img, blur_intensity)
# Input: image array, int blur intensity
# Returns: Blurred image array
#
# Applies blur to an image
def blur(img, blur_intensity):
    img = cv2.blur(img, (blur_intensity,blur_intensity))
    return img

# gaussian_blur(img, blur_intensity)
# Input: image array, int blur intensity
# Returns: gaussian blurred image array
#
# Applies gaussian blur to an image
def gaussian_blur(img, blur_intensity):
    img = cv2.GaussianBlur(img, (blur_intensity,blur_intensity), 0)
    return img

# isolate_edge_canny(img, blur_intensity, min, max, kernel)
# Input: image array, int blur intensity, int minimum threshold, int maximum threshold, int kernel value
# Returns: Black and white image of the edges of input image
#
# Blurs an image and applies OpenCV's Canny() function to it
def isolate_edge_canny(img, blur_intensity, min, max, kernel):
    img = blur(img, blur_intensity)
    img = cv2.Canny(img, min, max, kernel)
    return img

# isolate_edge_sobel(img, blur_intensity, scale=1, delta=0, ddepth=cv2.CV_16S
# Input: image array, int blur intensity, int scale, int delta, ddepth type
# Returns: sobel edge isolated image array
#
# Isolates edges using sobel edge detection
def isolate_edge_sobel(img, blur_intensity, scale=1, delta=0, ddepth=cv2.CV_16S):
    img = blur(img, blur_intensity)
    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad

# re_dimension(image, target_height)
# Input: image array, int target image height
# Returns: (int adjusted height, int adjusted width)
#
# Calculates the new dimension of the image using the new image height
def re_dimension(image, target_height):
    image_size_og = (image.shape[1], image.shape[0])
    image_wh_proportion = image_size_og[0] / image_size_og[1]
    TEM_new_dimension = (int(target_height*image_wh_proportion), int(target_height))
    if TEM_new_dimension[0] <= 0:
        TEM_new_dimension = (1, TEM_new_dimension[1])
    if TEM_new_dimension[1] <= 0:
        TEM_new_dimension = (TEM_new_dimension[0], 1)

    return TEM_new_dimension

# show_image(name, image, shape_by_height=True)
# Input: string image name, image array, boolean scale by height or width
# Returns: Null
#
# Displays an image in OpenCV, scaled to fit within the screen
def show_image(name, image, shape_by_height=True):
    old_dimensions = (image.shape[0], image.shape[1]) #height, width
    if(shape_by_height):
        proportion = old_dimensions[1] / old_dimensions[0]
        new_height = 500
        new_width = new_height * proportion # h * w/h = w
    else:
        proportion = old_dimensions[0] / old_dimensions[1]
        new_width = 1000
        new_height = new_width * proportion # w * h/w = h
    new_dimensions = (int(new_width), int(new_height))

    resized = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    cv2.imshow(name, resized)

# overlay_images(overlay, background, shift, alpha)
# Input: top image array, bottom image array, (int shift x, int shift y), float alpha transparency value
# Returns: Image array of overlay displayed over background image at specified coordinates
#
# Displays and returns an image over another using opencv and matplotlib
def overlay_images(overlay, background, shift, alpha):
    # translation_matrix = np.float32([[shift[0], 0, 0],
    #                       [0, shift[1], 0]])
    # new_overlay = cv2.warpAffine(overlay, translation_matrix, (background.shape[1], background.shape[0]))

    # Desired transparency level (0 to 1)
    desired_alpha = alpha
    # Split overlay into channels
    overlay_rgb = overlay[:, :, :3]
    # Dimensions
    h, w = overlay_rgb.shape[:2]
    #create alpha
    overlay_alpha = np.full((h, w), 0.5)
    # Apply desired overall transparency
    overlay_alpha = overlay_alpha * desired_alpha
    # Position of overlay
    x, y = shift[0], shift[1]
    # Crop region of interest (ROI) from background
    roi = background[y:y + h, x:x + w]
    # Blend overlay with ROI
    for c in range(3):
        roi[:, :, c] = (overlay_rgb[:, :, c] * overlay_alpha +
                        roi[:, :, c] * (1 - overlay_alpha)).astype(np.uint8)
    # Put blended ROI back into the background
    background[y:y + h, x:x + w] = roi

    plt.imshow(background)
    plt.show()
    return background
    #show_image('overlapping images', background)

# show_histogram(img)
# Input: image array
# Returns: Histogram of img
#
# Displays the histogram of an image in matplotlib and returns it
def show_histogram(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 255, color='red')
    plt.xlim([0, 255])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()

    return cdf

def create_window(map_img, lam_img):
    #OPEN POINT SELECTION WINDOW
    #these set up what happens upon these mouse events
    map_points = []
    lam_points = []

    def on_click(event):
        if event.button is MouseButton.LEFT:
            if event.inaxes == ax1:
                map_points.append([event.xdata, event.ydata])
                ax1.plot([event.xdata], [event.ydata], marker='*', markersize=5)
                plt.show()

                print(f'axis1 pixel coords {event.xdata} {event.ydata}')
            elif event.inaxes == ax2:
                lam_points.append([event.xdata, event.ydata])
                ax2.plot([event.xdata], [event.ydata], marker='*', markersize=5)
                plt.show()

                print(f'axis2 pixel coords {event.xdata} {event.ydata}')
        # if event.button is MouseButton.RIGHT:
        #     print('disconnecting callback')
        #     print("map points: ", map_points)
        #     print("lamellae points: ", lam_points)
        #     plt.disconnect(click_id)
        #
        #     plt.suptitle("close this window")

    def on_press(event):
        print("you pressed: ", event.key)
        if event.key == 'enter':
            print("hi you pressed enter")
            print("okay connected to onclick")
            plt.suptitle("select corresponding points, close out of window when done")
            plt.show()
            click_id = plt.connect('button_press_event', on_click)
            plt.show()
            plt.disconnect(key_id)

    ax1 = plt.subplot(1,2,1)
    plt.imshow(map_img)

    ax2 = plt.subplot(1,2,2)
    plt.imshow(lam_img)

    key_id = plt.connect('key_press_event', on_press)
    # key_id = plt.connect('button_press_event', on_click)
    plt.suptitle("use magnifying glass to select region, then press enter")

    plt.show()
    return (map_points, lam_points)