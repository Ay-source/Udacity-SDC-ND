import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('./images/bbox-example-image.jpg')

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched 
    spanx = x_start_stop[1] - x_start_stop[0]
    spany = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pixels_per_step = np.int16(xy_window[0] * (1 - xy_overlap[0]))
    ny_pixels_per_step = np.int16(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int16(xy_window[0] * xy_overlap[0])
    ny_buffer = np.int16(xy_window[1] * xy_overlap[1])
    total_windows_x = np.int16((spanx - nx_buffer)/nx_pixels_per_step)
    total_windows_y = np.int16((spany - ny_buffer)/ny_pixels_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
        # Calculate each window position
        # Append window position to list
    # Return the list of windows
    for posy in range(total_windows_y):
        for posx in range(total_windows_x):
            startx = posx * nx_pixels_per_step + x_start_stop[0]
            stopx = startx + xy_window[0]
            starty = posy * ny_pixels_per_step + y_start_stop[0]
            stopy = starty + xy_window[1]
            window_list.append(((startx, starty), (stopx, stopy)))
    return window_list

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))
                       
window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)                   
plt.imshow(window_img)
plt.show()