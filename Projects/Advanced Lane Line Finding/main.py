# Steps to performing a road curvature
# 1. Read each inage frame
# calibrate the camera
# Gradient and Thresholding
# perspective transform
# Finding the lane lines
# Image To Real World
# Combine marked frames into a video 


from video_image_conv import video_to_images, images_to_video
from cam_cal import camera_calibrate
from grad_thresh import gradient_and_thresholding
from perspective_transform import perspective_transform
from find_lane_lines import find_lane_lines
from pixel_real import pixel_real
from tqdm import tqdm
import numpy as np
import matplotlib.image as mpimg

import matplotlib.pyplot as plt

test_image = "./test_images/test1.jpg"
#test_image = "./camera_cal/calibration12.jpg"
video_path = "./project_video.mp4"
video_path = "./challenge_video.mp4"
#video_path = "./harder_challenge_video.mp4"
result_path = "./result/" + video_path[2:]
fps = 30

# Video to Image
print("Converting from video to Images...")
video_file = video_to_images(video_path, fps)
image_frames = video_file()
print("Done")



# Camera Calibration
print("Calibrating Camera...")
camera_images_dir = "./camera_cal/*.jpg"
nx = 9
ny = 6
calibration = camera_calibrate(camera_images_dir, nx, ny)
calibrate = calibration()
print("Done")

# Gradient and Thresholding
Gradient = gradient_and_thresholding()

# Perspective Transform
Perspective = perspective_transform(test_image)

# Finding Lane Lines
Find_lanes = find_lane_lines()

# Image Results
images = []


# Video to Image
#video_file = ""
#image_frames = [test_image]#video_to_images()

print("Processing Images...")
for image in tqdm(image_frames):
    # Camera Calibration
    calibrated = calibration.undist_image(image)

    # Gradient and Thresholding
    threshed = Gradient(calibrated, 3)

    # Perspective Transform
    perspect = Perspective(threshed)

    # Finding Lane Lines
    lanes = Find_lanes(perspect)

    # Getting the inverse transform
    inverse = Perspective.inverse(lanes)

    # Output the normal image with the marked out region
    lanes = Find_lanes.output(image, inverse)

    images.append(lanes)

print("Done")

print("Converting from images to video")

im_to_vid = images_to_video(np.array(images), result_path, fps)
im_to_vid()

print("Completed")