import cv2
import matplotlib.pyplot as plt


fname = "./images/calibration_test.png"
image = cv2.imread(fname)

nx = 8
ny = 6

# Conversion to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find corners in the imagge
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# Draw corners
# check if corners exist
if ret is True:
    cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
    plt.imshow(image)