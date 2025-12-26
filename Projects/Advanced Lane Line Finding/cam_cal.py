import cv2
import matplotlib.image as mpimg
import numpy as np
import glob

# Steps:
# Get images directory list
# Load the images
# find chessboard corners
# draw chessboard corners if validation is true
# Use the corners to calibrate camera
# Save the camera matrix in a file

class camera_calibrate():
    def __init__(self, images_directory, nx=None, ny=None, corner_display=False):
        self.images_directory = images_directory
        self.mtx = None
        self.dist = None
        self.nx = nx
        self.ny = ny
        self.corner_display = corner_display
        self.objp = np.zeros((self.nx*self.ny, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)
        self.image_points = []
        self.object_points = []
        self.fnames = glob.glob(self.images_directory)


    def __call__(self):
        images = []
        for fname in self.fnames:
            images.append(mpimg.imread(fname))
        self.finding_corners(images)
        gray = cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY)
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.image_points, gray.shape[::-1], None, None)
   
    
    def finding_corners(self, images):
        for image in images:
            grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(grey_image, (self.nx, self.ny), None)

            if ret == True:
                self.image_points.append(corners)
                self.object_points.append(self.objp)

                if self.corner_display:
                    cv2.drawChessboardCorners(image, (self.nx, self.ny), corners, ret)

                    cv2.imshow("img", image)
                    cv2.waitKey(500)
        cv2.destroyAllWindows()


    def undist_image(self, image):
        #image = mpimg.imread(image_path)
        calibrated_image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        return calibrated_image
    