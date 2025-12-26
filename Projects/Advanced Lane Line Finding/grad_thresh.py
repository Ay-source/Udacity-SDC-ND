import numpy as np
import cv2


class gradient_and_thresholding():
    def __init__(self):
        pass

    def single_thresh(self, vertical, thresh, gray, ksize):
        if vertical:
            sb = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        else:
            sb = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        abs_sb = np.absolute(sb)
        scaled_sb = np.uint8(255*abs_sb/np.max(abs_sb))
        sb_binary = np.zeros_like(scaled_sb)
        sb_binary[(scaled_sb > thresh[0]) & (scaled_sb < thresh[1])] = 1
        return sb_binary

    def direction_thresh(self, thresh, gray, ksize):
        sbx = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        sby = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        abs_sb = np.sqrt(sbx**2 + sby**2)
        scaled_sb = np.uint8(255*abs_sb/np.max(abs_sb))
        sb_binary = np.zeros_like(scaled_sb)
        sb_binary[(scaled_sb > thresh[0]) & (scaled_sb < thresh[1])] = 1
        return sb_binary

    def magnitude_thresh(self, thresh, gray, ksize):
        sbx = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        sby = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        abs_sb = np.arctan2(np.absolute(sby),np.absolute(sbx))
        scaled_sb = np.uint8(255*abs_sb/np.max(abs_sb))
        sb_binary = np.zeros_like(scaled_sb)
        sb_binary[(scaled_sb > thresh[0]) & (scaled_sb < thresh[1])] = 1
        return sb_binary

    def __call__(self, image, ksize):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        gray = image[:, :, -1]
        masked_image = np.zeros_like(gray)
        x = self.single_thresh(False, (15, 230), gray, ksize)
        y = self.single_thresh(True, (35, 230), gray, ksize)
        directional = self.direction_thresh((10, 100), gray, ksize)
        magnitude = self.magnitude_thresh((20, np.pi/3), gray, ksize)
        masked_image[(x == 1) & (y==1) | (directional==1) & (magnitude==1)] = 1
        return masked_image