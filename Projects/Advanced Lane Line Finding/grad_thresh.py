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

    def magnitude_thresh(self, thresh, gray, ksize):
        sbx = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        sby = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        abs_sb = np.sqrt(sbx**2 + sby**2)
        scaled_sb = np.uint8(255*abs_sb/np.max(abs_sb))
        sb_binary = np.zeros_like(scaled_sb)
        sb_binary[(scaled_sb > thresh[0]) & (scaled_sb < thresh[1])] = 1
        return sb_binary

    def direction_thresh(self, thresh, gray, ksize):
        sbx = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        sby = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        abs_sb = np.arctan2(np.absolute(sby),np.absolute(sbx))
        sb_binary = np.zeros_like(abs_sb)
        sb_binary[(abs_sb > thresh[0]) & (abs_sb < thresh[1])] = 1
        return sb_binary

    def __call__(self, image, ksize):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        gray = image[:, :, -1]
        masked_image = np.zeros_like(gray)
        x = self.single_thresh(False, (5, 230), gray, ksize) # 20
        y = self.single_thresh(True, (3, 230), gray, ksize)   # 20
        magnitude = self.magnitude_thresh((5, 100), gray, ksize)
        direction = self.direction_thresh((np.pi/5, np.pi/2), gray, ksize)
        masked_image[(x == 1) & (y==1) & (direction==1) & (magnitude==1)] = 1
        #masked_image[(x==1) & (y==1)] = 1
        return masked_image
