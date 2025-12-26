import cv2
import numpy as np
import matplotlib.image as mpimg

class perspective_transform():
    def __init__(self, cal_path):
        self.cal_image = mpimg.imread(cal_path)
        self.img_size = (self.cal_image.shape[1], self.cal_image.shape[0])
        self.src = np.float32([
            [550, 476],
            [724, 476],
            [1042, 675],
            [279, 666]
        ])
        self.dst = np.float32([
            [279, 0],
            [1042, 0],
            [1042, self.img_size[1]],
            [279, self.img_size[1]]
        ])

    def __call__(self, image):
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        warped_image = cv2.warpPerspective(image, M, self.img_size, flags=cv2.INTER_LINEAR)
        return warped_image
    
    def inverse(self, image):
        M = cv2.getPerspectiveTransform(self.dst, self.src)
        warped_image = cv2.warpPerspective(image, M, self.img_size, flags=cv2.INTER_LINEAR)
        return warped_image
    
    