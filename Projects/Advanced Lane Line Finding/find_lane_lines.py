import numpy as np
import cv2
import matplotlib.pyplot as plt
from perspective_transform import perspective_transform

class find_lane_lines():
    def __init__(self, nwindows=9, margin=100, minpix=50):
        self.counter = 0
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix
        self.eleftx = 0
        self.elefty = 0
        self.erightx = 0
        self.erighty = 0
        self.leftx = 0
        self.lefty = 0
        self.rightx = 0
        self.righty = 0


    def output(self, image):

        leftx_max = np.max(self.leftx)
        leftx_max = np.max(self.leftx)
        rightx_max = np.max(self.rightx)
        rightx_max = np.max(self.rightx)
        return output_image
    

    def conv(self, image):
        return image
    

    
    def slide(self, image):
        return image
    

    def fit_polynomial_slide(self, image):#, leftx, lefty, rightx, righty):
        self.find_first(image)

        try:
            left_fit = np.polyfit(self.leftx, self.lefty, 2)
            self.eleftx = self.leftx
            self.elefty = self.lefty
        except:
            left_fit = np.polyfit(self.eleftx, self.elefty, 2)
        try:
            right_fit = np.polyfit(self.rightx, self.righty, 2)
            self.erightx = self.rightx
            self.erighty = self.righty
        except:
            right_fit = np.polyfit(self.erightx, self.erighty, 2)

        ploty = np.linspace(0, image.shape[0]-1, image.shape[0])

        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        self.result_image[self.lefty, self.leftx] = [255, 0, 0]
        self.result_image[self.righty, self.rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        #return image
    

    def find_first(self, image):
        midpoint = image.shape[1]//2
        histogram = np.sum(image[:image.shape[0]//2], axis=0)
        left_x_base = np.argmax(histogram[:midpoint], axis=0)
        right_x_base = np.argmax(histogram[midpoint:], axis=0) + midpoint
        nonzero = image.nonzero()
        nonzerox = nonzero[1]
        nonzeroy = nonzero[0]
        height = image.shape[0]//self.nwindows

        left_x_current = left_x_base
        right_x_current = right_x_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.nwindows):
            y_low = image.shape[0] - (window*height)
            y_high = image.shape[0] - ((window+1)*height)
            left_x_low = left_x_current - self.margin
            left_x_high = left_x_current + self.margin
            right_x_low = right_x_current - self.margin
            right_x_high = right_x_current + self.margin

            cv2.rectangle(self.result_image, (left_x_low, y_low), 
                          (left_x_high, y_high), (0, 255, 0), 2)

            cv2.rectangle(self.result_image, (right_x_low, y_low), 
                          (right_x_high, y_high), (0, 255, 0), 2)

            good_left_lane_inds = (
                (nonzerox < left_x_high) &
                (nonzerox >= left_x_low) &
                (nonzeroy < y_high) &
                (nonzerox >= y_low) 
            ).nonzero()[0]
            good_right_lane_inds = (
                (nonzerox < right_x_high) &
                (nonzerox >= right_x_low) &
                (nonzeroy < y_high) &
                (nonzerox >= y_low) 
            ).nonzero()[0]

            left_lane_inds.append(good_left_lane_inds)
            right_lane_inds.append(good_right_lane_inds)
            if len(good_right_lane_inds) >= self.minpix:
                right_x_current = np.int16(np.mean(nonzerox[good_right_lane_inds]))
            if len(good_left_lane_inds) >= self.minpix:
                left_x_current = np.int16(np.mean(nonzerox[good_left_lane_inds]))

        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except Exception as e:
            #print(f"An error occured")
            raise e
            
        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds]
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds]

        #return leftx, lefty, rightx, righty


    def __call__(self, image):
        self.result_image = np.dstack((image, image, image)) * 255
        if self.counter:
            self.counter += 1
            self.slide(image)
        else:
            self.counter = 0
            #self.find_first(image)

            self.fit_polynomial_slide(image)#, leftx, lefty, rightx, righty)
        return self.result_image