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
        self.ploty = 0
        self.left_fitx = 0
        self.right_fitx = 0
        self.prev_margin = 200


    def draw_poly(self, real_image ,inverse):
        output_image = cv2.addWeighted(inverse, 0.95, real_image, 1, 0)#, real_image)
        return output_image
    

    def conv(self, image):
        return image
    

    
    def slide(self, image):
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        points_left = (
            (nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*(nonzeroy) + self.left_fit[2] - self.prev_margin)) &
            (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*(nonzeroy) + self.left_fit[2] + self.prev_margin))
        )
        points_right = (
            (nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*(nonzeroy) + self.right_fit[2] - self.prev_margin)) & 
            (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*(nonzeroy) + self.right_fit[2] + self.prev_margin))
        )
        self.leftx = nonzerox[points_left]
        self.lefty = nonzeroy[points_left]
        self.rightx = nonzerox[points_right]
        self.righty = nonzeroy[points_right]
        pixes = 500
        if ((self.leftx.size < pixes) | (self.lefty.size < pixes) | (self.rightx.size < pixes)| (self.righty.size < pixes)):
            return False
        else:
            return True
    

    def fit_polynomial_slide(self, image):

        try:
            self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
            self.eleftx = self.leftx
            self.elefty = self.lefty
        except:
            self.left_fit = np.polyfit(self.elefty, self.eleftx, 2)
            self.leftx = self.eleftx
            self.lefty = self.elefty
        try:
            self.right_fit = np.polyfit(self.righty, self.rightx, 2)
            self.erightx = self.rightx
            self.erighty = self.righty
        except:
            self.right_fit = np.polyfit(self.erighty, self.erightx, 2)
            self.rightx = self.erightx
            self.righty = self.erighty

        self.ploty = np.linspace(0, image.shape[0]-1, image.shape[0])

        try:
            self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
            self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            self.left_fitx = 1*self.ploty**2 + 1*self.ploty
            self.right_fitx = 1*self.ploty**2 + 1*self.ploty

        self.result_image[self.lefty, self.leftx] = [255, 0, 0]
        self.result_image[self.righty, self.rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.plot(self.left_fitx, self.ploty, color='yellow')
        plt.plot(self.right_fitx, self.ploty, color='yellow')
        new_image = np.zeros_like(image)
        new_image = np.dstack((new_image, new_image, new_image)) * 255
        #new_image[self.lefty, self.leftx] = 1
        #new_image[self.righty, self.rightx] = 1
        draw_margin = 20
        left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx-draw_margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx+draw_margin, 
                              self.ploty])))])
        left_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx-draw_margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx+draw_margin, 
                              self.ploty])))])
        right_pts = np.hstack((right_line_window1, right_line_window2))
        
        pts = np.hstack((left_line_window2, right_line_window1))
        cv2.fillPoly(new_image, np.int_([left_pts]), (255, 0, 0))
        cv2.fillPoly(new_image, np.int_([right_pts]), (0, 0, 255))
        cv2.fillPoly(new_image, np.int_([pts]), (0, 255, 0))
        out = cv2.addWeighted(self.result_image, 1, new_image, 0.3, 0)
        return out
    

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
            y_high = image.shape[0] - (window*height)
            y_low = image.shape[0] - ((window+1)*height)
            left_x_low = left_x_current - self.margin
            left_x_high = left_x_current + self.margin
            right_x_low = right_x_current - self.margin
            right_x_high = right_x_current + self.margin

            good_left_lane_inds = (
                (nonzerox < left_x_high) &
                (nonzerox >= left_x_low) &
                (nonzeroy < y_high) &
                (nonzeroy >= y_low) 
            ).nonzero()[0]
            good_right_lane_inds = (
                (nonzerox < right_x_high) &
                (nonzerox >= right_x_low) &
                (nonzeroy < y_high) &
                (nonzeroy >= y_low) 
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
            raise e
            
        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds]
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds]



    def __call__(self, image):
        copy_img = np.zeros_like(image)
        self.result_image = np.dstack((copy_img, copy_img, copy_img)) * 255
        if self.counter:
            value = self.slide(image)
            if value:
                self.counter += 1
            else:
                self.counter = 0
        if not self.counter:
            self.find_first(image)
            self.counter += 1
        new_image = self.fit_polynomial_slide(image)
        return new_image
    