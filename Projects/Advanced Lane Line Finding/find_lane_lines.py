import numpy as np
import cv2
import matplotlib.pyplot as plt
from perspective_transform import perspective_transform
from PIL import Image, ImageDraw, ImageFont
from pixel_real import *


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
        self.prev_margin = 50
        self.prev_center = [0, 0]#[left, right]


    def draw_poly(self, real_image ,inverse):
        output_image = cv2.addWeighted(inverse, 0.95, real_image, 1, 0)
        pil_image = Image.fromarray(output_image)
        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.truetype("arial.ttf", 25)
        # Get the radius of curvature
        rad_curv = pixel_real(self.ploty, self.left_fit, self.right_fit)
        left_rad, right_rad = rad_curv.measure_curvature_real()
        draw.text((50, 50), f"left lane radius of curvature {left_rad:.2f}", fill="white", font=font)
        draw.text((50, 100), f"right lane radius of curvature {right_rad:.2f}", fill="white", font=font)
        output_image = np.array(pil_image)
        return output_image
    

    def window_mask(self, width, height, img_ref, center,level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output

    def find_window_centroids(self, image, window_width=50, window_height=80, margin=100, minpix=100):
        window_centroids = []
        window = np.ones(window_width)
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        if max(np.convolve(window,l_sum)) > self.minpix:
            self.prev_center[0] = l_center
        elif self.prev_center[0] != 0 and max(np.convolve(window,l_sum)) == self.minpix:
            l_center = self.prev_center[0]
        else:
            self.prev_center[0] = l_center
        r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
        if max(np.convolve(window,r_sum)) > self.minpix:
            self.prev_center[1] = r_center
        elif self.prev_center[1] != 0 and max(np.convolve(window,r_sum)) == self.minpix:
            r_center = self.prev_center[1]
        else:
            self.prev_center[1] = r_center
        window_centroids.append((l_center,r_center))
        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(image.shape[0]/window_height)):
            image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,image.shape[1]))
            if max(conv_signal[l_min_index:l_max_index]) > self.minpix:
                l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
                self.prev_center[0] = l_center
            else:
                l_center = self.prev_center[0]
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,image.shape[1]))
            if max(conv_signal[r_min_index:r_max_index]) > self.minpix:
                r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
                self.prev_center[1] = r_center
            else:
                r_center = self.prev_center[1]
            window_centroids.append((l_center,r_center))

        return window_centroids

    def process_centers(self, window_centroids, warped, window_width, window_height):
        if len(window_centroids) > 0:
            l_points = np.zeros_like(warped)
            r_points = np.zeros_like(warped) 	
            for level in range(0,len(window_centroids)):
                l_mask = self.window_mask(window_width,window_height,warped,window_centroids[level][0],level)
                r_mask = self.window_mask(window_width,window_height,warped,window_centroids[level][1],level)
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
            l_nonzero = l_points.nonzero()
            r_nonzero = r_points.nonzero()
            self.leftx = l_nonzero[1]
            self.lefty = l_nonzero[0]
            self.rightx = r_nonzero[1]
            self.righty = r_nonzero[0]
            # Draw the results
            """
            template = np.array(r_points+l_points,np.uint8)
            zero_channel = np.zeros_like(template)
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8)
            warpage= np.dstack((warped, warped, warped))*255
            output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)
            plt.imshow(output)
            plt.show()
            """
            output = self.fit_polynomial_slide(warped)
        else:
            output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
        return output
    

    
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

    def method_1(self, image):
        if self.counter:
            value = self.slide(image)
            if value:
                self.counter += 1
            else:
                self.counter = 0
        if not self.counter:
            self.find_first(image)
            self.counter += 1
        processed_image = self.fit_polynomial_slide(image)
        return processed_image

    def method_2(self, image):
        window_width = 50
        window_height = 80
        margin = 100
        self.minpix = 0
        window_centroids = self.find_window_centroids(image, window_height, window_width, margin)
        output_image = self.process_centers(window_centroids, image, window_width, window_height)
        return output_image



    def __call__(self, image, conv=False):
        copy_img = np.zeros_like(image)
        self.result_image = np.dstack((copy_img, copy_img, copy_img)) * 255
        if not conv:
            new_image = self.method_1(image)
        else:
            new_image = self.method_2(image)
        return new_image
    