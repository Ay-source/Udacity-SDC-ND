import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt

class features_extract():
    def __init__(self, spatial=True, colors=True, hogs=True):
        self.spatial = spatial
        self.colors = colors
        self.hogs = hogs

    def light_independent_feat(self, image):
        grad = gradient_and_thresholding()(image, 3)
        image = grad.ravel()
        return image
    
    def get_bin_spatial(self, image, provided_color, new_size=(32, 32)):
        image = self.color_space(image, provided_color, "RGB")
        spatial_features = cv2.resize(image, new_size).ravel()
        return spatial_features
    
    def get_color_features(self, image, provided_color, bin=32, bin_range=(0, 256)):
        sat_im = image
        image = self.color_space(image, provided_color, "YUV")
        layer1 = np.histogram(image[:, :, 0], bins=bin, range=bin_range)
        layer2 = np.histogram(image[:, :, 0], bins=bin, range=bin_range)
        layer3 = np.histogram(image[:, :, 0], bins=bin, range=bin_range)
        features = np.concatenate((layer1[0], layer2[0], layer3[0]))
        return features

    def color_space(self, image, provided_color="RGB", out="HLS"):
        if provided_color != "RGB":
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if out=="HSV":
            return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif out=="HLS":
            return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif out=="YUV":
            return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif out=="YCrCb":
            return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        elif out=="LUV":
            return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        return image

    def get_hog_features(self, image, provided_color, orient, pix_per_cell, cell_per_block, visualize=False, feature_vector=False):
        hog_features = []
        hog_images = []
        if len(image.shape) == 2:
            image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        for channel in range(image.shape[-1]):
            if visualize==False:
                hog_feature = hog(
                    image[:, :, channel], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), 
                    cells_per_block=(cell_per_block, cell_per_block), visualize=visualize, 
                    feature_vector=feature_vector, block_norm="L2-Hys", transform_sqrt=True
                )
                hog_features.append(hog_feature)
                hog_images.append(None)
            else:
                hog_feature, hog_image = hog(
                    image[:, :, channel], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), 
                    cells_per_block=(cell_per_block, cell_per_block), visualize=visualize, 
                    feature_vector=feature_vector, block_norm="L2-Hys", transform_sqrt=True
                )
                hog_features.append(hog_feature)
                hog_images.append(hog_image)
        return hog_features, hog_images

    def __call__(self, image, provided_color, new_size=(32, 32), nbins=32, bin_range=(0, 256), orient=1, pix_per_cell=1, cell_per_block=1, features_list=[1, 1, 1], visualize=False, f_vector=False):
        image = self.color_space(image, provided_color, "RGB")
        total_features = [[], [], []]
        if features_list[0]:
            total_features[0] = self.get_bin_spatial(image, "RGB", new_size)
        if features_list[1]:
            total_features[1] = self.get_color_features(image, "RGB", nbins, bin_range)
        if features_list[2]:
            hog_image = self.color_space(image, "RGB", "YUV")
            total_features[2], hog_images = self.get_hog_features(hog_image, provided_color, orient, pix_per_cell, cell_per_block, visualize)
            total_features[2] = np.array(total_features[2]).ravel()
        if not visualize:
            total_features = np.concatenate(total_features)
            return total_features
        total_features = np.concatenate(total_features)
        return total_features, hog_images
        