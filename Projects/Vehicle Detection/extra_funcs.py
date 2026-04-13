import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from features import features_extract
from tqdm import tqdm
from tensorflow.keras.layers import Softmax


extract_features = features_extract()

def draw_boxes(image, coords, color, thick, thresh_value):
    im_copy = np.copy(image)
    labels = get_heatmap()(image, coords, thresh_value)
    """
    for coord in coords:
        cv2.rectangle(im_copy, coord[0], coord[1], color, thick)
    """
    for car_label in range(1, labels[1]+1):
        nonzero = (labels[0] == car_label).nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])
        coord_low = [np.min(nonzerox), np.min(nonzeroy)]
        coord_high = [np.max(nonzerox), np.max(nonzeroy)]
        cv2.rectangle(im_copy, coord_low, coord_high, color, thick)
    #plt.imshow(im_copy)
    #plt.show()
    #"""
    return im_copy

def read_list(path_lists, progress=False):
    images_lists = []
    if not progress:
        for path in path_lists:
            images_lists.append(cv2.imread(path))
    else:
        for path in tqdm(path_lists):
            images_lists.append(cv2.imread(path))
    return np.array(images_lists, np.uint8)

def split_train_test(path_list, train_percent=0.8):
    train_list = path_list[:int(len(path_list)*train_percent)]
    test_list = path_list[int(len(path_list)*train_percent):]
    return train_list, test_list

def find_cars(image, scales, X_scaler, pix_per_cell, cell_per_block, window, cells_per_step, orientation, clf, ymin, ymax, spatial_size, nbins, provided_color, bin_range):
    bboxes = []
    for scale in scales:
        sub_image = image[ymin:ymax, :, :]
        if scale != 1:
            im_shape = sub_image.shape
            sub_image = cv2.resize(sub_image, (int(im_shape[1]/scale), int(im_shape[0]/scale)), interpolation=cv2.INTER_AREA)
        hog_image = extract_features.color_space(sub_image, provided_color, "YUV")


        ch1 = hog_image[:, :, 0]
        ch2 = hog_image[:, :, 1]
        ch3 = hog_image[:, :, 2]


        nx_blocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        ny_blocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

        n_blocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        nx_steps = (nx_blocks - n_blocks_per_window) // cells_per_step + 1
        ny_steps = (ny_blocks - n_blocks_per_window) // cells_per_step + 1

        

        hog_feature1 = np.array(extract_features.get_hog_features(ch1, provided_color, orientation, pix_per_cell, cell_per_block)[0][0])
        hog_feature2 = np.array(extract_features.get_hog_features(ch2, provided_color, orientation, pix_per_cell, cell_per_block)[0][0])
        hog_feature3 = np.array(extract_features.get_hog_features(ch3, provided_color, orientation, pix_per_cell, cell_per_block)[0][0])




        for xs in range(nx_steps):
            for ys in range(ny_steps):
                ypos = ys * n_blocks_per_window
                xpos = xs * n_blocks_per_window
                hog_sub1 = hog_feature1[ypos:ypos+n_blocks_per_window, xpos:xpos+n_blocks_per_window].ravel()
                hog_sub2 = hog_feature2[ypos:ypos+n_blocks_per_window, xpos:xpos+n_blocks_per_window].ravel()
                hog_sub3 = hog_feature3[ypos:ypos+n_blocks_per_window, xpos:xpos+n_blocks_per_window].ravel()
                hog_features = np.hstack((hog_sub1, hog_sub2, hog_sub3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                try:
                    # Extract the image patch
                    win_image = cv2.resize(sub_image[ytop:ytop+window, xleft:xleft+window], (64,64))

                    bin_color = extract_features(win_image, provided_color, new_size=(spatial_size, spatial_size), nbins=nbins, bin_range=bin_range, features_list=[1, 1, 0])
                    total_features = np.hstack((bin_color, hog_features)).reshape(1, -1)
                    transform_features = X_scaler.transform(total_features)
                except Exception as e:
                    continue
                label = clf.predict(transform_features)

                if label:
                    xbox_left = int(xleft*scale)
                    ytop_draw = int(ytop*scale)
                    win_draw = int(window*scale)
                    bboxes.append(((xbox_left, ytop_draw+ymin),(xbox_left+win_draw,ytop_draw+win_draw+ymin)))
    if len(bboxes):
        drawn_image = draw_boxes(image, bboxes, (0, 0, 255), 6, 2)
        return drawn_image
    else:
        return image


def find_cars_cnn(image, scales, X_scaler, pix_per_cell, cell_per_block, window, cells_per_step, clf, ymin, ymax):
    bboxes = []
    for scale in scales:
        sub_image = image[ymin:ymax, :, :]
        if scale != 1:
            im_shape = sub_image.shape
            sub_image = cv2.resize(sub_image, (int(im_shape[1]/scale), int(im_shape[0]/scale)), interpolation=cv2.INTER_AREA)


        nx_blocks = (sub_image.shape[1] // pix_per_cell) - cell_per_block + 1
        ny_blocks = (sub_image.shape[0] // pix_per_cell) - cell_per_block + 1

        n_blocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        nx_steps = (nx_blocks - n_blocks_per_window) // cells_per_step + 1
        ny_steps = (ny_blocks - n_blocks_per_window) // cells_per_step + 1




        for xs in range(nx_steps):
            for ys in range(ny_steps):
                ypos = ys * n_blocks_per_window
                xpos = xs * n_blocks_per_window

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                try:
                    # Extract the image patch
                    win_image = cv2.resize(sub_image[ytop:ytop+window, xleft:xleft+window], (64,64))

                    feat = cv2.cvtColor(win_image, cv2.COLOR_BGR2HSV)
                    transform_features = (feat-X_scaler[0])/X_scaler[1]
                    transform_features = transform_features.reshape(1, 64, 64, 3)
                except Exception as e:
                    continue
                    print(e)
                label = Softmax()(clf.predict(transform_features))
                label = np.argmax(label[0])

                if label:
                    xbox_left = int(xleft*scale)
                    ytop_draw = int(ytop*scale)
                    win_draw = int(window*scale)
                    bboxes.append(((xbox_left, ytop_draw+ymin),(xbox_left+win_draw,ytop_draw+win_draw+ymin)))
    if len(bboxes):
        drawn_image = draw_boxes(image, bboxes, (0, 0, 255), 6, 2)
        return drawn_image
    else:
        return image



class get_heatmap():
    def __init__(self):
        pass 


    def heat_count(self, image ,bboxes):
        heat_image = np.zeros_like(image)
        for bbox in bboxes:
            heat_image[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1
        return heat_image

    def thresholding_image(self, heat_image, thresh_value):
        heat_image[heat_image < thresh_value] = 0
        return heat_image

    def get_labels(self, image):
        labels = label(image)
        return labels

    def __call__(self, image, bboxes, thresh_value):
        heat_image = self.heat_count(image, bboxes)
        threshed_image = self.thresholding_image(heat_image, thresh_value)
        labels = self.get_labels(threshed_image)
        return labels