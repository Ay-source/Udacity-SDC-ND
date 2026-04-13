import pickle
import cv2
import glob
from video import *
from extra_funcs import find_cars, find_cars_cnn
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model
from nn import *

# Extracting model and parameters from pickle file


nn = 0
pickle_file_dir = "car_model.p"
with open(pickle_file_dir, "rb") as pickle_file:
    dist_pickle = pickle.load(pickle_file)
if nn:
    k_model = "model_epoch-13_val_loss-0.226728.keras"
    clf = load_model(k_model, custom_objects={"MyModel":MyModel})
else:
    clf = dist_pickle["model"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
X_scaler = dist_pickle["X_scaler"]
orientation=dist_pickle["orient"]
spatial_size = dist_pickle["spatial_size"]
nbins = dist_pickle["nbins"]
out_space = dist_pickle["out_space"]
bin_range = dist_pickle["bin_range"]
testing = False
fps = 7
start_scale = 0.75
interval = 0.25
end_scale = 2
scale = [start_scale+interval*i for i in range(int(end_scale/interval)) if i > start_scale]
scale = [i for i in scale if i <= end_scale]
processed_images = []

# Break videos into images
if testing:
    test_image_dir = "./test_images/*.jpg"
    images = np.array([mpimg.imread(i) for i in glob.glob(test_image_dir)])
    vid_result_path = "./test_result_video.mp4"
else:
    video_file_path = "./project_video.mp4"
    images = video_to_images(video_file_path, fps)()
    vid_result_path = "./result_video.mp4"


# Extract image features
ymin = 400
ymax = 670
window = 64
cells_per_step = 1
provided_color = "RGB"
cnn = 0

for image in tqdm(images):
    if not cnn:
        drawn_image = find_cars(image, scale, X_scaler, pix_per_cell, cell_per_block, window, cells_per_step, orientation, clf, ymin, ymax, spatial_size, nbins, provided_color, bin_range)
    else:
        k_model = "cnn_model_epoch-21_val_loss-0.141118.keras"
        clf = load_model(k_model, custom_objects={"CNNModel":CNNModel})
        cnn_details_dir = "cnn_details.p"
        with open(cnn_details_dir, "rb") as pickle_file:
            dist_pickle = pickle.load(pickle_file)
        X_scaler = dist_pickle["X_scaler"]
        drawn_image = find_cars_cnn(image, scale, X_scaler, pix_per_cell, cell_per_block, window, cells_per_step, clf, ymin, ymax)
    #plt.imshow(drawn_image)
    #plt.show()
    processed_images.append(drawn_image)


# Convert processed images into video
vid_create = images_to_video(np.array(processed_images), vid_result_path, fps)
vid_create()
