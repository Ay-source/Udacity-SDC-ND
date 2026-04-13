from sklearn.svm import LinearSVC, SVC
from features import features_extract
import glob
import numpy as np
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pickle
from extra_funcs import read_list, split_train_test
from tqdm import tqdm
from math import ceil


features_func = features_extract()
pickle_file = "car_model.p"
process_extra = True
batch_size = 128


def batching(total_dataset, total_label, batch_size):
    total_size = len(total_dataset)
    steps = ceil(total_size / batch_size)
    X, y = shuffle(total_dataset, total_label, random_state=43)
    for i in range(0, total_size, steps):
        X = total_dataset[steps:i+batch_size]
        y = total_label[steps:i+batch_size]
        yield X, y

# Getting the dataset for training
# Get vehicles dataset
vehicle_dir_prefix = "./data/vehicles/"
vehicle_dir_fix = ["GTI_Far", "GTI_Left", "GTI_MiddleClose", "GTI_Right", "KITTI_extracted", "Extra_data"]
vehicle_dir_suffix = "/*g"
non_vehicle_dir = "./data/non-vehicles/*/*g"
train_vehicles = []
test_vehicles = []
percent_split = 0.7
for element in vehicle_dir_fix:
    dir_images = glob.glob(vehicle_dir_prefix+element+vehicle_dir_suffix)
    if element == vehicle_dir_fix[-1]:
        dir_images = shuffle(dir_images)
    vehicles = split_train_test(dir_images, percent_split)
    train_vehicles.append(vehicles[0])
    test_vehicles.append(vehicles[1])
    vehicles = None



extra_train_vehicles_paths = []
extra_test_vehicles_paths = []
extra_train_non_vehicles_paths = []
extra_test_non_vehicles_paths = []
if process_extra:
    path = "./data/extra/"
    dst_dirs = ["vehicles/", "non-vehicles/"]
    extra_vehicles_path = path+dst_dirs[0]
    extra_non_vehicles_path = path+dst_dirs[1]
    dsts = ["extra1", "extra2"]
    print("Extracting extra data")
    for dst in dsts:
        dir_images = glob.glob(extra_vehicles_path+dst+vehicle_dir_suffix)
        vehicles = split_train_test(dir_images, percent_split)
        extra_train_vehicles_paths.append(vehicles[0])
        extra_test_vehicles_paths.append(vehicles[1])
        vehicles = None
        dir_images = glob.glob(extra_non_vehicles_path+dst+vehicle_dir_suffix)
        non_vehicles = split_train_test(dir_images, percent_split)
        extra_train_non_vehicles_paths.append(non_vehicles[0])
        extra_test_non_vehicles_paths.append(non_vehicles[1])
        non_vehicles = None
    extra_train_vehicles_paths = extra_train_vehicles_paths[0] + extra_train_vehicles_paths[1]
    extra_test_vehicles_paths = extra_test_vehicles_paths[0] + extra_test_vehicles_paths[1]
    extra_train_non_vehicles_paths = extra_train_non_vehicles_paths[0] + extra_train_non_vehicles_paths[1]
    extra_test_non_vehicles_paths = extra_test_non_vehicles_paths[0] + extra_test_non_vehicles_paths[1]
    print("Completed extraction of extra data")

train_vehicles = train_vehicles[0] + train_vehicles[1] + train_vehicles[2] + train_vehicles[3] + train_vehicles[4] + extra_train_vehicles_paths
test_vehicles = test_vehicles[0] + test_vehicles[1] + test_vehicles[2] + test_vehicles[3] + test_vehicles[4] + extra_test_vehicles_paths
train_vehicles = shuffle(train_vehicles, random_state=43)
test_vehicles = shuffle(test_vehicles, random_state=43)
# Get non vehicle dataset
non_vehicle_list = shuffle(glob.glob(non_vehicle_dir), random_state=43)
# Print total number of images
print(f"Total number of vehicle images {len(train_vehicles)+len(test_vehicles)}")
print(f"Total number of non vehicle images {len(non_vehicle_list)+len(extra_train_non_vehicles_paths)+len(extra_test_non_vehicles_paths)}")
# Split the non vehicle images
train_non_vehicles, test_non_vehicles = train_test_split(non_vehicle_list, test_size=1-percent_split, random_state=43)
train_non_vehicles = train_non_vehicles + extra_train_non_vehicles_paths
test_non_vehicles = test_non_vehicles + extra_test_non_vehicles_paths
# Print the length of train images
print(f"Total number of train vehicle images {len(train_vehicles)}")
print(f"Total number of train non vehicle images {len(train_non_vehicles)}")
# Concatenate test and non test images into a numpy array
print("X_train")
X_train = np.vstack((read_list(train_vehicles[0:7000], True), read_list(train_non_vehicles[0:7000], True))).astype(np.uint8)
print(X_train.shape)
print("y_train")
y_train = np.hstack((np.ones(len(train_vehicles[0:7000])), np.zeros(len(train_non_vehicles[0:7000])))).astype(np.int16)
print(y_train.shape)
X_train, y_train = shuffle(X_train, y_train, random_state=43)
train_vehicles = None
train_non_vehicles = None
print("train done")
print("test")
test_vehicles_size = len(test_vehicles)
test_non_vehicles_size = len(test_non_vehicles)
print(f"Size of X_train is {X_train.shape} and size of y_train is {y_train.shape}")
print("Data loaded")


# Features extraction
# Calculate the bin spatial
# Get color features
# Get hog features
print("\nExtrating features")
spatial_size = 64
nbins=48
bin_range=(0, 256)
orient=9
pix_per_cell=9
cell_per_block=2
in_color_type = "BGR"
out_space = "HSV"
X_train_features = []
X_test_features = []
for working_image in tqdm(X_train):
    features = features_func(working_image, in_color_type, (spatial_size, spatial_size), nbins, bin_range, orient, pix_per_cell, cell_per_block, [1, 1, 1])
    X_train_features.append(features)
X_train=X_train_features
print(X_train[0].shape)
X_train_features=None
print("Features extracted")


# Data Preprocessing
X_scaler = StandardScaler().fit(X_train)
X_train = X_scaler.transform(X_train)


# Training Classifier
print("\nTraining the classifier")
#param_gridm= {"C": [0.0001, 0.01, 0.1, 1, 10, 20, 25, 100]}
#model = LinearSVC(max_iter=10000, dual=False)
#clf = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", verbose=1)
clf = LinearSVC(dual=False)
clf.fit(X_train, y_train)
#b_c = clf.best_params_["C"]
#print(f"Best C value: {b_c}")
#print(f"Best Cross-Validation Accuracy: {grid.best_score_:.4f}")
X_train, y_train = None, None

print("Working on test")
X_test = np.vstack((read_list(test_vehicles[0:3000], True), read_list(test_non_vehicles[0:3000], True)))
y_test = np.hstack((np.ones(len(test_vehicles[0:3000])), np.zeros(len(test_non_vehicles[0:3000])))).astype(np.int16)
X_test, y_test = shuffle(X_test, y_test, random_state=43)
print(f"Size of X_test is {X_test.shape} and size of y_test is {y_test.shape}")
test_vehicles = None
test_non_vehicles = None
for working_image in tqdm(X_test):
    features = features_func(working_image, in_color_type, (spatial_size, spatial_size), nbins, bin_range, orient, pix_per_cell, cell_per_block, [1, 1, 1])
    X_test_features.append(features)
X_test=X_test_features
X_test_features=None
print("Test features extracted")


X_test = X_scaler.transform(X_test)
print(f"Accuracy is {clf.score(X_test, y_test)}")

print("\nSaving data as pickle...")
with open(pickle_file, "wb") as f:
    pickle.dump(
        {
            "model": clf,
            "pix_per_cell": pix_per_cell,
            "cell_per_block": cell_per_block,
            "X_scaler": X_scaler,
            "orient": orient,
            "spatial_size": spatial_size,
            "out_space": out_space,
            "nbins": nbins,
            "bin_range": bin_range
        }, f
    )
print("Data Saved")

