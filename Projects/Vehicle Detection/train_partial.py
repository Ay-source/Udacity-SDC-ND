from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
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
from nn import *
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model


features_func = features_extract()
pickle_file = "car_model.p"
process_extra = True
batch_size = 256
spatial_size = 64
nbins=48
bin_range=(0, 256)
orient=9
pix_per_cell=9
cell_per_block=2
in_color_type = "BGR"
out_space = "HSV"


# Features extraction
# Calculate the bin spatial
# Get color features
# Get hog features
def extract_feat(X):
    X_features = []
    for working_image in X:
        features = features_func(working_image, in_color_type, (spatial_size, spatial_size), nbins, bin_range, orient, pix_per_cell, cell_per_block, [1, 1, 1])
        X_features.append(features)
    X=X_features
    X_features=None
    return X



def batching(total_dataset, total_label, batch_size, scaler=None, progress=1):
    total_size = len(total_dataset)
    total_dataset, total_label = shuffle(total_dataset, total_label, random_state=43)
    if progress:
        looper = tqdm(range(0, total_size, batch_size))
    else:
        looper = range(0, total_size, batch_size)
    for i in looper:
        X_paths = total_dataset[i:i+batch_size]
        X = read_list(X_paths)
        X = extract_feat(X)
        y = total_label[i:i+batch_size]
        if scaler:
            X_scaler.transform(X)
        yield np.array(X), np.array(y)

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

# Get non vehicle dataset
non_vehicle_list = glob.glob(non_vehicle_dir)
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


tot_train = train_vehicles + train_non_vehicles
tot_train_labels = np.hstack((np.ones(len(train_vehicles)), np.zeros(len(train_non_vehicles))))
#batched_train = batching(tot_train, tot_train_labels, batch_size)
tot_test = test_vehicles + test_non_vehicles
tot_test_labels = np.hstack((np.ones(len(test_vehicles)), np.zeros(len(test_non_vehicles))))
#batched_test = batching(tot_test, tot_test_labels, batch_size)


# Data Preprocessing
print("Normalizing")
X_scaler = StandardScaler()
for (X_train, y_train) in batching(tot_train, tot_train_labels, batch_size):
    X_scaler.partial_fit(X_train)

print("Training")
epochs = 10
clf = SGDClassifier(loss="hinge", learning_rate="adaptive", eta0=0.0001, alpha=0.0001)
for epoch in tqdm(range(epochs)):
    for (X_train, y_train) in batching(tot_train, tot_train_labels, batch_size, progress=0):
        X_train = X_scaler.transform(X_train)
        clf.partial_fit(X_train, y_train, classes=[0, 1])
"""
model = MyModel()
resume = 1
if resume:
    model = load_model("model_epoch-13_val_loss-0.226728.keras", custom_objects={"MyModel":MyModel})
W = 16320
def train_gen():
    return batching(tot_train, tot_train_labels, batch_size, 1, 0)
def val_gen():
    return batching(tot_test, tot_test_labels, batch_size, 1, 0)
train_steps = int(ceil(len(tot_train)/batch_size))
val_steps = int(ceil(len(tot_test)/batch_size))
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
train_ds = tf.data.Dataset.from_generator(
    train_gen,
    output_signature=(
        tf.TensorSpec(shape=(None, W), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
).repeat()

val_ds = tf.data.Dataset.from_generator(
    val_gen,
    output_signature=(
        tf.TensorSpec(shape=(None, W), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
).repeat()

filepath = "model_epoch-{epoch:02d}_val_loss-{val_loss:4f}.keras"
checkpoint_callback = ModelCheckpoint(
    filepath = filepath,
    monitor = "val_loss",
    save_best_only = True,
    mode="min",
    verbose=1,
)

lr_callback = ReduceLROnPlateau(
    monitor = "val_loss",
    factor=0.1,
    patience=5,
    min_lr=1e-6,
    verbose=1
)


model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    epochs=40,
    steps_per_epoch = train_steps,
    initial_epoch = 13,
    validation_data = val_ds,
    validation_steps = val_steps,
    callbacks = [checkpoint_callback, lr_callback]
)

model.save("nn_model.keras")
print("Model saved successfully")
"""

print("Testing")
p = []
for (X_test, y_test) in batching(tot_test, tot_test_labels, batch_size):
    X_test = X_scaler.transform(X_test)
    p.append(clf.score(X_test, y_test))


print(f"Accuracy is {sum(p)/len(p)}")

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

#"""
