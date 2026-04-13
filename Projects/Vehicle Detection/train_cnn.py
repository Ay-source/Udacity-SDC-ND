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
import pickle
from extra_funcs import read_list, split_train_test
from tqdm import tqdm
from math import ceil
from nn import CNNModel
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras import losses


features_func = features_extract()
pickle_file = "cnn_details.p"
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
        features = cv2.cvtColor(working_image, cv2.COLOR_BGR2HSV)
        X_features.append(features[:, :, :])
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
            X = (X-scaler[0])/scaler[1]
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


#train_vehicles = train_vehicles[0:5000]
#test_vehicles = train_vehicles[0:1000]
#train_non_vehicles = train_non_vehicles[0:5000]
#test_non_vehicles = test_non_vehicles[0:1000]

tot_train = train_vehicles + train_non_vehicles
tot_train_labels = np.hstack((np.ones(len(train_vehicles)), np.zeros(len(train_non_vehicles))))
tot_test = test_vehicles + test_non_vehicles
tot_test_labels = np.hstack((np.ones(len(test_vehicles)), np.zeros(len(test_non_vehicles))))


# Data Preprocessing
print("Normalizing")
ind_means = [] # To store each individual mean which are then averged to get the total mean.
ind_stds = []
for (X_train, y_train) in batching(tot_train, tot_train_labels, batch_size):
    ind_means.append(np.mean(X_train))
    ind_stds.append(np.std(X_train))

total_mean = np.mean(ind_means)
total_std = np.mean(ind_stds)

with open(pickle_file, "wb") as f:
    pickle.dump(
        {
            "pix_per_cell": pix_per_cell,
            "cell_per_block": cell_per_block,
            "X_scaler": [total_mean, total_std],
            "orient": orient,
            "spatial_size": spatial_size,
            "out_space": out_space,
            "nbins": nbins,
            "bin_range": bin_range
        }, f
    )
print("CNN details saved")

print("Training")
epochs = 10

model = CNNModel()
resume = 1
initial_epoch = 8
if resume:
    model = load_model("cnn_model_epoch-21_val_loss-0.141118.keras", custom_objects={"CNNModel":CNNModel})
def train_gen():
    return batching(tot_train, tot_train_labels, batch_size, [total_mean, total_std], 0)
def val_gen():
    return batching(tot_test, tot_test_labels, batch_size, [total_mean, total_std], 0)
train_steps = int(ceil(len(tot_train)/batch_size))
val_steps = int(ceil(len(tot_test)/batch_size))
optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
train_ds = tf.data.Dataset.from_generator(
    train_gen,
    output_signature=(
        tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
).repeat()

val_ds = tf.data.Dataset.from_generator(
    val_gen,
    output_signature=(
        tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
).repeat()

filepath = "cnn_model_epoch-{epoch:02d}_val_loss-{val_loss:4f}.keras"
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
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    epochs=40,
    steps_per_epoch = train_steps,
    initial_epoch = initial_epoch,
    validation_data = val_ds,
    validation_steps = val_steps,
    callbacks = [checkpoint_callback, lr_callback]
)

model.save("nn_model.keras")
print("Model saved successfully")
