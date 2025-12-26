import csv
import random
import matplotlib.image as mpimg
import numpy as np
import cv2

from tensorflow import keras
from keras.layers import Rescaling, Dense, Conv2D, LayerNormalization, BatchNormalization, Activation, MaxPooling2D, Dropout, Cropping2D, Flatten, Lambda, Resizing, Input, ZeroPadding2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split

p_dir = "./data/"
csv_file = p_dir + "driving_log.csv"
images_dir = p_dir + "IMG/"
LeNet = True

# Loading the csv file
def csv_read():
    lines = []
    with open(csv_file, "r") as csv_data:
        csv_data = csv.reader(csv_data)
        for line in csv_data:
            for member in line[0:3]:
                image_suffix = member.split("\\")[-1]
                member = images_dir + image_suffix
                lines.append([member, line[3], line[4], line[5], line[6]])
    return lines


def load_and_sort_data(lines_data, LeNet):
    image_data = []
    image_label = {
        "Steering Angle": [],
        "Throttle": [],
        "Break": [],
        "Speed": []
    }

    for line in lines_data:
        try:
            image = mpimg.imread(line[0])
            if LeNet:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                image = image[:, :, 0]
        except:
            print(f"Image Error: Image not identified")
            continue
        if image is None:
            continue
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            line[1] = -1.0 * float(line[1])
        image_data.append(image)
        image_label["Steering Angle"].append(float(line[1]))
        image_label["Throttle"].append(float(line[2]))
        image_label["Break"].append(float(line[3]))
        image_label["Speed"].append(float(line[4]))
    return image_data, image_label


def data_generator(sample, batch_size=1):
    length = len(sample)
    for i in range(0,length,batch_size):
        j = i+batch_size+1
        image_data, image_label = load_and_sort_data(sample[i:j], LeNet)
        image_steer_label = image_label["Steering Angle"]
        # Use this to add other labels:
        #image_throttle_label = image_label["throttle"]
        yield np.array(image_data), np.array(image_steer_label)


# Used for debugging the output sizes of a layer
class PrintLayer(keras.layers.Layer):
    def call(self, x):
        print(x.get_shape)
        return x


def Ayo_LeNet_Model():
    model = Sequential([

        # Preprocessing Layer
        Input(shape=(160, 320, 1)),
        Cropping2D(cropping=((50, 27), (0, 0))),
        Resizing(height=32, width=32),
        #Lambda(lambda x: (x-128) / 128, input_shape=(160, 320, 3)),
        Rescaling(1/128, offset=-128/128),

        # Conv Layer 1
        Conv2D(16, 3),
        BatchNormalization(),
        Activation("relu"),


        # Conv Layer 2
        Conv2D(32, 3),
        BatchNormalization(),
        Activation("relu"),

        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        # Conv Layer 3
        Conv2D(64, 3),
        BatchNormalization(),
        Activation("relu"),

        # Conv ayer 4
        Conv2D(128, 3),
        BatchNormalization(),
        Activation("relu"),

        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        # Flatten Layer
        Flatten(),

        # Fully Connected Layer 1
        Dense(1024),
        LayerNormalization(),
        Activation("relu"),
        Dropout(rate=0.5),

        # Fully Connected Layer 1
        Dense(180),
        LayerNormalization(),
        Activation("relu"),
        Dropout(rate=0.5),

        # Output Layer
        Dense(1)
        ])

    return model



def Nvidia_Model():
    model = Sequential([

        # Preprocessing Layer
        Input(shape=(160, 320, 3)),
        Cropping2D(cropping=((50, 27), (0, 0))),
        Resizing(height=66, width=200),
        #Lambda(lambda x: (x-128) / 128, input_shape=(160, 320, 3)),
        Rescaling(1/128, offset=-128/128),

        

        # Conv Layer 1
        Conv2D(24, 5),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        # Conv Layer 2
        Conv2D(36, 5),
        BatchNormalization(),
        Activation("relu"),
        ZeroPadding2D(padding=((0, 1), (0, 1))),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),


        # Conv Layer 3
        Conv2D(48, 5),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        # Conv Layer 4
        Conv2D(64, 3),
        BatchNormalization(),
        Activation("relu"),

        # Conv Layer 5
        Conv2D(64, 3),
        BatchNormalization(),
        Activation("relu"),


        # Flatten Layer
        Flatten(),

        # Fully Connected Layer 1
        Dense(100),
        LayerNormalization(),
        Activation("relu"),
        Dropout(rate=0.5),

        # Fully Connected Layer 1
        Dense(500),
        LayerNormalization(),
        Activation("relu"),
        Dropout(rate=0.5),

        # Fully Connected Layer 1
        Dense(10),
        LayerNormalization(),
        Activation("relu"),
        Dropout(rate=0.5),

        # Output Layer
        Dense(1)
    ])

    return model


# Load the csv_file
print("Reading csv file")
csv_lines = csv_read()
print("Splitting train data")
train_data, valid_data = train_test_split(
    csv_lines,
    random_state=42,
    test_size=0.15
)
print("Splitting test data")
train_data, test_data = train_test_split(
    train_data,
    random_state=42,
    test_size=0.2
)

batch_size=32 #Nvidia = 4, LeNet=128
steps_per_epoch = int(np.ceil((len(train_data)//batch_size)))
val_steps = int(np.ceil((len(valid_data)//batch_size)))
train_gen = data_generator(train_data)#load_and_sort_data(train_data)
valid_gen = data_generator(valid_data)#load_and_sort_data(valid_data) 
test_gen = data_generator(test_data)#load_and_sort_data(test_data) 


# 

if LeNet:
    model = Ayo_LeNet_Model()
    print("Using LeNet Model")
else:
    print("Using Nvidia's model")
    model = Nvidia_Model()

model.compile(
    optimizer="adam",
    loss="mse"
    #metrics=["mse"]
)

model.fit(
    train_gen,
    epochs=13,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_gen,
    validation_steps=val_steps
)
print("Training Completed")
print("Test started")

test_loss= model.evaluate(test_gen)
print(f"Test loss {test_loss}")# and test accuracy {test_accuracy}")

model.summary()

if LeNet:
    model.save("Ayo_model.h5")
else:
    model.save("Nvidia_model.h5")
