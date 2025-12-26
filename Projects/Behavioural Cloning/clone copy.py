import csv
import random
import matplotlib.image as mpimg

from tensorflow import keras
from keras.layers import Lambda, Dense, Convolution2D, LayerNormalization, BatchNormalization
from keras.models import Sequential
from sklearn.model_selection import train_test_split

p_dir = "./data"
csv_file = p_dir + "/driving_log.csv"
images_dir = p_dir + "IMG/"

# Loading the csv file
def csv_read():
    lines = []
    with open(csv_file, "r") as csv_data:
        csv_data = csv.reader(csv_data)
        for line in csv_data:
            for member in line[0:3]:
                lines.append([member, line[3], line[4], line[5], line[6]])
    return lines


def retrieve_images(image_paths):
    images = []
    for image_path in image_paths:
        image = mpimg.imread(image_path)
        images.append((image_path, image))
    return images


def load_and_sort_data(lines_data):
    image_data = {
        "center": [],
        "left": [],
        "right": []
    }

    image_label = {
        "Steering Angle": [],
        "Throttle": [],
        "Break": [],
        "Speed": []
    }

    aug_image_data = {
        "center": [],
        "left": [],
        "right": []
    }

    aug_image_label = {
        "Steering Angle": [],
        "Throttle": [],
        "Break": [],
        "Speed": []
    }
    for line in lines_data:
        #images = retrieve_images(line[0])
        #for image in image_path:
        for member in line:
            image = mpimg.imread(member[0])
            if random.random() > 0.5:
                image = cv2.flip(image[1])
                line[3] = -1.0 * float(line[3])
            if "center" in image[0]:
                image_data["center"].append(image[1])
                #aug_image_data["center"].append(image[1])
            elif "left" in image[0]:
                image_data["left"].append(image[1])
                #aug_image_data["left"].append(image[1])
            elif "right" in image[0]:
                image_data["right"].append(image[1])
                #aug_image_data["right"].append(image[1])
            else:
                raise "Image sort error"
            image_label["Steering Angle"].append(float(line[3]))
            image_label["Throttle"].append(float(line[4]))
            image_label["Break"].append(float(line[5]))
            image_label["Speed"].append(float(line[6]))
        #aug_image_label["Steering Angle"].append(float(line[3]) * -1.0)
        #aug_image_label["Throttle"].append(float(line[4]))
        #aug_image_label["Break"].append(float(line[5]))
        #aug_image_label["Speed"].append(float(line[6]))
    return image_data, image_label#, aug_image_data, aug_image_label


def batching(data_sample, batch_size):
    image_data = data_sample[]
    yield batch_X, batch_Y

def data_generator(sample, batch_size=1):
    length = len(sample)
    for i in range(length):
        j = i+batch_size+1
        image_data, image_label = load_and_sort_data(sample[i:j])
        image_data = np.concatenate(image_data["center"],image_data["left"],image_data["right"])
        image_label = np.concatenate(image_label["Steering Angle"],image_label["Steering Angle"],image_label["Steering Angle"])
        i = j
        yield batch_X, batch_Y


class Ayo_LeNet_Model(keras.Model):
    def __init__(self):
        super().__init__()

        self.aug = Sequential(
            Randomflip("horizontal")
        )

        self.model = Sequential(
            data_augmentation,

            # Conv Layer 1

            # Conv Layer 2

            # Conv Layer 3

            # Flatten Layer 
            
            # Dense Layer 1

            # Dense Layer 2
        )

    def __call__(self, input):
        return self.model(input)



class Nvidia_Model(keras.Model):
    def __init__(self):
        super().__init__()
        
        self.model = Sequential(
            Lambda(lambda x: ),

            # Conv Layer 1

            # Conv Layer 2

            # Conv Layer 3

            # Flatten Layer 
            
            # Dense Layer 1

            # Dense Layer 2
        )

    def __call__(self, input):
        return self.model(input)


# Load the csv_file
csv_lines = csv_read()
train_data, valid_data = train_test_split(
    csv_lines,
    random_state=42,
    test_size=0.15
)
train_data, test_data = train_test_split(
    train_data,
    random_state=42,
    test_size=0.2
)

batch_size=4
steps_per_epoch = int(np.ceil((len(train_data)//batch_size)))
val_steps = int(np.ceil((len(valid_data)//batch_size)))
train_gen = load_and_sort_data(train_data)
valid_gen = load_and_sort_data(valid_data) 
test_gen = load_and_sort_data(test_data) 


# 


# Using LeNet model
model = Ayo_LeNet_Model()

# Using Nvidia model
#model = Nvidia_Model()

model.compile(
    optimizer="Adam",
    loss="MSE",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_gen=valid_gen,
    validation_steps=val_steps
)

test_loss, test_accuracy = model.evaluate(test_gen)

model.summary()

#model.save("Ayo_model.h5")
model.save("Nvidia_model.h5")