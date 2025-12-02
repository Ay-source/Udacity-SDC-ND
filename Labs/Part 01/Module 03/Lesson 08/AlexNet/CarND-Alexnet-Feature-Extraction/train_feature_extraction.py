import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import pandas as pd

# TODO: Load traffic signs data.
sign_names = pd.read_csv("signnames.csv")
nb_classes = len(sign_names)
print(nb_classes)
with open("./AlexNet\ Dataset/train.p", "rb") as f:
    data = pickle.load(f)

X_data, y_data = data["features"], data["labels"]


# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data,
    shuffle=42,
    test_size=0.3
)

# TODO: Define placeholders and resize operation.
def resize(images, ):
    resized = []
    for image in images:
        resized.append(tf.image.resize(image, [227, 227]))
    return resized

X_train, X_test = resize(X_train), resize(X_test)
mean = np.mean(X_train)
std = np.stddev(X_train)

def normalize(imgs):
    return (imgs - 128) / 128

X_train, X_test = normalize(X_train), normalize(X_test)


# Parameters
batch_size = 10
epochs = 10
learning_rate = 0.001


# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(X_train, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape.as_list()[-1], nb_classes)
W = tf.Variable(tf.random.normal(shape), trainable=True)
B = tf.Variable(tf.random.normal([shape[-1]]), trainable=True)
logits = tf.nn.bias_add(tf.matmul(fc7, W), B)
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
def loss(logits, y_actual):
    err = tf.nn.sparse_categorical_softmax_with_logits()
    return error

define training():
    return None

def accuracy(y_pred, y_actual):
    return acc

# TODO: Train and evaluate the feature extraction model.
