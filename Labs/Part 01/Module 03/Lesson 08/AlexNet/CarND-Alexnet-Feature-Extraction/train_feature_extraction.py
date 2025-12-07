import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import pandas as pd
import numpy as np
import cv2

# TODO: Load traffic signs data.
sign_names = pd.read_csv("signnames.csv")
nb_classes = len(sign_names)
print(nb_classes)
with open("./AlexNet Dataset/train.p", "rb") as f:
    data = pickle.load(f)

X_train, y_train = data["features"], data["labels"]

#X_train, y_train = X_train[0:1000], y_train[0:1000]


# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train,
    random_state=42,
    test_size=0.3
)

n_train = len(X_train)
n_test = len(X_test)
print(f"Number of train is {n_train}")
print(f"Number of test is: {n_test}")

mean = np.mean(X_train)
std = np.std(X_train)
print(f"Mean and std calculated as {mean} {std} respectively")

# TODO: Define placeholders and resize operation.
def resize(image):
    return tf.image.resize(image, (227, 227))

#X_train, X_test = resize(X_train), resize(X_test)

def normalize(img):
    return (img - mean) / std

#X_train, X_test = normalize(X_train), normalize(X_test)
#print("Data normalized")

def preprocess(images):#, label):
    processed = []
    for image in images:
        image = normalize(resize(image))
        processed.append(image)
    processed = tf.Variable(processed)
    return processed#image, label

"""
print(X_train.shape)
X_train = preprocess(X_train)
print("Train Preprocessed")
print(X_train.shape)
X_test = preprocess(X_test)
print("Test Preprocessed")
"""


# Parameters
batch_size = 32
epochs = 10
learning_rate = 0.001
training_steps = int(np.ceil(len(X_train)/batch_size))

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#train_data = train_data.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
#print("Resized and Normalized")
train_data = train_data.shuffle(n_train).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
print("Data now as tensors")
#test_data = test_data.shuffle(n_train)


test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
#test_data = test_data.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.shuffle(n_test).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# TODO: pass placeholder as first argument to `AlexNet`.
#fc7 = AlexNet(X_train, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
#fc7 = tf.stop_gradient(fc7)

#shape = (fc7.get_shape.as_list()[-1], nb_classes)
#print(shape)
W = tf.Variable(tf.random.normal([4096, nb_classes]), trainable=True)
B = tf.Variable(tf.random.normal([nb_classes]), trainable=True)

# TODO: Add the final layer for traffic sign classification.
def Alex_mod(X_train):
    fc7 = AlexNet(X_train, feature_extract=True)
    fc7 = tf.stop_gradient(fc7)
    #shape = (fc7.get_shape().as_list()[-1], nb_classes)
    logits = tf.nn.bias_add(tf.matmul(fc7, W), B)
    return logits


# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

optimizer = tf.optimizers.Adam(learning_rate)


def ce_loss(pure_logits, labels):
    labels = tf.cast(labels, tf.int64)
    err = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pure_logits, labels=labels)
    return tf.reduce_mean(err)

def training(batch_X, batch_Y):
    with tf.GradientTape() as g:
        logits = Alex_mod(batch_X)
        loss = ce_loss(logits, batch_Y)

    gradients = g.gradient(loss, [W, B])
    optimizer.apply_gradients(zip(gradients, [W, B]))

def accuracy(pred, labels):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.cast(labels, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)



# TODO: Train and evaluate the feature extraction model.
for epoch in range(1, epochs+1):
    for step, (batch_X, batch_Y) in enumerate(train_data.take(training_steps)):
        batch_X = preprocess(batch_X)
        training(batch_X, batch_Y)

    t_logits = Alex_mod(batch_X)
    t_loss = ce_loss(t_logits, batch_Y)
    t_probs = tf.nn.softmax(t_logits)
    t_acc = accuracy(t_probs, batch_Y)
    #print(f"Epoch {epoch}, train loss {t_loss}, train accuracy {t_acc}")

    all_preds = []
    all_labels = []
    all_logits = []
    for s, (batch_tx, batch_ty) in enumerate(test_data.take(training_steps)):
        batch_tx = preprocess(batch_tx)
        v_logits = Alex_mod(batch_tx, batch_ty)
        #v_loss = ce_loss(v_logits, batch_ty)
        v_probs = tf.nn.softmax(v_logits)
        #v_acc = accuracy(v_probs, batch_ty)
        all_preds.append(v_probs)
        all_labels.append(batch_ty)
        all_logits.append(v_logits)

    v_logits = tf.concat(all_logits, axis=0)
    v_probs = tf.concat(all_preds, axis=0)
    v_labels = tf.concat(all_labels, axis=0)

    v_loss = ce_loss(v_logits, v_labels)
    v_acc = accuracy(v_probs, v_labels)

    print(f"Epoch {epoch}, train loss {t_loss}, train accuracy {t_acc}, valid loss {v_loss}, valid accuracy {v_acc}")
