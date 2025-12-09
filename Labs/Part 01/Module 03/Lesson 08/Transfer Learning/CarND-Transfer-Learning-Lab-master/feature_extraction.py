import pickle
from absl import app, flags
import tensorflow as tf
from keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

# TODO: import Keras layers you need here
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    shape = X_train.shape[1:]

    

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    inp = Input(shape=shape)
    Flat = Flatten()(inp)
    fc1 = Dense(10, activation="softmax")(Flat)
    model = Model(inp, fc1)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
        loss = "categorical_crossentropy",
        metrics=["accuracy"]
    )

    # TODO: train your model here
    model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=5,
        validation_split=0.2
    )
    test_loss, test_accuracy = model.evaluate(X_val, y_val)

    print(f"Test accuracy {test_accuracy}, Test loss {test_loss}")


# parses flags and calls the `main` function above
if __name__ == '__main__':
    app.run(main)
