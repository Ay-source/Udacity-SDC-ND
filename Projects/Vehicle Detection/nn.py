from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LayerNormalization, Input, Dropout, Resizing, Rescaling, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten

class MyModel(Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)

        self.model = Sequential([
            Input(shape=(16320, )),
            Dense(1024, activation = "relu"),
            LayerNormalization(),
            Dense(256, activation = "relu"),
            LayerNormalization(),
            Dense(64, activation = "relu"),
            LayerNormalization(),
            Dense(16, activation = "relu"),
            LayerNormalization(),
            Dense(2, activation="softmax")
        ])

    def __call__(self, X):
        model = self.model(X)
        return model


class CNNModel(Model):
    def __init__(self, **kwargs):
        super(CNNModel, self).__init__(**kwargs)

        self.model = Sequential([

        # Preprocessing Layer
        Input(shape=(64, 64, 3)),
        Resizing(height=32, width=32),
        #Rescaling(1/128, offset=-128/128),

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
        Dense(2)
        ])

    def __call__(self, X):
        model = self.model(X)
        return model