from tensorflow import keras
from keras.layers import Lambda, Dense, Convolution2D, LayerNormalization, BatchNormalization, Activation, MaxPooling2D, Dropout, Cropping2D, Flatten
from keras.models import Sequential


class Ayo_LeNet_Model(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = Sequential([
            Lambda(lambda x: (x-128) / 128, input_shape=(160, 320, 3)),

            Cropping2D(cropping=((50, 27), (0, 0))),

            # Conv Layer 1
            Convolution2D(16, (3, 3)),
            BatchNormalization(),
            Activation("relu"),

            # Conv Layer 2
            Convolution2D(32, (3, 3)),
            BatchNormalization(),
            Activation("relu"),

            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            # Conv Layer 3
            Convolution2D(64, (3, 3)),
            BatchNormalization(),
            Activation("relu"),

            # Conv ayer 4
            Convolution2D(128, (3, 3)),
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

    def __call__(self, input):
        return self.model(input)



class Nvidia_Model(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.model = Sequential([

            Lambda(lambda x: (x-128) / 128, input_shape=(160, 320, 3)),

            Cropping2D(cropping=((50, 27), (0, 0))),

            # Conv Layer 1
            Convolution2D(24, (5, 5)),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            # Conv Layer 2
            Convolution2D(36, (5, 5)),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            # Conv Layer 3
            Convolution2D(48, (5, 5)),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            # Conv Layer 4
            Convolution2D(64, (3, 3)),
            BatchNormalization(),
            Activation("relu"),

            # Conv Layer 5
            Convolution2D(64, (3, 3)),
            BatchNormalization(),
            Activation("relu"),

            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

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

    def __call__(self, input):
        return self.model(input)