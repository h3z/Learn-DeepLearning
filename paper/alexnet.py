from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization,Conv2D, MaxPooling2D, Dense, Dropout, Flatten


def AlexNet():
    return keras.models.Sequential([
        # layer1
        Conv2D(96, 11, strides=4, activation='relu', input_shape=(227, 227, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3), strides=2),

        # layer 2
        Conv2D(256, 5, activation='relu', padding='SAME'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3), strides=2),

        # layer 3
        Conv2D(384, 3, activation='relu', padding='SAME'),

        # layer 4
        Conv2D(384, 3, activation='relu', padding='SAME'),

        # layer 5
        Conv2D(256, 3, activation='relu', padding='SAME'),
        MaxPooling2D(pool_size=(3, 3), strides=2),

        # layer 6
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),

        # layer 7
        Dense(4096, activation='relu'),
        Dropout(0.5),

        # layer 8
        Dense(1000, activation='softmax')
    ])
