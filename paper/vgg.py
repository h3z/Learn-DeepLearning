from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dense, Dropout, Flatten


def VGG(is19):
    model = keras.models.Sequential()

    # layer1,2
    model.add(Conv2D(64, 3, activation='relu', padding='SAME', input_shape=(224, 224, 3)))
    model.add(Conv2D(64, 3, activation='relu', padding='SAME'))
    model.add(MaxPooling2D())

    # layer3,4
    model.add(Conv2D(128, 3, activation='relu', padding='SAME'))
    model.add(Conv2D(128, 3, activation='relu', padding='SAME'))
    model.add(MaxPooling2D())

    # layer5,6,7
    model.add(Conv2D(256, 3, activation='relu', padding='SAME'))
    model.add(Conv2D(256, 3, activation='relu', padding='SAME'))
    model.add(Conv2D(256, 3, activation='relu', padding='SAME'))
    if is19:
        model.add(Conv2D(256, 3, activation='relu', padding='SAME'))
    model.add(MaxPooling2D())

    # layer8,9,10
    model.add(Conv2D(512, 3, activation='relu', padding='SAME'))
    model.add(Conv2D(512, 3, activation='relu', padding='SAME'))
    model.add(Conv2D(512, 3, activation='relu', padding='SAME'))
    if is19:
        model.add(Conv2D(512, 3, activation='relu', padding='SAME'))
    model.add(MaxPooling2D())

    # layer11,12,13
    model.add(Conv2D(512, 3, activation='relu', padding='SAME'))
    model.add(Conv2D(512, 3, activation='relu', padding='SAME'))
    model.add(Conv2D(512, 3, activation='relu', padding='SAME'))
    if is19:
        model.add(Conv2D(512, 3, activation='relu', padding='SAME'))
    model.add(MaxPooling2D())

    # layer 14,15,16
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='softmax'))

    return model


print('vgg16:')
VGG(False).summary()
print('vgg19:')
VGG(True).summary()
