from tensorflow import keras
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, \
    BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

Y = df_train.pop('label')
Y = to_categorical(Y, 10)
X_train = df_train.values.reshape((-1, 28, 28, 1)) / 255.
X_test = df_test.values.reshape((-1, 28, 28, 1)) / 255.

nets = 15
models = [keras.models.Sequential()] * 15
for i in range(nets):
    models[i] = keras.models.Sequential()
    models[i].add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    models[i].add(BatchNormalization())
    models[i].add(Conv2D(32, kernel_size=3, activation='relu'))
    models[i].add(BatchNormalization())
    models[i].add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    models[i].add(BatchNormalization())
    models[i].add(Dropout(0.4))

    models[i].add(Conv2D(64, kernel_size=3, activation='relu'))
    models[i].add(BatchNormalization())
    models[i].add(Conv2D(64, kernel_size=3, activation='relu'))
    models[i].add(BatchNormalization())
    models[i].add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    models[i].add(BatchNormalization())
    models[i].add(Dropout(0.4))

    models[i].add(Conv2D(128, kernel_size=4, activation='relu'))
    models[i].add(BatchNormalization())
    models[i].add(Flatten())
    models[i].add(Dropout(0.4))
    models[i].add(Dense(10, activation='softmax'))

    models[i].compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])

imgGenerator = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
batch_size = 64
epochs = 50
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
history = [0] * nets
for i in range(nets):
    X_train2, X_valid, Y_train, Y_valid = train_test_split(X_train, Y)
    history[i] = models[i].fit_generator(imgGenerator.flow(X_train2, Y_train, batch_size=batch_size),
                                         epochs=epochs,
                                         # 生成的图片是无限的，所以指定每epoch训练几次就可以。每次用多少数据的batch_size在generator里
                                         # 这个step对应源码training_v2里117那行while step < target_steps:
                                         steps_per_epoch=X_train2.shape[0] // batch_size,
                                         validation_data=(X_valid, Y_valid),  # 没有提供比例参数项，还是因为生成的数据可能是无限的
                                         callbacks=[annealer])

results = np.zeros((X_test.shape[0], 10))
for i in range(nets):
    results = results + models[i].predict(X_test)
results = np.argmax(results, axis=1)
results = pd.Series(results, name='Label')
submission = pd.concat([pd.Series(range(1, 28001), name='ImageId'), results], axis=1)
submission.to_csv('MNIST-CNN-ENSEMBLE.csv', index=False)
