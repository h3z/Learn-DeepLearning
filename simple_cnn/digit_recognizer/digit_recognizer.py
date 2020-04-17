from tensorflow import keras
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, \
    BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

Y = df_train.pop('label')
Y = to_categorical(Y, 10)
X_train = df_train.values.reshape((-1, 28, 28, 1)) / 255.
X_test = df_test.values.reshape((-1, 28, 28, 1)) / 255.

model = keras.models.Sequential()
model.add(Conv2D(32, 3, input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, 3))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(128, 3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(128, 5, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

imgGenerator = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y)

batch_size = 64
epochs = 50
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
history = model.fit_generator(imgGenerator.flow(X_train, Y_train, batch_size=batch_size),
                              epochs=epochs,
                              # 生成的图片是无限的，所以指定每epoch训练几次就可以。每次用多少数据的batch_size在generator里
                              # 这个step对应源码training_v2里117那行while step < target_steps:
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              validation_data=(X_valid, Y_valid),  # 没有提供比例参数项，还是因为生成的数据可能是无限的
                              callbacks=[annealer])

model.save('simple_cnn_data_augmentation.h5')
print('model save')
