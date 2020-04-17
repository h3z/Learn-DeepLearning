import tensorflow.keras as keras
import pandas as pd
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Activation, Flatten, Dropout, MaxPooling2D, \
    BatchNormalization
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import math

# def get_callback():
#     def decay(epoch):
#         if epoch < 10:
#             return 0.01
#         elif epoch < 50:
#             return 0.001
#         else:
#             return 0.01 / epoch
#
#     lr_scheduler = LearningRateScheduler(decay)
#     earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
#     return [lr_scheduler, earlyStopping]


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
model.add(Conv2D(256, 5, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

history = model.fit(X_train, Y,
                    batch_size=3780,
                    epochs=300,
                    validation_split=0.1)

model.save('simple_cnn_2.h5')
print('model save')
