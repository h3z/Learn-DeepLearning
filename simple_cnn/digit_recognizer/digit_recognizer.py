import tensorflow.keras as keras
import pandas as pd
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Activation, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def graph(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    epochs = range(len(loss_values))

    line1 = plt.plot(epochs, loss_values, label='loss')
    line2 = plt.plot(epochs, history_dict['val_loss'], label='val loss')
    plt.setp(line1, linewidth=1.0, marker='+', markersize=1.0)
    plt.setp(line2, linewidth=1.0, marker='4', markersize=1.0)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

    line1 = plt.plot(epochs, history_dict['val_accuracy'], label='Validation/Test Accuracy')
    line2 = plt.plot(epochs, history_dict['accuracy'], label='Training Accuracy')
    plt.setp(line1, linewidth=1.0, marker='+', markersize=1.0)
    plt.setp(line2, linewidth=1.0, marker='4', markersize=1.0)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

Y = df_train.pop('label')
Y = to_categorical(Y, 10)
X_train = df_train.values.reshape((-1, 28, 28, 1)) / 255.
# X_test = df_test.values.reshape((-1, 28, 28, 1)) / 255.

X_train, X_test, y_train, y_test = train_test_split(X_train, Y, test_size=0.1, random_state=2)

model = keras.models.Sequential()
model.add(Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10, activation='softmax'))

model.summary()

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=64,
                    epochs=100,
                    validation_data=(X_test, y_test))

model.save('simple_cnn.h5')
print('model save')

graph(history)
