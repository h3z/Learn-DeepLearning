import tensorflow.keras as keras
import pandas as pd
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Activation, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

'''
在笔记本上结果
Epoch 100/100
4199/4199 [==============================] - 4s 964us/sample - loss: 0.0130 - accuracy: 0.9988 - val_loss: 0.1837 - val_accuracy: 0.9486
在k10
'''

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

Y = df_train.pop('label')
Y = to_categorical(Y, 10)
X_train = df_train.values.reshape((-1, 28, 28, 1)) / 255.
X_test = df_test.values.reshape((-1, 28, 28, 1)) / 255.

model = keras.models.Sequential()
model.add(keras.layers.ZeroPadding2D(padding=(2, 2), input_shape=(28, 28, 1), name='INPUT'))
model.add(Conv2D(filters=6, kernel_size=(5, 5), name='C1'))
model.add(AveragePooling2D(pool_size=(2, 2), name='S2'))  # 原文中是 k*sum+b，包含了2个参数的。这里等于是k=1, b=0了
model.add(Activation(activation='sigmoid'))
model.add(Conv2D(filters=16, kernel_size=(5, 5), name='C3'))  # 原文中这里C3和S2的连接是有设计的，我不知道这里怎么实现
model.add(AveragePooling2D(pool_size=(2, 2), name='S4'))  # 同S2
model.add(Activation(activation='sigmoid'))
model.add(Conv2D(filters=120, kernel_size=(5, 5), name='C5'))
model.add(Flatten())
model.add(Dense(units=84, activation='tanh', name='F6'))  # 原文中的激活函数是 A*tanh(Sa)
model.add(Dense(units=10, activation='softmax', name='OUTPUT'))  # 原文中是RBF，不认识RBF是什么

print(model.summary())

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

history = model.fit(X_train, Y,
                    batch_size=128,
                    epochs=100,
                    validation_split=0.9)

print(history)
history_dict = history.history
loss_values = history_dict['loss']
epochs = range(len(loss_values))
model.save('simple_cnn.h5')
print('model save')

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
