from digit_recognizer.hnn import model
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

Y = df_train.pop('label')
Y = to_categorical(Y, 10)
X_train = df_train.values.reshape((-1, 784)) / 255.
# X_test = df_test.values.reshape((-1, 28, 28, 1)) / 255.

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y, test_size=0.1)
print(X_train.shape)
print(Y_train.shape)

model(X_train.T, Y_train.T, layer_dims=[784, 10])
