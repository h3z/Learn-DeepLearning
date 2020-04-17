import tensorflow.keras as keras
import pandas as pd
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Activation, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as PLT
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping




df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

Y = to_categorical(df_train.pop('label'), 10)
X = df_train.values.reshape((-1, 28, 28, 1)) / 255.
test = df_test.values.reshape((-1, 28, 28, 1)) / 255.


