import tensorflow.keras as keras
from nn.data_util import load_data
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import normalize

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()

train_X = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)

train_X = normalize(train_X)

inputs = Input(shape=(train_X.shape[1],), name='A0')

X = Dense(units=7, activation='relu', name='hidden_layer_0')(inputs)

outputs = Dense(units=1, activation='sigmoid', name='hidden_layer_1')(X)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0075),
              loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

model.summary()

model.fit(train_X, train_set_y_orig.T, epochs=120)
