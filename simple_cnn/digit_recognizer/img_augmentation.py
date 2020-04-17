import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

df_train = pd.read_csv("train.csv")
labels = df_train.pop('label')
X = df_train.values.reshape((-1, 28, 28, 1))

# rotation_range 10°    zoom_range [1-0.1, 1+0.1]
imgGenerator = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)

count = 0
# 生成的图片是无限多的…… X形状是，个数 * 图片 ， y形状是个数*label，
for i in imgGenerator.flow(X, labels):
    count += 1
    if count > 10:
        break
    plt.imshow(i[0][0].reshape(28, 28))  # 生成的图片
    print(i[1][0])  # label
