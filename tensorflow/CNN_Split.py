import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd

data = arff.loadarff('hypothyroid_modified_cjlin.arff')
df = pd.DataFrame(data[0])

data_x = df.drop(columns=['Class'])
data_y = df['Class']
data_x = pd.get_dummies(data_x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

# 載入 MNIST 數據集
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 對數據進行正規化和形狀調整
# 將像素值縮放到0到1之間，並將資料的形狀調整為(28, 28, 1)的格式，以符合CNN的輸入要求
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

# 將類別標籤進行 one-hot 編碼 (二進制向量)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# 定義 CNN 模型
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    # 添加卷積層和池化層
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # 展平特徵圖
    model.add(layers.Flatten())
    # 添加全連接層
    model.add(layers.Dense(64, activation='relu'))
    # 輸出層
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# 指定輸入形狀和類別數量
input_shape = (28, 28, 1)  # MNIST 數據集的輸入形狀為 (28, 28, 1)
# 對於 MNIST 數據集，每個圖像都屬於 0 到 9 中的一個類別
num_classes = 10  # 類別數量為 10

# 創建模型
model = create_cnn_model(input_shape, num_classes)

# 編譯模型
model.compile(optimizer='adam', # 使用Adam優化器
              loss='categorical_crossentropy', # 損失函數為分類交叉熵
              metrics=['accuracy']) # 評估指標為準確率

# 訓練模型
history = model.fit(x_train, y_train, 
                    epochs=5, # 訓練5個epochs
                    batch_size=64, # 每個batch包含64個樣本
                    validation_data=(x_test, y_test))

# 評估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)