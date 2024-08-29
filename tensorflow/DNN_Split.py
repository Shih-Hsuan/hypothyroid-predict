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

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)
y_train_encoded

from keras.utils import to_categorical
# 將數字表示的目標變量轉換為 one-hot 編碼格式
y_train_categorical = to_categorical(y_train_encoded,4)
y_test_categorical = to_categorical(y_test_encoded,4)

# 將特徵和目標變量轉換為 NumPy 數組
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train_encoded = np.array(y_train_categorical)
y_test_encoded = np.array(y_test_categorical)

model = tf.keras.Sequential()
# 向模型中添加了三個全連接層（Dense layers），分別具有 16、8 和 4 個神經元。
# 第一層需要指定輸入形狀（input_shape），即特徵的形狀
model.add(tf.keras.layers.Dense(16, input_shape=(x_train.shape[1],), activation='sigmoid'))
model.add(tf.keras.layers.Dense(8, activation='sigmoid'))
model.add(tf.keras.layers.Dense(4, activation='softmax'))
model.summary()


# 創建一個Adam優化器的實例 ， 指定學習率的初始值 = 0.001
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
# 將優化器、損失函數和評估指標添加到模型中
# 1. 使用 Adam 優化器（optimizer）編譯模型
# 2. 指定損失函數（loss function）為分類交叉熵（Categorical Crossentropy）
# 3. 加入評估指標，這裡使用的是分類準確率（Categorical Accuracy）
model.compile(optimizer=opt, # from_logits = True表示模型的輸出未經過softmax激活函數轉換
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[tf.metrics.CategoricalAccuracy(name='accuracy')])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# 模型的訓練過程
history = model.fit(x_train, # 訓練資料集的特徵資料
            y_train_encoded,# 訓練資料集的目標
            epochs=40, # 進行 40 輪訓練
            batch_size=20, # 每輪訓練使用 20 個樣本
            # 在每個訓練輪次結束時，使用測試資料集來驗證模型的表現
            validation_data=(x_test, y_test_encoded),  
            verbose=1) # 以詳細模式顯示訓練過程

# 評估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)