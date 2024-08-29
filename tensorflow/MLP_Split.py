import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 載入數據
data = arff.loadarff('hypothyroid_modified_cjlin.arff')
df = pd.DataFrame(data[0])

# 特徵和目標變量
X = df.drop(columns=['Class'])
y = df['Class']

# 將特徵進行獨熱編碼
X = pd.get_dummies(X)

# 對目標變量進行標籤編碼
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 將特徵和目標變量轉換為 NumPy 數組
X = np.array(X)
y = np.array(y)

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定義 MLP 模型 
# 創建一個序列模型，並添加三個全連接層（Dense layers）
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam', # 使用Adam優化器
              loss='categorical_crossentropy', # 損失函數為分類交叉熵
              metrics=['accuracy']) # 評估指標為準確率（accuracy）

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

import tensorflow as tf
# 對目標數據進行 one-hot 編碼
y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=4)

# 訓練模型
history = model.fit(X_train, 
                    y_train_encoded, 
                    epochs=10, # 訓練10個epochs
                    batch_size=32, # 每個batch包含32個樣本
                    validation_data=(X_test, y_test_encoded))

test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print('Test accuracy:', test_acc)