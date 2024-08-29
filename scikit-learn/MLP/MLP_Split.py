from scipy.io import arff
import pandas as pd
data = arff.loadarff('hypothyroid_modified_cjlin.arff')
df = pd.DataFrame(data[0])

# Artribute 特徵
X = df.drop(columns=['Class'])
# 目標變量
y = df['Class']
# 將分類變量轉換為 哑變量 / 獨熱編碼
X = pd.get_dummies(X)

from sklearn.model_selection import train_test_split
# 將數據集劃分為訓練集和測試集 random_state: 確保每次運行時劃分的訓練集和測試集是相同的
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.neural_network import MLPClassifier
# 創建 MLPClassifier 實例
mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)

from sklearn.preprocessing import LabelEncoder
# 實例化 LabelEncoder
label_encoder = LabelEncoder()
# 將類別型變量轉換為數值型變量
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)

# 訓練模型
mlp_clf.fit(X_train, y_train_encoded)
# 進行預測
y_pred = mlp_clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test_encoded, y_pred)
print("Accuracy:", accuracy)