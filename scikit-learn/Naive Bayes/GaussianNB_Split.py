from scipy.io import arff # 提供加載和解析 ARFF 文件的功能
import pandas as pd
# 載入資料 
data = arff.loadarff('hypothyroid_modified_cjlin.arff')
df = pd.DataFrame(data[0]) # 建立二維的資料表格 ( 方便對資料做操作 )

# Artribute ( 訓練樣本集的列 ) 移除分類類別
data_x = df.drop(columns=['Class'])
# 目標變量 ( 模型評估和預測 )
data_y = df['Class']

# 將類別特徵進行one hot decode。
# 將原來的類別 Artribute 被拆分成多個二元 Artribute，方便模型進行訓練
data_x = pd.get_dummies(data_x)

# 將資料集劃分為 訓練集(80%) 和 測試集(20%)
# 隨機種子為42 確保每次分割資料時都得到相同的結果
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

from sklearn.naive_bayes import GaussianNB
# 實例化 GaussianNB 作為分類器
clf = GaussianNB()
# 實例化 LabelEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
# 將類別型變量轉換為數值型變量 將每個類別映射到一個整數值
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)

# 訓練模型
clf.fit(x_train, y_train_encoded)

# 進行預測
y_pred = clf.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test_encoded, y_pred)
print("Accuracy:", accuracy)