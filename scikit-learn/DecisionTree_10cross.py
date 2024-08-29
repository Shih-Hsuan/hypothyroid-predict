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

from sklearn.preprocessing import MultiLabelBinarizer
# 實例化 MultiLabelBinarizer
mlb = MultiLabelBinarizer()
# 將多標籤目標變量轉換為二進制數組表示
data_y = mlb.fit_transform(data_y)

# 實例化 DecisionTreeClassifier 作為分類器
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)

from sklearn.model_selection import cross_val_score
# 使用 cross_val_score 進行交叉驗證 ( 10 交叉驗證 )
scores = cross_val_score(clf, data_x, data_y, cv=10)  

# 输出每次交叉驗證的準確率
print("Cross-validation scores:", scores)
# 输出平均準確率
print("Average accuracy:", scores.mean())