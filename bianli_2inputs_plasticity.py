import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import openpyxl
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_csv('11.csv')

# 提取特征和目标值
X = data.iloc[:, :4].values
y = data.iloc[:, 5].values

# 初始化一个字典来存储每个模型的性能
model_r2 = {}
model_a = {}
model_sqrt = {}
# 数据归一化
scaler = StandardScaler()

# 遍历所有可能的四个特征的组合
for i in range(X.shape[1]):
    for j in range(i + 1, X.shape[1]):
        # 创建一个特征子集
        feature_subset = [i, j]
        # 使用这四个特征训练模型
        model = LGBMRegressor()
        n_runs = 100
        all_prediction_y = []
        for n in range(n_runs):
            x_train_true, x_test_true, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            x_train = scaler.fit_transform(x_train_true)
            x_test = scaler.transform(x_test_true)
            model.fit(x_train[:, feature_subset], y_train)
            # 在测试集上进行预测
            y_pred = model.predict(x_test[:, feature_subset])
            # 计算R2分数
            all_prediction_y.append(y_pred)
        average_prediction_y = np.mean(all_prediction_y, axis=0)
        # 计算平均预测值的准确度（R2分数）和均方根误差
        accuracy = r2_score(y_test, average_prediction_y)
        sqrterror1 = math.sqrt(mean_squared_error(y_test, average_prediction_y))
        S_accuracy = 1 - np.mean(np.abs(average_prediction_y - y_test) / y_test)
        # 将性能结果存储在字典中
        model_r2[tuple(feature_subset)] = accuracy
        model_a[tuple(feature_subset)] = S_accuracy
        model_sqrt[tuple(feature_subset)] = sqrterror1



# 找出性能最好的四个特征组合
best_features = max(model_r2, key=model_r2.get)
best_r2 = model_r2[best_features]

print(f"Best features: {best_features}")
print(f"Best R2 score: {best_r2}")

print(model_r2)
print(model_a)
print(model_sqrt)