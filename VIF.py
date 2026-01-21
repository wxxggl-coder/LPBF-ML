import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# 1.读取数据
df = pd.read_csv('11.csv')

# 2.选择需要计算 VIF 的特征
features = ['Ti', 'B', 'W', 'V']
X = df[features].copy()

# 3.数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

# 4.计算 VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X_scaled.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled.values, i)
                   for i in range(X_scaled.shape[1])]

# 5.输出结果
print("各特征的方差膨胀因子（VIF）如下：")
print(vif_data)

# 一般判断标准：
# VIF < 5 ： 无明显多重共线性（可接受）
# 5 <= VIF < 10 ： 存在中等共线性（需关注）
# VIF >= 10 ： 严重共线性（建议剔除或合并特征）
