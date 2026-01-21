import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

#1.读取数据
data = pd.read_csv("11.csv",encoding='gbk')
para = np.array(data.iloc[:, :])  #转为矩阵
#2.求相关系数
para_corr = np.corrcoef(para,rowvar=False)
para_corr = pd.DataFrame(data=para_corr, columns=['Ti', 'B', 'power','speed','strength','elongation'],
                         index=['Ti', 'B', 'power','speed','strength','elongation'])
#求p-value
# 初始化一个与 para 形状相同的数组来存储 p-value
p_values = np.zeros((6,6))
# 计算每对列之间的 p-value
for i in range(6):
    for j in range(i + 1, 6):
        p_value = pearsonr(para[:, i], para[:, j])[1]
        p_values[i][j] = p_value
        p_values[j][i] = p_value

print(para_corr)
print(p_values)
#3.画相关系数热力图
ax = sns.heatmap(para_corr,square=True, annot=True, fmt='.3f',
                 linewidth=1, cmap='coolwarm',linecolor='white', cbar=True,
                 annot_kws={'size': 16, 'weight': 'normal', 'color': 'black'},
                 cbar_kws={'fraction': 0.046, 'pad': 0.03})

plt.rcParams['font.sans-serif'] = ['Arial']    # 设置字体
plt.xticks(fontsize=17, fontweight='bold')
plt.yticks(fontsize=17, fontweight='bold')
plt.xticks(rotation=0)  #倾斜45度
plt.yticks(rotation=90)  #倾斜45度
plt.show()



