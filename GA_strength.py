import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sko.GA import GA
import openpyxl
from sko.operators import crossover as crossover_ops

start_time = time.time()

# 读取数据
data = pd.read_csv('11.csv')

# 提取特征和目标值
x = data.iloc[:, :4].values
y = data.iloc[:, 4].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LGBMRegressor(learning_rate=0.948, max_depth=61, max_features=37, min_child_samples=11,
                      n_estimators=708, reg_alpha=7, reg_lambda=533)
# 数据归一化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
model.fit(x_train, y_train)

# 记录每次迭代的输入输出
record_data = []
eval_counter = 0
def objective_fun(params):
    global eval_counter
    X = params.reshape(1, -1)
    X_scale = scaler.transform(X)  # 将参数数组标准化
    Y = model.predict(X_scale)
    y_pred = Y[0]
    iteration = eval_counter // ga.size_pop + 1
    record_data.append(list(params)+[y_pred,iteration])
    eval_counter += 1
    return -y_pred  # 返回负的预测值

# 包装器：以概率 prob_cross 执行原交叉算子，否则返回原染色体（不交叉）
def crossover_with_prob(algorithm, prob_cross=0.8):
    if np.random.rand() < prob_cross:
        return crossover_ops.crossover_2point(algorithm)
    else:
        return algorithm.Chrom

ga = GA(func=objective_fun, n_dim=4, size_pop=50, max_iter=100,  prob_mut=0.01,
        lb=[0.01, 0.01, 200, 800], ub=[5, 3, 400, 1700], precision=1e-7)

# 注册我们带概率的交叉算子（register 会把 keyword args 传给 operator）
ga.register(operator_name='crossover', operator=crossover_with_prob, prob_cross=0.8)

best_x, best_y = ga.run()
print('best_x:', best_x, '\n','best_y:', -best_y)

# 保存输入输出记录
columns = ['Ti', 'B', 'power', 'scanning speed', 'y_pred', 'Iteration']
record_df = pd.DataFrame(record_data, columns=columns)


# 保存为 Excel 文件
save_path = 'F:/Ti、B有能量密度/优化算法寻优数据/LGBM/所有历史数据/GA_strength_11.xlsx'
record_df.to_excel(save_path, index=False)
print(f"每次迭代输入输出已保存到：{save_path}")

# 绘制随迭代变化的曲线
plt.figure(figsize=(8, 5))
plt.plot(record_df['Iteration'], record_df['y_pred'], 'o', markersize=3, alpha=0.5, label='All samples')
plt.plot(record_df.groupby('Iteration')['y_pred'].max(), '-r', label='Best in each iteration')
plt.xlabel('Iteration')
plt.ylabel('Predicted Y')
plt.title('GA Optimization Process (LGBM Prediction)')
plt.legend()
plt.grid(True)
plt.show()

# 绘制历史最优曲线
Y_history = pd.DataFrame(-np.array(ga.all_history_Y))
plt.figure()
plt.plot(Y_history.index, Y_history.max(axis=1).cummax(), '-b')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness (Y)')
plt.title('Evolution of Best Fitness')
plt.show()

#  绘制每个参数的收敛曲线
param_names = ['Ti', 'B', 'power', 'scanning speed']

plt.figure(figsize=(10, 8))
for i, param in enumerate(param_names, 1):
    plt.subplot(2, 2, i)
    plt.plot(record_df['Iteration'], record_df[param], 'o', alpha=0.5, markersize=3)
    plt.plot(record_df.groupby('Iteration')[param].mean(), '-r', linewidth=2, label='Mean per iteration')
    plt.xlabel('Iteration')
    plt.ylabel(param)
    plt.title(f'{param} Convergence Curve')
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.show()

end_time =time.time()
execution_time = end_time - start_time
print('所用时间：{:2f}秒'.format(execution_time))
