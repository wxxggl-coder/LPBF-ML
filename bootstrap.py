import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

DATA_PATH = "11.csv"
FEATURE_COLS = ['Ti', 'B', 'power', 'speed']
TARGET_COL = 'strength'
TEST_SIZE = 0.2
RANDOM_STATE = 42

N_BOOTSTRAP = 50
SAMPLE_RATIO = 0.8


LGBM_PARAMS = {
    "n_estimators": 708,
    "learning_rate": 0.948,
    "max_depth": 61,
    "min_child_samples": 11,
    "reg_alpha": 7,
    "reg_lambda": 533
}

OUTPUT_DIR = "lgbm_bootstrap_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# 1. 读数据
# -----------------------
df = pd.read_csv(DATA_PATH)

X_df = df[FEATURE_COLS].copy()
y_ser = df[TARGET_COL].copy()

X = X_df.reset_index(drop=True)
y = y_ser.reset_index(drop=True)

# -----------------------
# 2. 划分并标准化
# -----------------------
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_test_scaled = scaler.transform(X_test_df)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_df.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_df.columns)

# -----------------------
# 3. 训练 bootstrap 模型
# -----------------------

def bootstrap_train(X_train_df, y_train, n_models=50, sample_ratio=0.8, params=None, random_state=None):
    rng = np.random.default_rng(random_state)
    models = []
    preds_train = []
    n_train = len(X_train_df)
    n_bs = int(n_train * sample_ratio)

    for i in range(n_models):
        idx = rng.choice(n_train, size=n_bs, replace=True)
        X_bs = X_train_df.iloc[idx]
        y_bs = y_train.iloc[idx]

        params_local = params.copy()
        params_local["random_state"] = int(rng.integers(1, 1e9))

        model = LGBMRegressor(**params_local)
        model.fit(X_bs, y_bs)
        models.append(model)

        # train prediction for uncertainty analysis
        preds_train.append(model.predict(X_train_df))

    return models, np.array(preds_train)


models, pred_train_matrix = bootstrap_train(
    X_train_scaled_df, y_train.reset_index(drop=True),
    n_models=N_BOOTSTRAP,
    sample_ratio=SAMPLE_RATIO,
    params=LGBM_PARAMS,
    random_state=RANDOM_STATE
)


pred_test_matrix = np.array([m.predict(X_test_scaled_df) for m in models])

# -----------------------
# 4. 不确定性
# -----------------------
y_pred_train_mean = pred_train_matrix.mean(axis=0)
y_pred_train_std = pred_train_matrix.std(axis=0)
residuals_train = y_train.values - y_pred_train_mean

y_pred_test_mean = pred_test_matrix.mean(axis=0)
y_pred_test_std = pred_test_matrix.std(axis=0)
residuals_test = y_test.values - y_pred_test_mean

rmse_train = mean_squared_error(y_train, y_pred_train_mean, squared=False)
mae_train = mean_absolute_error(y_train, y_pred_train_mean)
r2_train = r2_score(y_train, y_pred_train_mean)

rmse_test = mean_squared_error(y_test, y_pred_test_mean, squared=False)
mae_test = mean_absolute_error(y_test, y_pred_test_mean)
r2_test = r2_score(y_test, y_pred_test_mean)

print("Train RMSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}".format(rmse_train, mae_train, r2_train))
print("Test  RMSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}".format(rmse_test, mae_test, r2_test))


train_rmse_list = []
train_r2_list = []

test_rmse_list = []
test_r2_list = []

for i in range(N_BOOTSTRAP):
    # --- Train ---
    y_pred_train_i = pred_train_matrix[i]
    train_rmse_list.append(mean_squared_error(y_train, y_pred_train_i, squared=False))
    train_r2_list.append(r2_score(y_train, y_pred_train_i))

    # --- Test ---
    y_pred_test_i = pred_test_matrix[i]
    test_rmse_list.append(mean_squared_error(y_test, y_pred_test_i, squared=False))
    test_r2_list.append(r2_score(y_test, y_pred_test_i))


train_rmse_arr = np.array(train_rmse_list)
train_r2_arr = np.array(train_r2_list)

test_rmse_arr = np.array(test_rmse_list)
test_r2_arr = np.array(test_r2_list)


def ci95(arr):
    return np.percentile(arr, [2.5, 97.5])


# ★ 训练集统计
train_stats = {
    "RMSE_mean": train_rmse_arr.mean(),
    "RMSE_var": train_rmse_arr.var(),
    "RMSE_CI_low": ci95(train_rmse_arr)[0],
    "RMSE_CI_high": ci95(train_rmse_arr)[1],

    "R2_mean": train_r2_arr.mean(),
    "R2_var": train_r2_arr.var(),
    "R2_CI_low": ci95(train_r2_arr)[0],
    "R2_CI_high": ci95(train_r2_arr)[1],
}

# ★ 测试集统计
test_stats = {
    "RMSE_mean": test_rmse_arr.mean(),
    "RMSE_var": test_rmse_arr.var(),
    "RMSE_CI_low": ci95(test_rmse_arr)[0],
    "RMSE_CI_high": ci95(test_rmse_arr)[1],

    "R2_mean": test_r2_arr.mean(),
    "R2_var": test_r2_arr.var(),
    "R2_CI_low": ci95(test_r2_arr)[0],
    "R2_CI_high": ci95(test_r2_arr)[1],
}

train_metrics_df = pd.DataFrame([train_stats])
test_metrics_df = pd.DataFrame([test_stats])




# 5. 保存 Excel（原始输入值）

df_train_out = X_train_df.reset_index(drop=True).copy()
df_train_out["y_true"] = y_train.values
df_train_out["y_pred_mean"] = y_pred_train_mean
df_train_out["y_pred_std"] = y_pred_train_std
df_train_out["residual"] = residuals_train

df_test_out = X_test_df.reset_index(drop=True).copy()
df_test_out["y_true"] = y_test.values
df_test_out["y_pred_mean"] = y_pred_test_mean
df_test_out["y_pred_std"] = y_pred_test_std
df_test_out["residual"] = residuals_test


excel_path = os.path.join(OUTPUT_DIR, "bootstrap_predictions.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_train_out.to_excel(writer, sheet_name="Train", index=False)
    df_test_out.to_excel(writer, sheet_name="Test", index=False)

    train_metrics_df.to_excel(writer, sheet_name="Train_Bootstrap_Metrics", index=False)
    test_metrics_df.to_excel(writer, sheet_name="Test_Bootstrap_Metrics", index=False)
print("训练集与测试集 bootstrap RMSE/R2 统计已写入 Excel。")

# 6. 绘图

plt.rcParams['font.family'] = 'Arial'
plt.style.use('seaborn-v0_8-whitegrid')

# -------------------------------
# Fig1：True vs Predicted 散点图
# -------------------------------
# ======================================================
# ★★★ Fig1：加入测试集误差棒的散点图（修改后的版本）★★★
# ======================================================
fig1, ax1 = plt.subplots(figsize=(7, 6), dpi=300)

min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())

# ---- 训练集：无误差 ----
ax1.scatter(
    y_train,
    y_pred_train_mean,
    color='#1f77b4',
    s=40,
    alpha=0.8,
    label='Train'
)

# ---- 测试集：带误差棒 ----
ax1.errorbar(
    y_test,
    y_pred_test_mean,
    yerr=y_pred_test_std,
    fmt='o',
    markersize=6,
    color='#ff7f0e',
    ecolor='#ff7f0e',
    elinewidth=1.2,
    capsize=3,
    alpha=0.9,
    label='Test (±STD)'
)

# 理想线
ax1.plot([min_val, max_val], [min_val, max_val], '--', color='black', linewidth=1.2)

# 边框
for spine in ax1.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.2)

ax1.set_xlabel("True Value", fontsize=14)
ax1.set_ylabel("Predicted Value", fontsize=14)
ax1.tick_params(axis='both', labelsize=12, width=1.2)
ax1.legend(frameon=False, fontsize=12)
ax1.set_title("Prediction (Train vs Test with Test Uncertainty)", fontsize=15)

plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "prediction_scatter_train_test_with_errorbar.png")
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Fig1（含误差棒）已保存：{fig1_path}")

# -------------------------------
# Prediction Uncertainty (按预测值排序)
# -------------------------------
fig2, ax2 = plt.subplots(figsize=(7,6), dpi=300)

# --- 训练集排序 ---
train_sort_idx = np.argsort(y_pred_train_mean)
train_x_sorted = y_pred_train_mean[train_sort_idx]
train_y_sorted = y_pred_train_mean[train_sort_idx]
train_std_sorted = y_pred_train_std[train_sort_idx]

# 训练集误差带 + 曲线
ax2.fill_between(train_x_sorted,
                 train_y_sorted - train_std_sorted,
                 train_y_sorted + train_std_sorted,
                 color='#1f77b4', alpha=0.2, label='Train Uncertainty')
ax2.plot(train_x_sorted, train_y_sorted, color='#1f77b4', lw=1.5, label='Train Prediction')

# --- 测试集排序 ---
test_sort_idx = np.argsort(y_pred_test_mean)
test_x_sorted = y_pred_test_mean[test_sort_idx]
test_y_sorted = y_pred_test_mean[test_sort_idx]
test_std_sorted = y_pred_test_std[test_sort_idx]

# 测试集误差带 + 曲线
ax2.fill_between(test_x_sorted,
                 test_y_sorted - test_std_sorted,
                 test_y_sorted + test_std_sorted,
                 color='#ff7f0e', alpha=0.25, label='Test Uncertainty')
ax2.plot(test_x_sorted, test_y_sorted, color='#ff7f0e', lw=1.5, label='Test Prediction')

# 边框美化
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.2)

ax2.set_xlabel("Predicted Value (sorted)", fontsize=14)
ax2.set_ylabel("Predicted Value ± STD", fontsize=14)
ax2.tick_params(axis='both', labelsize=12, width=1.2)
ax2.legend(frameon=False, fontsize=12)
ax2.set_title("Prediction Uncertainty (Train vs Test)", fontsize=15)

plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, "prediction_uncertainty_sorted.png")
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Fig2 已保存：{fig2_path}")

# 残差图
fig3, ax3 = plt.subplots(figsize=(7,6), dpi=300)
# 训练集残差
ax3.scatter(y_train.values[train_sort_idx],
           residuals_train[train_sort_idx],
           color='#1f77b4',
           alpha=0.8,
           s=40,
           label='Train')

# 测试集残差
ax3.scatter(y_test.values[test_sort_idx],
           residuals_test[test_sort_idx],
           color='#ff7f0e',
           alpha=0.9,
           s=50,
           label='Test')

# 参考线 y=0
ax3.axhline(0, color='black', linestyle='--', linewidth=1.2)

# 边框美化
for spine in ax3.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.2)

ax3.set_xlabel("True Value", fontsize=14)
ax3.set_ylabel("Residual (Predicted - True)", fontsize=14)
ax3.set_ylim(-40, 40)  # <-- 设置 y 轴范围
ax3.tick_params(axis='both', labelsize=12, width=1.2)
ax3.legend(frameon=False, fontsize=12)
ax3.set_title("Residual Plot (Train vs Test)", fontsize=15)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "residual_plot_train_test.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"残差图已保存：{save_path}")

# # 训练集误差分布
# plt.figure(figsize=(8,5))
# sns.histplot(residuals_train, bins=30, kde=True)
# plt.xlabel("Residual (y_true - y_pred)")
# plt.ylabel("Count")
# plt.title("Training Residual Distribution")
# plt.grid(True)
# plt.show()
#
# # 测试集误差分布
# plt.figure(figsize=(8,5))
# sns.histplot(residuals_test, bins=30, kde=True)
# plt.xlabel("Residual (y_true - y_pred)")
# plt.ylabel("Count")
# plt.title("Test Residual Distribution")
# plt.grid(True)
# plt.show()

# 训练+测试误差分布图（统一颜色）
fig4, ax4 = plt.subplots(figsize=(7,6), dpi=300)

# 合并残差（train + test）
all_residuals = np.concatenate([residuals_train, residuals_test])

# 绘制直方图 + KDE
sns.histplot(all_residuals,
             bins=40,
             kde=True,
             color='gray',       # 统一颜色
             alpha=0.8,
             ax=ax4)

# 黑色边框
for spine in ax4.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.3)

# 字体设置为 Arial
plt.rcParams['font.family'] = 'Arial'

ax4.set_xlabel("Residual (Pred - True)", fontsize=14)
ax4.set_ylabel("Count", fontsize=14)
ax4.tick_params(axis='both', labelsize=12, width=1.2)
ax4.set_title("Residual Distribution (Train + Test Combined)", fontsize=15)

plt.tight_layout()

# 保存图片
save_path = os.path.join(OUTPUT_DIR, "residual_distribution_combined.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"误差分布图已保存：{save_path}")
