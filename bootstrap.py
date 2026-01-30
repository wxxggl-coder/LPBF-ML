
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


df = pd.read_csv(DATA_PATH)

X_df = df[FEATURE_COLS].copy()
y_ser = df[TARGET_COL].copy()

X = X_df.reset_index(drop=True)
y = y_ser.reset_index(drop=True)


X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_test_scaled = scaler.transform(X_test_df)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_df.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_df.columns)



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

#  绘图

plt.rcParams['font.family'] = 'Arial'
plt.style.use('seaborn-v0_8-whitegrid')


fig1, ax1 = plt.subplots(figsize=(7, 6), dpi=300)

min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())


ax1.scatter(
    y_train,
    y_pred_train_mean,
    color='#1f77b4',
    s=40,
    alpha=0.8,
    label='Train'
)


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

ax1.set_xlabel("True Value", fontsize=18)
ax1.set_ylabel("Predicted Value", fontsize=18)
ax1.tick_params(axis='both', labelsize=18, width=1.2)
ax1.legend(frameon=False, fontsize=18)
ax1.set_title("Prediction (Train vs Test )", fontsize=18)

plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "prediction_scatter_train_test_with_errorbar.png")
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Fig1（含误差棒）已保存：{fig1_path}")
