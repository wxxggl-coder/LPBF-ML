import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sko.GA import GA
import matplotlib.pyplot as plt


OUTPUT_DIR = "GA_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


data = pd.read_csv('11.csv')
x = data.iloc[:, :4].values
y = data.iloc[:, 4].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
 
model = LGBMRegressor(
    learning_rate=0.948,
    max_depth=61,
    max_features=37,
    min_child_samples=11,
    n_estimators=708,
    reg_alpha=7,
    reg_lambda=533
)
model.fit(x_train_scaled, y_train)


def make_objective(model, scaler):
    def objective(params):
        X = params.reshape(1, -1)
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        return -y_pred[0]
    return objective



num_runs = 100
results = []

for i in range(num_runs):
    np.random.seed(i)
    ga = GA(
        func=make_objective(model, scaler),
        n_dim=4,
        size_pop=50,
        max_iter=100,
        prob_mut=0.005,
        lb=[0.01, 0.01, 200, 800],
        ub=[5, 3, 400, 1700],
        precision=1e-7
    )


    best_x, best_y = ga.run()
    results.append(list(best_x) + [-best_y[0]])
    print(f"Run {i+1}: Best_Y={-best_y}, Params={best_x}")



columns = ['Ti', 'B', 'Power', 'Speed', 'Best_Y']
df = pd.DataFrame(results, columns=columns)


stats_df = df.describe().T[['mean', 'std', 'min', 'max']]
stats_df.index.name = 'Parameter'

excel_path = os.path.join(OUTPUT_DIR, "GA_multiple_runs_results.xlsx")

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='GA_multiple_runs_results', index=False)
    stats_df.to_excel(writer, sheet_name='Statistics')

print(f"\n所有 GA 运行结果与统计数据已保存到：{excel_path}")


def journal_style(ax):
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)



plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18
fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=300)

# ---- (1) Ti ----
ti_mean = df['Ti'].mean()
ti_std = df['Ti'].std()
axs[0,0].bar(0, ti_mean, yerr=ti_std, capsize=3, color='#6baed6', edgecolor='black', alpha=0.7)
axs[0,0].set_xticks([0])
axs[0,0].set_xticklabels(['Ti'])
axs[0,0].set_ylabel('Value')
axs[0,0].set_title('Ti (Mean ± STD)')
axs[0,0].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

# ---- (2) B ----
b_mean = df['B'].mean()
b_std = df['B'].std()
axs[0,1].bar(0, b_mean, yerr=b_std, capsize=3, color='#fd8d3c', edgecolor='black', alpha=0.7)
axs[0,1].set_xticks([0])
axs[0,1].set_xticklabels(['B'])
axs[0,1].set_ylabel('Value')
axs[0,1].set_title('B (Mean ± STD)')
axs[0,1].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

# ---- (3) Power ----
power_mean = df['Power'].mean()
power_std = df['Power'].std()
axs[1,0].bar(0, power_mean, yerr=power_std, capsize=3, color='#74c476', edgecolor='black', alpha=0.7)
axs[1,0].set_xticks([0])
axs[1,0].set_xticklabels(['Power'])
axs[1,0].set_ylabel('Value')
axs[1,0].set_title('Power (Mean ± STD)')
axs[1,0].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

# ---- (4) Speed ----
speed_mean = df['Speed'].mean()
speed_std = df['Speed'].std()
axs[1,1].bar(0, speed_mean, yerr=speed_std, capsize=3, color='#9e9ac8', edgecolor='black', alpha=0.7)
axs[1,1].set_xticks([0])
axs[1,1].set_xticklabels(['Speed'])
axs[1,1].set_ylabel('Value')
axs[1,1].set_title('Speed (Mean ± STD)')
axs[1,1].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "GA_params_2x2_bar.png"), dpi=600, bbox_inches='tight')
plt.close()


best_y_mean = df['Best_Y'].mean()
best_y_std = df['Best_Y'].std()
plt.figure(figsize=(6,5), dpi=300)
plt.hist(df['Best_Y'], bins=10, color='lightgray', edgecolor='black', alpha=0.7)
plt.axvline(best_y_mean, color='red', linestyle='--', label=f'Mean={best_y_mean:.2f}')
plt.fill_betweenx(y=[0, plt.gca().get_ylim()[1]], x1=best_y_mean-best_y_std, x2=best_y_mean+best_y_std,
                  color='red', alpha=0.2, label=f'±1 STD={best_y_std:.2f}')
plt.xlabel('Predicted Strength')
plt.ylabel('Frequency')
plt.title('Distribution of Best_Y (Predicted Strength)')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Best_Y_distribution_nature.png"), dpi=600, bbox_inches='tight')
plt.close()

print(f"\n所有 Nature 风格图像已保存到文件夹：{OUTPUT_DIR}/")

