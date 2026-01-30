import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import openpyxl



data = pd.read_csv('11.csv')


X = data.iloc[:, :4].values
y = data.iloc[:, 5].values


model_r2 = {}
model_a = {}
model_sqrt = {}


for i in range(X.shape[1]):
    for j in range(i + 1, X.shape[1]):
        for k in range(j + 1, X.shape[1]):

            feature_subset = [i, j, k]

            model = LGBMRegressor()
            n_runs = 100
            all_prediction_y = []
            for n in range(n_runs):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model.fit(X_train[:, feature_subset], y_train)

                y_pred = model.predict(X_test[:, feature_subset])

                all_prediction_y.append(y_pred)
            average_prediction_y = np.mean(all_prediction_y, axis=0)

            accuracy = r2_score(y_test, average_prediction_y)
            sqrterror1 = math.sqrt(mean_squared_error(y_test, average_prediction_y))
            S_accuracy = 1 - np.mean(np.abs(average_prediction_y - y_test) / y_test)

            model_r2[tuple(feature_subset)] = accuracy
            model_a[tuple(feature_subset)] = S_accuracy
            model_sqrt[tuple(feature_subset)] = sqrterror1



best_features = max(model_r2, key=model_r2.get)
best_r2 = model_r2[best_features]

print(f"Best features: {best_features}")
print(f"Best R2 score: {best_r2}")
print(model_r2)
print(model_a)
print(model_sqrt)
