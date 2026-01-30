import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

start_time = time.time()


data = pd.read_csv('11.csv')


x = data.iloc[:, :4].values
y = data.iloc[:, 5].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
opt_score = []

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

def optuna_objective(trial):

    learning_rate = trial.suggest_float('learning_rate', 0.01, 2, log=True)
    max_depth = trial.suggest_int('max_depth', 10, 100)
    max_features = trial.suggest_int('max_features', 10, 100)
    min_child_samples = trial.suggest_int('min_child_samples', 10, 100)
    n_estimators = trial.suggest_int('n_estimators', 10, 1000)
    reg_alpha = trial.suggest_int('reg_alpha', 10, 1000)
    reg_lambda = trial.suggest_int('reg_lambda', 10, 1000)


    estimator = LGBMRegressor(learning_rate=learning_rate,
                              max_depth=max_depth,
                              max_features=max_features,
                              min_child_samples=min_child_samples,
                              n_estimators=n_estimators,
                              reg_alpha=reg_alpha,
                              reg_lambda=reg_lambda,
                              random_state=42
                              )


    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    validation_loss = cross_validate(estimator, x_train, y_train, scoring='neg_root_mean_squared_error',
                                     cv=cv, verbose=False,
                                     error_score='raise'
                                     )
    opt_score.append(np.mean(validation_loss['test_score']))
    return np.mean(validation_loss['test_score'])


def opt_optuna(n_trial):
    opt = optuna.create_study(sampler=optuna.samplers.TPESampler(),
                               direction='maximize')
    opt.optimize(optuna_objective, n_trials=n_trial, show_progress_bar=True)
    print('\n', 'best params:', opt.best_trial.params,
          '\n', 'best cvscore:', opt.best_trial.values)
    return opt.best_trial.params, opt.best_trial.values

optuna.logging.set_verbosity(optuna.logging.ERROR)
best_params, best_score = opt_optuna(1000)
end_time =time.time()
execution_time = end_time - start_time
print('所用时间：{:2f}秒'.format(execution_time))
