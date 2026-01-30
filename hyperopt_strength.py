import numpy as np
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import KFold, cross_validate, train_test_split
from lightgbm import LGBMRegressor
import time
import pandas as pd

start_time = time.time()


data = pd.read_csv('11.csv')


x = data.iloc[:, :4].values
y = data.iloc[:, 4].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
opt_score = []

def hyperopt_objective(params):
    # 解包超参数
    learning_rate = params['learning_rate']
    max_depth = int(params['max_depth'])
    max_features = int(params['max_features'])
    min_child_samples = int(params['min_child_samples'])
    n_estimators = int(params['n_estimators'])
    reg_alpha = params['reg_alpha']
    reg_lambda = params['reg_lambda']
    boosting_type = params['boosting_type']


    estimator = LGBMRegressor(learning_rate=learning_rate,
                               max_depth=max_depth,
                               max_features=max_features,
                               min_child_samples=min_child_samples,
                               n_estimators=n_estimators,
                               reg_alpha=reg_alpha,
                               reg_lambda=reg_lambda,
                               boosting_type=boosting_type,
                               random_state=42)


    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    validation_loss = cross_validate(estimator, x_train, y_train, scoring='neg_root_mean_squared_error',
                                     cv=cv, verbose=False,
                                     error_score='raise'  # 如果交叉验证中的算法执行报错，告诉我们错误的理由
                                     )
    opt_score.append(np.mean(-validation_loss['test_score']))
    return np.mean(-validation_loss['test_score'])


space = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 2),
    'max_depth': hp.quniform('max_depth', 10, 100, 1),
    'max_features': hp.quniform('max_features',10, 100, 1),
    'min_child_samples': hp.quniform('min_child_samples', 10, 100, 1),
    'n_estimators': hp.quniform('n_estimators', 10, 1000, 10),
    'reg_alpha': hp.quniform('reg_alpha', 10, 1000, 10),
    'reg_lambda': hp.quniform('reg_lambda', 10, 1000, 10),
    'boosting_type': hp.choice('boosting_type',['gbdt', 'goss']),

        }


trials = Trials()
rstate = np.random.default_rng(42)
best = fmin(fn=hyperopt_objective,
            space=space,
            algo=tpe.suggest,
            max_evals=1000,
            trials=trials,
            rstate=rstate)


print("Best hyperparameters:")
best_trial = trials.best_trial
for key, value in best_trial['misc']['vals'].items():
    print(f"{key}: {value[0]}")
print(f"Best cross-validation score (root_mean_squared_error): {best_trial['result']['loss']}")

end_time =time.time()
execution_time = end_time - start_time
print('所用时间：{:2f}秒'.format(execution_time))