from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor, 
    RandomForestRegressor,
)

from time import time
import os
import sys
sys.path.append(os.path.abspath('.'))
from utils import *


# ======================================================================
# Define models
# ======================================================================

# xgboost
params_0 = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 5,
    'max_leaves': 2**5,
    'colsample_bytree': 0.1,
    'gpu_id': 0,
    'tree_method': 'gpu_hist'
}
model_0 = XGBRegressor(**params_0)

# lightgbm
params_1 = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 5,
    'num_leaves': 2**5,
    'colsample_bytree': 0.1,
    'device': 'gpu',
}
model_1 = LGBMRegressor(**params_1)

# catboost
params_2 = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 5,
    'num_leaves': 2**5,
    # 'task_type': 'GPU',
    # 'devices': '0',
    'colsample_bylevel': 0.1,
    'logging_level': 'Silent',
}
model_2 = CatBoostRegressor(**params_2)

# extratrees
params_3 = {
    'n_estimators': 2000,
    'max_depth': 5,
    'max_leaf_nodes': 2**5,
}
model_3 = ExtraTreesRegressor(**params_3)

# randomforest
params_4 = {
    'n_estimators': 2000,
    'max_depth': 5,
    'max_leaf_nodes': 2**5,
}
model_4 = RandomForestRegressor(**params_4)

# list
models = [
    # model_0,
    # model_1,
    model_2,
    # model_3,
    # model_4,
]


# ======================================================================
# train all models
# ======================================================================

results = {
    'algorithm': [],
    'time_to_train': [],
    'score_insample': [],
    'score_outofsample': [],
}

x_cols = FEAT_S
y_cols = [Y_TRUE]

X_trn, y_trn, e_trn = read_Xye(
    'train', 
    x_cols=x_cols, 
    y_cols=y_cols, 
    eras=np.arange(1, 21),
)

X_val, y_val, e_val = read_Xye(
    'train', 
    x_cols=x_cols, 
    y_cols=y_cols, 
    eras=np.arange(201, 251),
)

for model in models:
    model_name = str(type(model))[8:][:-2].split('.')[-1]
    model_path = f'./cv/cv-0/{model_name}.pkl'
    
    # train model in 1/4 of train
    t0 = time()
    model.fit(X_trn, y_trn)
    t1 = time() - t0
    joblib.dump(model, model_path)

    # predict in 1/4 of train
    y_pred_trn = model.predict(X_trn) 
    corr_insample = corr(y_trn, y_pred_trn, rank_b=e_trn)

    # predict in validation
    y_pred_val = model.predict(X_val) 
    corr_oosample = corr(y_val, y_pred_val, rank_b=e_val)

    # save results
    results['algorithm'].append(model_name)
    results['time_to_train'].append(t1)
    results['score_insample'].append(corr_insample)
    results['score_outofsample'].append(corr_oosample)

results = pd.DataFrame(results)
results.to_excel('./cv/cv-0/cv_0.xlsx')