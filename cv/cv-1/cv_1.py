from catboost import CatBoostRegressor
from time import time
import os
import sys
sys.path.append(os.path.abspath('.'))
from utils import *

params_0 = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 5,
    'num_leaves': 2**5,
    # 'logging_level': 'Silent',
}

params_gpu = {'task_type': 'GPU', 'devices': '0'}
params_gpu.update(**params_0)

params_col = {'colsample_bylevel': 0.1}
params_col.update(**params_0)

results = {
    'algorithm': [],
    'time_to_train': [],
    'score_insample': [],
    'score_outofsample': [],
}

x_cols = X_COLS
y_cols = [Y_TRUE]

X_trn, y_trn, e_trn = read_Xye(
    'train', 
    x_cols=x_cols, 
    y_cols=y_cols, 
    eras=np.arange(1, 575),
)

X_val, y_val, e_val = read_Xye(
    'validation', 
    x_cols=x_cols, 
    y_cols=y_cols, 
    eras=np.arange(575, 575 + 100),
)


# ----------------------------------------------------------------------
# gpu
# ----------------------------------------------------------------------

model = CatBoostRegressor(**params_gpu)
model_name = 'gpu'
model_path = f'./cv/cv-1/{model_name}.pkl'
    
# train model in train
t0 = time()
model.fit(X_trn, y_trn)
t1 = time() - t0
joblib.dump(model, model_path)

# predict in train
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


# ----------------------------------------------------------------------
# col
# ----------------------------------------------------------------------

model = CatBoostRegressor(**params_col)
model_name = 'col'
model_path = f'./cv/cv-1/{model_name}.pkl'
    
# train model in train
t0 = time()
model.fit(X_trn, y_trn)
t1 = time() - t0
joblib.dump(model, model_path)

# predict in train
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
results.to_excel('./cv/cv-1/cv_1.xlsx')