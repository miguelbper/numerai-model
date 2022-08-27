from utils import *
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from utils import objective_corr
import numpy as np


X, y = read_Xy('train', FEAT_S, [Y_TRUE], eras=np.arange(1, 51))


def objective_corr(y_true, y_pred):
    m = len(y_true)

    var_e = (m - 1) / m**2
    std_t = np.std(y_true)
    std_p = np.std(y_pred)

    cov_tp = np.cov(y_true, y_pred, ddof=0)[0, 1]
    cov_ep = (y_pred - np.mean(y_pred)) / m
    cov_et = (y_true - np.mean(y_true)) / m

    grad = - (- cov_tp * cov_ep / std_p**3 + cov_et / std_p) / std_t
    hess = - (- 2 * cov_ep * cov_et
              + 3 * cov_ep**2 * cov_tp / std_p**2
              - cov_tp * var_e) / (std_t * std_p**3)

    return grad, hess


def objective_corr_ones(y_true, y_pred):
    m = len(y_true)

    var_e = (m - 1) / m**2
    std_t = np.std(y_true)
    std_p = np.std(y_pred)

    cov_tp = np.cov(y_true, y_pred, ddof=0)[0, 1]
    cov_ep = (y_pred - np.mean(y_pred)) / m
    cov_et = (y_true - np.mean(y_true)) / m

    grad = - (- cov_tp * cov_ep / std_p**3 + cov_et / std_p) / std_t
    hess = np.ones(m)

    return grad, hess


def corr_(a, b):
    return np.corrcoef(a, b)[0, 1]


# # ======================================================================
# # experiment with XGBoost
# # ======================================================================

# # ----------------------------------------------------------------------
# # mean squared error
# # ----------------------------------------------------------------------

# xgb_params = {
#     'n_estimators': 200,
#     'learning_rate': 0.01,
#     'max_depth': 5,
#     'max_leaves': 2**5,
#     'colsample_bytree': 0.1,
#     'gpu_id': 0,
#     'tree_method': 'gpu_hist',
# }

# print('\nTraining mse model... ', end='')
# model_mse = XGBRegressor(**xgb_params)
# model_mse.fit(X, y)
# c_mse = corr(y, model_mse.predict(X))
# print('Done')


# # ----------------------------------------------------------------------
# # initial model
# # ----------------------------------------------------------------------

# xgb_params_init = dict(xgb_params)
# xgb_params_init['n_estimators'] = 1

# print('Training init model... ', end='')
# model_init = XGBRegressor(**xgb_params_init)
# model_init.fit(X, y)
# base_margin = model_init.predict(X)
# print('Done')


# # ----------------------------------------------------------------------
# # corr (hessian = ones)
# # ----------------------------------------------------------------------

# xgb_params_obj = dict(xgb_params)
# xgb_params_obj['objective'] = objective_corr_ones
# xgb_params_obj['eval_metric'] = corr_

# print('Training main model (hessian = ones)... ', end='')
# model_obj = XGBRegressor(**xgb_params_obj)
# model_obj.fit(X, y, base_margin=base_margin, eval_set=[(X,y)])
# print('Done')
# c_obj = corr(y, model_obj.predict(X))


# # ----------------------------------------------------------------------
# # corr (with hessian, big learning rate)
# # ----------------------------------------------------------------------

# xgb_params_hes = dict(xgb_params)
# xgb_params_hes['objective'] = objective_corr
# xgb_params_hes['eval_metric'] = corr_
# xgb_params_hes['learning_rate'] = 0.5

# print('Training main model... (with hessian)', end='')
# model_hes = XGBRegressor(**xgb_params_hes)
# model_hes.fit(X, y, base_margin=base_margin, eval_set=[(X,y)])
# print('Done')
# c_hes = corr(y, model_hes.predict(X))


# # ----------------------------------------------------------------------
# # print results
# # ----------------------------------------------------------------------

# print('\n')
# print(f'c_mse = {c_mse}')
# print(f'c_obj = {c_obj}')
# print(f'c_hes = {c_hes}')


# ======================================================================
# experiment with LightGBM
# ======================================================================

lgbm_params = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 5,
    'num_leaves': 2**5,
    'colsample_bytree': 0.1,
    'device': 'gpu',
    'objective': objective_corr,
}

param_init = {
    'n_estimators': 1,
    'learning_rate': 0.01,
    'max_depth': 5,
    'num_leaves': 2**5,
    'colsample_bytree': 0.1,
    'device': 'gpu',
}
init_model = LGBMRegressor(**param_init)
init_model.fit(X, y)

model = LGBMRegressor(**lgbm_params)
model.fit(X, y, init_model=init_model)