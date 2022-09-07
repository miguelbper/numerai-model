from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
import os
import sys
sys.path.append(os.path.abspath('.'))
from utils import *

# options for CV
x_cols = FEAT_L
y_cols = Y_FULL
eras = None # np.arange(1, 21)
n_splits = 4

# dataset and CV splitter
X, y, e = read_Xye('train', x_cols, y_cols, eras)
spl = TimeSeriesSplitGroups(n_splits=n_splits)

# model
params = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 7,
    'num_leaves': 5 * 2**(7 - 3),
    'colsample_bytree': 0.1,
    'device': 'gpu',
}
model = LGBMRegressor(**params)
model = MultiOutputRegressor(model)

# results
results = {'algo': ['nomi', '10', '20', 'X+20', '0.5']}
for j in range(n_splits):
    results[f'ins_{j}'] = []
    results[f'oos_{j}'] = []

for j, (trn, val) in enumerate(spl.split(X, y, e)):
    print(f'fold {j}/{n_splits - 1}')
    # define train and val sets
    X_trn = X.iloc[trn]
    X_val = X.iloc[val]
    y_trn = y.iloc[trn]
    y_val = y.iloc[val]
    e_trn = e.iloc[trn] #; print(e_trn)
    e_val = e.iloc[val]

    y_true_trn = y_trn[Y_TRUE] #; print(y_true_trn)
    y_true_val = y_val[Y_TRUE]

    print('\ttrain multioutput')
    # train model in train
    model.fit(X_trn, y_trn)
    joblib.dump(model, './cv/cv-4/model.pkl')

    # predict in train
    y_pred_trn = model.predict(X_trn)

    # predict in val
    y_pred_val = model.predict(X_val)

    # ------------------------------------------------------------------

    print('\tnomi')
    # nomi
    y_pred_trn_vec = y_pred_trn[:, 0] #; print(y_pred_trn_vec)
    y_pred_val_vec = y_pred_val[:, 0]
    corr_insample = corr(y_true_trn, y_pred_trn_vec, rank_b=e_trn)
    corr_oosample = corr(y_true_val, y_pred_val_vec, rank_b=e_val)
    results[f'ins_{j}'].append(corr_insample)
    results[f'oos_{j}'].append(corr_oosample)

    print('\t10')
    # 10
    X_reg_trn = y_pred_trn[:, np.arange(0, 20, 2)]
    X_reg_val = y_pred_val[:, np.arange(0, 20, 2)]

    linreg = LinearRegression()
    linreg.fit(X_reg_trn, y_true_trn)

    y_pred_trn_vec = linreg.predict(X_reg_trn)
    y_pred_val_vec = linreg.predict(X_reg_val)
    corr_insample = corr(y_true_trn, y_pred_trn_vec, rank_b=e_trn)
    corr_oosample = corr(y_true_val, y_pred_val_vec, rank_b=e_val)
    results[f'ins_{j}'].append(corr_insample)
    results[f'oos_{j}'].append(corr_oosample)

    print('\t20')
    # 20
    X_reg_trn = y_pred_trn
    X_reg_val = y_pred_val

    linreg = LinearRegression()
    linreg.fit(X_reg_trn, y_true_trn)

    y_pred_trn_vec = linreg.predict(X_reg_trn)
    y_pred_val_vec = linreg.predict(X_reg_val)
    corr_insample = corr(y_true_trn, y_pred_trn_vec, rank_b=e_trn)
    corr_oosample = corr(y_true_val, y_pred_val_vec, rank_b=e_val)
    results[f'ins_{j}'].append(corr_insample)
    results[f'oos_{j}'].append(corr_oosample)

    print('\tX + 20')
    # X + 20
    X_reg_trn = np.concatenate((X_trn.to_numpy(), y_pred_trn), axis=1) #; print(X_reg_trn)
    X_reg_val = np.concatenate((X_val.to_numpy(), y_pred_val), axis=1) #; print(X_reg_val)

    linreg = LinearRegression()
    linreg.fit(X_reg_trn, y_true_trn)

    y_pred_trn_vec = linreg.predict(X_reg_trn)
    y_pred_val_vec = linreg.predict(X_reg_val)
    corr_insample = corr(y_true_trn, y_pred_trn_vec, rank_b=e_trn)
    corr_oosample = corr(y_true_val, y_pred_val_vec, rank_b=e_val)
    results[f'ins_{j}'].append(corr_insample)
    results[f'oos_{j}'].append(corr_oosample)

    print('\t0.5')
    # my vals
    w = [0.5, 0.1, -0.1, 0, 0.15, 0, 0, 0, 0.35, 0]

    y_pred_trn_vec = y_pred_trn[:, np.arange(0, 20, 2)] @ w
    y_pred_val_vec = y_pred_val[:, np.arange(0, 20, 2)] @ w
    corr_insample = corr(y_true_trn, y_pred_trn_vec, rank_b=e_trn)
    corr_oosample = corr(y_true_val, y_pred_val_vec, rank_b=e_val)
    results[f'ins_{j}'].append(corr_insample)
    results[f'oos_{j}'].append(corr_oosample)

res = pd.DataFrame(results)
res['ins_mean'] = res[[f'ins_{j}' for j in range(n_splits)]].mean(axis=1)
res['oos_mean'] = res[[f'oos_{j}' for j in range(n_splits)]].mean(axis=1)
res.to_excel('./cv/cv-4/res.xlsx')