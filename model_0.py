# ======================================================================
# imports
# ======================================================================

from numerapi import NumerAPI
from lightgbm import LGBMRegressor
import joblib
import gc
from utils import *

# ======================================================================
# download data
# ======================================================================

napi = NumerAPI()
round = napi.get_current_round()
era = round + 695
napi.download_dataset('v4/features.json', 'data/features.json')
napi.download_dataset('v4/train_int8.parquet', 'data/train.parquet')
napi.download_dataset('v4/validation_int8.parquet', 'data/validation.parquet')
napi.download_dataset('v4/live_int8.parquet', f'data/live_{round}.parquet')


# ======================================================================
# cross validation
# ======================================================================

params = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 5,
    'num_leaves': 2**5,
    'colsample_bytree': 0.1,
    'device': 'gpu',
}

x_cols = FEAT_L
eras = None 
n_splits = 4

df = read_data('train', x_cols, eras=eras)
X = df[x_cols]
Y = df[Y_COLS]
y = df[Y_TRUE]
e = df[ERA]
del df
gc.collect()

spl = TimeSeriesSplitGroups(n_splits=n_splits)


# ----------------------------------------------------------------------
# number of era subsamples
# ----------------------------------------------------------------------

subsamples = range(1, 13)
cv_era = np.zeros((len(subsamples), n_splits))

for i in tqdm_(subsamples, desc='CV number of era subsamples'):
    for j, (trn, val) in tqdm_(enumerate(spl.split(X, y, e)), 
                          desc='CV fold', 
                          total=n_splits):
        X_trn = X.iloc[trn]
        X_val = X.iloc[val]
        y_trn = y.iloc[trn]
        y_val = y.iloc[val]
        e_trn = e.iloc[trn]
        e_val = e.iloc[val]

        model_name = f'model-0/saved-variables/cv_era_{i}_{j}.pkl'
        try:
            model = joblib.load(model_name)
        except:
            model = EraSubsampler(LGBMRegressor(**params), n_subsamples=i)
            model.fit(X_trn, y_trn, eras=e_trn)
            joblib.dump(model, model_name)

        y_name = f'model-0/saved-variables/y_era_{i}_{j}.pkl'
        try:
            y_val_pred = joblib.load(y_name)
        except:
            y_val_pred = model.predict(X_val)
            joblib.dump(y_val_pred, y_name)

        cv_era[i-1, j] = corr(y_val, y_val_pred, rank_b=e_val)

cv_era = pd.DataFrame(cv_era)
cv_era['mean'] = cv_era.mean(axis=1)
cv_era.to_excel('model-0/cross-validation/cv_era.xlsx')


# ----------------------------------------------------------------------
# weights for training on many targets + feature neutralization
# ----------------------------------------------------------------------

exit()
n_targ = 2 # 10
n_feat = [5, 10, 25, 50, 100]
alphas = np.arange(0, 1, 0.01)

y = Y[Y_COLS[:n_targ]]

cv_tar_wei = np.zeros((n_targ, n_splits))
cv_tar_cor = np.zeros((n_splits))

cv_neu_alp = np.zeros((len(n_feat), n_splits))
cv_neu_cor = np.zeros((len(n_feat), n_splits))
cv_neu_opt = np.zeros((3, n_splits))

for j, (trn, val) in tqdm_(enumerate(spl.split(X, y, e)), 
                           desc='CV targets', 
                           total=n_splits):
    X_trn = X.iloc[trn]
    X_val = X.iloc[val]
    y_trn = y.iloc[trn]
    y_val = y.iloc[val]
    e_trn = e.iloc[trn]
    e_val = e.iloc[val]

    model_name = f'model-0/saved-variables/cv_targets_{j}.pkl'
    try:
        model = joblib.load(model_name)
    except IOError:
        model = LGBMRegressor(**params)
        model = EraSubsampler(model, n_subsamples=7)
        model = MultiOutputRegressor(model)
        model.fit(X_trn, y_trn, eras=e_trn)
        joblib.dump(model, model_name)

    y_val_true = y_val[Y_TRUE]
    y_name = f'model-0/saved-variables/y_targets_{j}.pkl'
    try:
        y_val_pred = joblib.load(y_name)
    except:
        y_val_pred = model.predict(X_val)
        joblib.dump(y_val_pred, y_name)

    def f_corr(w):
        return corr(y_val_true, y_val_pred @ w, rank_b=e_val)

    w_max = maximum(f_corr, n_targ)
    y_val_pred_w = y_val_pred @ w_max
    cv_tar_wei[:, j] = w_max
    cv_tar_cor[j] = f_corr(w_max)

    # CV for feature neutralization
    exposures = corr(X_val, y_val_pred_w, rank_a=e_val, rank_b=e_val)
    riskiest = [(v, i) for i, v in enumerate(exposures)]
    riskiest = sorted(riskiest, reverse=True)
    riskiest = [i for _, i in riskiest]

    for k, n_features in enumerate(n_feat):
        riskiest_n = riskiest[0:n_features]

        def aux_linreg(df):
            X_ = df[df.columns[0:-1]]
            y_ = df[df.columns[-1]]
            model = LinearRegression()
            model.fit(X_, y_)
            return pd.Series(model.predict(X_) - model.intercept_)

        R = X_val[:, riskiest]
        df_Ry = pd.DataFrame(np.hstack((R, np.atleast_2d(y_val_pred_w).T)))
        y_linr = df_Ry.groupby(e_val).apply(aux_linreg).to_numpy()

        cs = np.array([
            corr(y_val_true, y_val_pred_w - a * y_linr, rank_b=e_val) 
            for a in alphas
        ])
        mx = np.argmax(cs)
        cv_neu_alp[k, j] = alphas[mx]
        cv_neu_cor[k, j] = cs[mx]
        
    mx = np.argmax(cv_neu_cor, axis=0)
    cv_neu_opt[0, j] = n_feat[mx]
    cv_neu_opt[1, j] = cv_neu_alp[mx, j]
    cv_neu_opt[2, j] = cv_neu_cor[mx, j]


cv_tar_wei = pd.DataFrame(cv_tar_wei)
cv_tar_wei['mean'] = cv_tar_wei.mean(axis=1)
cv_tar_wei.to_excel('model-0/cross-validation/cv_tar_wei.xlsx')

cv_tar_cor = pd.DataFrame(cv_tar_cor)
cv_tar_cor.to_excel('model-0/cross-validation/cv_tar_cor.xlsx')

cv_neu_alp = pd.DataFrame(cv_neu_alp)
cv_neu_alp.to_excel('model-0/cross-validation/cv_neu_alp.xlsx')

cv_neu_cor = pd.DataFrame(cv_neu_cor)
cv_neu_cor.to_excel('model-0/cross-validation/cv_neu_cor.xlsx')

cv_neu_opt = pd.DataFrame(cv_neu_opt)
cv_neu_opt.to_excel('model-0/cross-validation/cv_neu_opt.xlsx')

# ======================================================================
# final model
# ======================================================================

# model = LGBMRegressor(**params)
# model = EraSubsampler(model, n_subsamples=7)

# ----------------------------------------------------------------------
# train
# ----------------------------------------------------------------------

# df_trn = read_data('train')
# model.fit(df_trn[X_COLS], df_trn[Y_TRUE], eras=df_trn[ERA])
# joblib.dump(model, f'model-0/saved-variables/lgbm_{now_dt()}.pkl')


# ----------------------------------------------------------------------
# predict
# ----------------------------------------------------------------------

# df_liv = read_data('live')
# df_liv[Y_PRED] = model.predict(df_liv[X_COLS])
# df_liv[Y_RANK] = df_liv[Y_PRED].rank(pct=True)
# df_liv[Y_RANK].to_csv(f'model-0/predictions/lgbm_live_predictions_{round}_{now_dt()}.csv')