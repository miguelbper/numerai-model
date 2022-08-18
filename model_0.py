# ======================================================================
# imports
# ======================================================================

from numerapi import NumerAPI
from lightgbm import LGBMRegressor
import joblib
import gc
from utils import *
from pprint import pprint
import matplotlib.pyplot as plt

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

# subsamples = range(1, 13)
# cv_era = np.zeros((len(subsamples), n_splits))

# for i in tqdm_(subsamples, desc='CV number of era subsamples'):
#     for j, (trn, val) in tqdm_(enumerate(spl.split(X, y, e)), 
#                           desc='CV fold', 
#                           total=n_splits):
#         X_trn = X.iloc[trn]
#         X_val = X.iloc[val]
#         y_trn = y.iloc[trn]
#         y_val = y.iloc[val]
#         e_trn = e.iloc[trn]
#         e_val = e.iloc[val]

#         model_name = f'model-0/saved-variables/cv_era_{i}_{j}.pkl'
#         try:
#             model = joblib.load(model_name)
#         except:
#             model = EraSubsampler(LGBMRegressor(**params), n_subsamples=i)
#             model.fit(X_trn, y_trn, eras=e_trn)
#             joblib.dump(model, model_name)

#         y_name = f'model-0/saved-variables/y_era_{i}_{j}.pkl'
#         try:
#             y_val_pred = joblib.load(y_name)
#         except:
#             y_val_pred = model.predict(X_val)
#             joblib.dump(y_val_pred, y_name)

#         cv_era[i-1, j] = corr(y_val, y_val_pred, rank_b=e_val)

# cv_era = pd.DataFrame(cv_era)
# cv_era['mean'] = cv_era.mean(axis=1)
# cv_era.to_excel('model-0/cross-validation/cv_era.xlsx')


# ----------------------------------------------------------------------
# weights for training on many targets + feature neutralization
# ----------------------------------------------------------------------

# n_targ = 10
# n_feat = list(range(1, 16))
# alphas = np.arange(0, 1, 0.01)

# y = Y[Y_COLS[:n_targ]]

# cv_tar_wei = np.zeros((n_targ, n_splits))
# cv_tar_cor = np.zeros((n_splits))

# cv_neu_alp = np.zeros((len(n_feat), n_splits))
# cv_neu_cor = np.zeros((len(n_feat), n_splits))
# cv_neu_opt = np.zeros((3, n_splits))

# for j, (trn, val) in tqdm_(enumerate(spl.split(X, y, e)), 
#                            desc='CV targets', 
#                            total=n_splits):
#     X_trn = X.iloc[trn]
#     X_val = X.iloc[val]
#     y_trn = y.iloc[trn]
#     y_val = y.iloc[val]
#     e_trn = e.iloc[trn]
#     e_val = e.iloc[val]

#     model_name = f'model-0/saved-variables/cv_targets_{j}.pkl'
#     try:
#         model = joblib.load(model_name)
#     except IOError:
#         model = LGBMRegressor(**params)
#         model = EraSubsampler(model, n_subsamples=7)
#         model = MultiOutputRegressor(model)
#         model.fit(X_trn, y_trn, eras=e_trn)
#         joblib.dump(model, model_name)

#     y_val_true = y_val[Y_TRUE]
#     y_name = f'model-0/saved-variables/y_targets_{j}.pkl'
#     try:
#         y_val_pred = joblib.load(y_name)
#     except:
#         y_val_pred = model.predict(X_val)
#         joblib.dump(y_val_pred, y_name)

#     def f_corr(w):
#         return corr(y_val_true, y_val_pred @ w, rank_b=e_val)

#     w_max = maximum(f_corr, n_targ)
#     y_val_pred_w = y_val_pred @ w_max
#     cv_tar_wei[:, j] = w_max
#     cv_tar_cor[j] = f_corr(w_max)

#     # CV for feature neutralization
#     exposures = corr(X_val, y_val_pred_w, rank_a=e_val, rank_b=e_val)
#     riskiest = [(v, i) for i, v in enumerate(exposures)]
#     riskiest = sorted(riskiest, reverse=True)
#     riskiest = [i for _, i in riskiest]

#     for k, n_features in enumerate(n_feat):
#         riskiest_n = riskiest[0:n_features]
#         riskiest_n = [x_cols[i] for i in riskiest_n]

#         def aux_linreg(df):
#             X_ = df[df.columns[0:-1]]
#             y_ = df[df.columns[-1]]
#             model = LinearRegression()
#             model.fit(X_, y_)
#             return pd.Series(model.predict(X_) - model.intercept_)

#         R = X_val[riskiest_n].to_numpy()
#         df_Ry = pd.DataFrame(np.hstack((R, np.atleast_2d(y_val_pred_w).T)))
#         y_linr = df_Ry.groupby(e_val.to_numpy()).apply(aux_linreg).to_numpy()

#         cs = np.array([
#             corr(y_val_true, y_val_pred_w - a * y_linr, rank_b=e_val) 
#             for a in alphas
#         ])
#         mx = np.argmax(cs)
#         cv_neu_alp[k, j] = alphas[mx]
#         cv_neu_cor[k, j] = cs[mx]

#         fig, ax = plt.subplots()
#         ax.plot(alphas, cs)
#         ax.set_title(f'split = {j}, n_features = {n_features}')
#         fig.savefig(f'model-0/figures/corr_alpha_{j}_{n_features}.png')
#         plt.close(fig)
        
#     mx = np.argmax(cv_neu_cor, axis=0)[j]
#     cv_neu_opt[0, j] = n_feat[mx]
#     cv_neu_opt[1, j] = cv_neu_alp[mx, j]
#     cv_neu_opt[2, j] = cv_neu_cor[mx, j]


# cv_tar_wei = pd.DataFrame(cv_tar_wei)
# cv_tar_wei['mean'] = cv_tar_wei.mean(axis=1)
# cv_tar_wei.to_excel('model-0/cross-validation/cv_tar_wei.xlsx')

# cv_tar_cor = pd.DataFrame(cv_tar_cor)
# cv_tar_cor.to_excel('model-0/cross-validation/cv_tar_cor.xlsx')

# cv_neu_alp = pd.DataFrame(cv_neu_alp)
# cv_neu_alp.to_excel('model-0/cross-validation/cv_neu_alp.xlsx')

# cv_neu_cor = pd.DataFrame(cv_neu_cor)
# cv_neu_cor.to_excel('model-0/cross-validation/cv_neu_cor.xlsx')

# cv_neu_opt = pd.DataFrame(cv_neu_opt)
# cv_neu_opt.to_excel('model-0/cross-validation/cv_neu_opt.xlsx')

# ----------------------------------------------------------------------
# conclusion
# ----------------------------------------------------------------------

# era subsamples: 7

# targets: 
# 0: 0,5525      ->  0.55 
# 1: 0,1225      ->  0.12
# 2: -0,2475     -> -0.25
# 3: 1,38778E-17 ->
# 4: 0,1725      ->  0.17
# 5: 0,0375      ->
# 6: -0,03       ->
# 7: -0,05       ->
# 8: 0,3675      ->  0.37
# 9: 0,075       ->
# should retrain targets with only 0, 1, 2, 4, 8 available
# also with only 0, 1, 4, 8 (since 2 gives negative coef)

# feature neutralization:
# n_features: around 10 maybe
# alpha: around 10% - 40%


# ----------------------------------------------------------------------
# retrain / grid search
# ----------------------------------------------------------------------

# target weights
y = Y

# ran_w1 = [x / 100 for x in range(0, 25, 5)] # np.arange(0.1, 0.2, 0.01)
# ran_w2 = [x / 100 for x in range(-35, 5, 5)] # np.arange(0, -0.3, -0.05)
# ran_w4 = [x / 100 for x in range(0, 25, 5)] # np.arange(0.1, 0.2, 0.01)
# ran_w8 = [x / 100 for x in range(20, 45, 5)] # np.arange(0.3, 0.4, 0.01)
# len_ws = len(ran_w1) * len(ran_w2) * len(ran_w4) * len(ran_w8)

# corr_dict = {
#     'w0': [],
#     'w1': [],
#     'w2': [],
#     'w4': [],
#     'w8': [],
#     'c0': [],
#     'c1': [],
#     'c2': [],
#     'c3': [],
# }

# for j, (trn, val) in tqdm(enumerate(spl.split(X, y, e)), 
#                            desc='cv split', 
#                            total=n_splits,
#                            leave=None,
#                            position=0):
#     X_trn = X.iloc[trn]
#     X_val = X.iloc[val]
#     y_trn = y.iloc[trn]
#     y_val = y.iloc[val]
#     e_trn = e.iloc[trn]
#     e_val = e.iloc[val]

#     model_name = f'model-0/saved-variables/cv_targets_{j}.pkl'
#     try:
#         model = joblib.load(model_name)
#     except IOError:
#         model = LGBMRegressor(**params)
#         model = EraSubsampler(model, n_subsamples=7)
#         model = MultiOutputRegressor(model)
#         model.fit(X_trn, y_trn, eras=e_trn)
#         joblib.dump(model, model_name)

#     y_val_true = y_val[Y_TRUE]
#     y_name = f'model-0/saved-variables/y_targets_{j}.pkl'
#     try:
#         y_val_pred = joblib.load(y_name)
#     except:
#         y_val_pred = model.predict(X_val)
#         joblib.dump(y_val_pred, y_name)

#     # ran_w1 = [x / 100 for x in range(10, 20)]
#     # ran_w2 = [x / 100 for x in range(0, -30, -5)]
#     # ran_w4 = [x / 100 for x in range(10, 20)]
#     # ran_w8 = [x / 100 for x in range(30, 40)]
#     # ran_ws = product(ran_w1, ran_w2, ran_w4, ran_w8)
#     # len_ws = len(ran_w1) * len(ran_w2) * len(ran_w4) * len(ran_w8)

#     for (w1, w2, w4, w8) in tqdm(product(ran_w1, ran_w2, ran_w4, ran_w8), 
#                                  desc='weights', 
#                                  total=len_ws,
#                                  leave=False,
#                                  position=1):
#         w0 = 1 - w1 - w2 - w4 - w8
#         w = np.array([w0, w1, w2, 0, w4, 0, 0, 0, w8, 0])
#         y_val_pred_w = y_val_pred @ w
        
#         cr = corr(y_val_true, y_val_pred_w, rank_b=e_val)

#         corr_dict[f'c{j}'].append(cr)
#         if j == 0:
#             corr_dict['w0'].append(w0)
#             corr_dict['w1'].append(w1)
#             corr_dict['w2'].append(w2)
#             corr_dict['w4'].append(w4)
#             corr_dict['w8'].append(w8)

# corr_df = pd.DataFrame(corr_dict)
# corr_df['ca'] = corr_df[['c0', 'c1', 'c2', 'c3']].mean(axis=1)
# corr_df = corr_df.sort_values('ca', ascending=False)
# corr_df.to_excel('model-0/cross-validation/cv_era_grd.xlsx')

# feature neutralization

ran_nf = np.arange(5, 55, 5)
ran_al = np.arange(0, 1, 0.05)

corr_dict = {
    'nf': [],
    'al': [],
    'c0': [],
    'c1': [],
    'c2': [],
    'c3': [],
}

for j, (trn, val) in tqdm(enumerate(spl.split(X, y, e)), 
                           desc='cv split', 
                           total=n_splits,
                           leave=None,
                           position=0):
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

    
    w = np.array([0.6, 0.1, -0.25, 0, 0.2, 0, 0, 0, 0.35, 0])
    y_val_pred_w = y_val_pred @ w
    
    exposures = corr(X_val, y_val_pred_w, rank_a=e_val, rank_b=e_val)
    riskiest = [(v, i) for i, v in enumerate(exposures)]
    riskiest = sorted(riskiest, reverse=True)
    riskiest = [i for _, i in riskiest]

    for nf in tqdm(ran_nf, desc='n_features', leave=False, position=1):
        riskiest_n = riskiest[0:nf]
        riskiest_n = [x_cols[i] for i in riskiest_n]

        def aux_linreg(df):
            X_ = df[df.columns[0:-1]]
            y_ = df[df.columns[-1]]
            model = LinearRegression()
            model.fit(X_, y_)
            return pd.Series(model.predict(X_) - model.intercept_)

        R = X_val[riskiest_n].to_numpy()
        df_Ry = pd.DataFrame(np.hstack((R, np.atleast_2d(y_val_pred_w).T)))
        y_linr = df_Ry.groupby(e_val.to_numpy()).apply(aux_linreg).to_numpy()

        for al in tqdm(ran_al, desc='alpha', leave=False, position=2):
            y_neut = y_val_pred_w - al * y_linr
            cr = corr(y_val_true, y_neut, rank_b=e_val)

            corr_dict[f'c{j}'].append(cr)
            if j == 0:
                corr_dict['nf'].append(nf)
                corr_dict['al'].append(al)
            

corr_df = pd.DataFrame(corr_dict)
corr_df['ca'] = corr_df[['c0', 'c1', 'c2', 'c3']].mean(axis=1)
corr_df = corr_df.sort_values('ca', ascending=False)
corr_df.to_excel('model-0/cross-validation/cv_neu_grd.xlsx')


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