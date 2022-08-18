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

ran_ns = range(1, 13)

cv_era = {'n_subsamples': []}
for j in range(n_splits):
    cv_era[f'split_{j + 1}'] = []

for i in tqdm(ran_ns, desc='n_subsamples', leave=None, position=0):
    cv_era['n_subsamples'].append(i)

    for j, (trn, val) in tqdm(enumerate(spl.split(X, y, e)), 
                                        desc='CV fold',
                                        leave=None,
                                        position=1,
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

        c = corr(y_val, y_val_pred, rank_b=e_val)
        cv_era[f'split_{j + 1}'].append(c)

cv_era = pd.DataFrame(cv_era)
cv_era['mean'] = cv_era[[f'split_{j+1}' for j in range(n_splits)]].mean(axis=1)
cv_era.to_excel('model-0/cross-validation/cv_era.xlsx')
n_subsamples = cv_era['n_subsamples'][cv_era['mean'].argmax()] 
# n_subsamples = 7


# ----------------------------------------------------------------------
# weights for training on many targets
# ----------------------------------------------------------------------

# choose the best weights for each split and take the average (using a 
# discrete gradient descent algorithm):
# targets: 
# 0:  0,5525 
# 1:  0,1225
# 2: -0,2475
# 3:  0,0000
# 4:  0,1725
# 5:  0,0375
# 6: -0,0300
# 7: -0,0500 
# 8:  0,3675
# 9:  0,0750
# consider only targets 0, 1, 2, 4, 8 and choose the weights which give
# the best corr (on average, over the splits)

ran_w1 = [x / 100 for x in range(0, 20, 5)]
ran_w2 = [x / 100 for x in range(-30, 5, 5)]
ran_w4 = [x / 100 for x in range(0, 30, 5)]
ran_w8 = [x / 100 for x in range(0, 45, 5)]
len_ws = len(ran_w1) * len(ran_w2) * len(ran_w4) * len(ran_w8)

cv_tar = {
    'w[0]': [],
    'w[1]': [],
    'w[2]': [],
    'w[4]': [],
    'w[8]': [],
}
for j in range(n_splits):
    cv_tar[f'split_{j + 1}'] = []

for j, (trn, val) in tqdm(enumerate(spl.split(X, y, e)), 
                           desc='cv split', 
                           total=n_splits,
                           leave=None,
                           position=0):
    X_trn = X.iloc[trn]
    X_val = X.iloc[val]
    y_trn = Y.iloc[trn]
    y_val = Y.iloc[val]
    e_trn = e.iloc[trn]
    e_val = e.iloc[val]

    model_name = f'model-0/saved-variables/cv_targets_{j}.pkl'
    try:
        model = joblib.load(model_name)
    except IOError:
        model = LGBMRegressor(**params)
        model = EraSubsampler(model, n_subsamples=n_subsamples)
        model = MultiOutputRegressor(model)
        model.fit(X_trn, y_trn, eras=e_trn)
        joblib.dump(model, model_name)

    y_name = f'model-0/saved-variables/y_targets_{j}.pkl'
    try:
        y_val_pred = joblib.load(y_name)
    except:
        y_val_pred = model.predict(X_val)
        joblib.dump(y_val_pred, y_name)
    y_val_true = y_val[Y_TRUE]

    for (w1, w2, w4, w8) in tqdm(product(ran_w1, ran_w2, ran_w4, ran_w8), 
                                 desc='weights', 
                                 total=len_ws,
                                 leave=False,
                                 position=1):
        w0 = 1 - w1 - w2 - w4 - w8
        w = np.array([w0, w1, w2, 0, w4, 0, 0, 0, w8, 0])
        y_val_pred_w = y_val_pred @ w
        
        c = corr(y_val_true, y_val_pred_w, rank_b=e_val)

        cv_tar[f'split_{j + 1}'].append(c)
        if j == 0:
            cv_tar['w[0]'].append(w0)
            cv_tar['w[1]'].append(w1)
            cv_tar['w[2]'].append(w2)
            cv_tar['w[4]'].append(w4)
            cv_tar['w[8]'].append(w8)

cv_tar = pd.DataFrame(cv_tar)
cv_tar['mean'] = cv_tar[[f'split_{j+1}' for j in range(n_splits)]].mean(axis=1)
cv_tar.to_excel('model-0/cross-validation/cv_tar.xlsx')

ar_max = cv_tar['mean'].argmax()
w0 = cv_tar['w[0]'][ar_max]
w1 = cv_tar['w[1]'][ar_max]
w2 = cv_tar['w[2]'][ar_max]
w4 = cv_tar['w[4]'][ar_max]
w8 = cv_tar['w[8]'][ar_max]
w = np.array([w0, w1, w2, 0, w4, 0, 0, 0, w8, 0])
print(f'w = {w}')
# w = [0.6, 0.1, -0.25, 0, 0.2, 0, 0, 0, 0.35, 0]

# ----------------------------------------------------------------------
# feature neutralization
# ----------------------------------------------------------------------

ran_nf = range(5, 55, 5)
ran_al = [x / 100 for x in range(0, 105, 5)]

cv_neu = {
    'n_features': [],
    'alpha': [],
}
for j in range(n_splits):
    cv_neu[f'split_{j + 1}'] = []

for j, (trn, val) in tqdm(enumerate(spl.split(X, y, e)), 
                           desc='cv split', 
                           total=n_splits,
                           leave=None,
                           position=0):
    X_val = X.iloc[val]
    y_val = Y.iloc[val]
    e_val = e.iloc[val]
    y_val_true = y_val[Y_TRUE]
    y_val_pred = joblib.load(f'model-0/saved-variables/y_targets_{j}.pkl')
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
            c = corr(y_val_true, y_neut, rank_b=e_val)

            cv_neu[f'split_{j + 1}'].append(c)
            if j == 0:
                cv_neu['n_features'].append(nf)
                cv_neu['alpha'].append(al)
            

cv_neu = pd.DataFrame(cv_neu)
cv_neu['mean'] = cv_neu[[f'split_{j+1}' for j in range(n_splits)]].mean(axis=1)
cv_neu.to_excel('model-0/cross-validation/cv_neu.xlsx')

ar_max = cv_neu['mean'].argmax()
n_features = cv_neu['n_features'][ar_max]
alpha = cv_neu['alpha'][ar_max]
print(f'n_features = {n_features}')
print(f'alpha = {alpha}')
# n_features = 30, alpha = 0.3

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