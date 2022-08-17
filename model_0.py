# ======================================================================
# imports
# ======================================================================

from numerapi import NumerAPI
from lightgbm import LGBMRegressor
import joblib
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
y = df[Y_TRUE]
e = df[ERA]
del df

spl = TimeSeriesSplitGroups(n_splits=n_splits)

# ----------------------------------------------------------------------
# number of era subsamples
# ----------------------------------------------------------------------

# subsamples = range(1, 9)
# corrs_nsubsamples = np.zeros((len(subsamples), n_splits))

# for i in tqdm(subsamples, desc='i', leave=False):
#     model = EraSubsampler(LGBMRegressor(**params), n_subsamples=i)
#     j = -1
#     for trn, val in tqdm(spl.split(X, y, e), desc='j', leave=False, total=n_splits):
#         j += 1
#         X_trn = X.iloc[trn]
#         X_val = X.iloc[val]
#         y_trn = y.iloc[trn]
#         y_val = y.iloc[val]
#         e_trn = e.iloc[trn]
#         e_val = e.iloc[val]

#         model.fit(X_trn, y_trn, eras=e_trn)

#         y_val_true = y_val
#         y_val_pred = model.predict(X_val)

#         corrs_nsubsamples[i - 1, j] = corr(y_val_true, y_val_pred, rank_b=e_val)

# corrs_nsubsamples = pd.DataFrame(corrs_nsubsamples)
# corrs_nsubsamples['mean'] = corrs_nsubsamples.mean(axis=1)

# ----------------------------------------------------------------------
# weights for training on many targets
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# number of features to neutralize
# ----------------------------------------------------------------------



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