# -------
# imports

from numerapi import NumerAPI
from lightgbm import LGBMRegressor
import joblib
from utils import *


# -------------
# download data

napi = NumerAPI()
round = napi.get_current_round()
era = round + 695
napi.download_dataset('v4/live_int8.parquet', f'data/live_{round}.parquet')


# ------------------------
# define model and dataset

params = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 5,
    'num_leaves': 2**5,
    'colsample_bytree': 0.1,
    'device': 'gpu',
}

model = LGBMRegressor(**params)
model = EraSubsampler(model, n_subsamples=7)


# -----
# train

df_trn = read_data('train')
model.fit(df_trn[X_COLS], df_trn[Y_TRUE], eras=df_trn[ERA])
joblib.dump(model, f'model-0/saved-variables/lgbm_{now_dt()}.pkl')


# -------
# predict

df_liv = read_data('live')
df_liv[Y_PRED] = model.predict(df_liv[X_COLS])
df_liv[Y_RANK] = df_liv[Y_PRED].rank(pct=True)
df_liv[Y_RANK].to_csv(f'model-0/predictions/lgbm_live_predictions_{round}_{now_dt()}.csv')