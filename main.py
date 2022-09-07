# ======================================================================
# imports
# ======================================================================
import numpy as np
from lightgbm import LGBMRegressor
from utils import *
import os
import joblib


# ======================================================================
# NumerAPI + update datasets
# ======================================================================
with open('data/keys.json', 'r') as f:
    keys = json.load(f)
pub = keys['public']
sec = keys['secret']
napi = NumerAPI(pub, sec)
round = napi.get_current_round()
update_dataset('live')


# ======================================================================
# define parameters for the model
# ======================================================================
params = {
    'n_estimators': 20000,
    'learning_rate': 0.001,
    'max_depth': 7,
    'num_leaves': 5 * 2**(7 - 3),
    'colsample_bytree': 0.1,
    'device': 'gpu',
}
n_subsamples = 4
y_cols = [
    'target_nomi_v4_20',    # 0, weight = 1/2
    'target_jerome_v4_20',  # 1, weight = 1/6 
    'target_alan_v4_20',    # 4, weight = 1/6
    'target_arthur_v4_20',  # 8, weight = 1/6
]
weights = np.array([3, 1, 1, 1]) / 6


# ======================================================================
# fit in train + validation, predict in live
# ======================================================================

# fit
os.makedirs('models', exist_ok=True)
path_trn = 'models/model_ful.pkl'
try:
    model = joblib.load(path_trn)
except:
    X, y, e = read_Xye('full', X_COLS, y_cols)
    model = LGBMRegressor(**params)
    model = EraSubsampler(model, n_subsamples)
    model = MultiOutputTrainer(model, weights)
    model.fit(X, y, eras=e)
    joblib.dump(model, path_trn)

# predict
os.makedirs('predictions', exist_ok=True)
path_prd = f'predictions/live_{round}.csv'
df = read_df('live', X_COLS, y_cols)
df[Y_PRED] = model.predict(df[X_COLS])
df[Y_RANK] = df[Y_PRED].rank(pct=True)
df[Y_RANK].to_csv(path_prd)

# submit
with open('data/model_name.txt', 'r') as f:
    model_name = f.read()
model_id = napi.get_models()[model_name]
napi.upload_predictions(path_prd, model_id=model_id)