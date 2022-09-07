# ======================================================================
# imports
# ======================================================================
import numpy as np
from lightgbm import LGBMRegressor
from utils import *
import os


# ======================================================================
# NumerApi + update datasets
# ======================================================================
pub, sec = '' # TODO: have keys in a json
napi = NumerAPI(pub, sec)
round = napi.get_current_round()
update_dataset('train')
update_dataset('validation')
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
# TODO: check in old code that 0, 1, 4, 8 are the desired targets
y_cols = [
    'target_nomi_v4_20',    # 0, weight = 1/2
    'target_jerome_v4_20',  # 1, weight = 1/6 
    'target_alan_v4_20',    # 4, weight = 1/6
    'target_arthur_v4_20',  # 8, weight = 1/6
]
weights = np.array([3, 1, 1, 1]) / 6


# ======================================================================
# fit in train, predict in validation
# ======================================================================

# fit
os.makedirs('models', exist_ok=True)
path_trn = 'models/model_trn.pkl'
try:
    model = joblib.load(path_trn)
except:
    X, y, e = read_Xye('train', X_COLS, y_cols)
    model = LGBMRegressor(**params)
    model = EraSubsampler(model, n_subsamples)
    model = MultiOutputTrainer(model, weights)
    model.fit(X, y, eras=e)
    joblib.dump(model, path_trn)

# predict
os.makedirs('predictions', exist_ok=True)
path_prd = f'predictions/validation_{round}.csv'
df = read_df('validation', X_COLS, y_cols)
df[Y_PRED] = model.predict(df[X_COLS])
df[Y_RANK] = df[Y_PRED].rank(pct=True)
df[Y_RANK].to_csv(path_prd)

# submit
with open('data/model_name.txt', 'r') as f:
    model_name = f.read()
model_id = napi.get_models()[model_name]
napi.upload_diagnostics(path_prd, model_id=model_id)