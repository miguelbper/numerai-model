# ======================================================================
# imports
# ======================================================================

# Numerai API
from numerapi import NumerAPI

# data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# machine learning models
from sklearn.linear_model import LinearRegression

# other
import json
from tqdm import tqdm

# my utils
from utils import *


# ======================================================================
# download data + features + dataframe
# ======================================================================

napi = NumerAPI()
round = napi.get_current_round()
era = round + 695

napi.download_dataset('v4/features.json', 'data/features.json')
napi.download_dataset('v4/train_int8.parquet', 'data/train.parquet')
napi.download_dataset('v4/validation_int8.parquet', 'data/validation.parquet')
napi.download_dataset('v4/live_int8.parquet', f'data/live_{round}.parquet')

ITC = 'intercept'
COEFS = X_COLS + [ITC]

df = read_data('train', X_COLS)


# ======================================================================
# Compute coefs of linear regression by era
# ======================================================================

def coefs_linreg(df):
    model = LinearRegression()
    model.fit(df[X_COLS], df[Y_TRUE])
    y_prd = pd.Series(model.predict(df[X_COLS]))
    y_rnk = y_prd.rank(pct=True)
    ncorr = np.corrcoef(df[Y_TRUE], y_rnk)[0, 1]
    coefs = {X_COLS[i]: [model.coef_[i]] for i in range(model.n_features_in_)}
    coefs[ITC] = [model.intercept_]
    coefs['corr'] = [ncorr]
    coefs = pd.DataFrame(coefs)
    return coefs


df_coefs = df.groupby(ERA).apply(coefs_linreg)
df_coefs[ERA] = np.arange(len(df_coefs)) + 1


# ======================================================================
# Do a linear regression to predict the coefs as a function of the era
# ======================================================================

X = df_coefs[ERA].to_numpy().reshape(-1, 1)
y = df_coefs[COEFS].to_numpy()

coef_predictor = LinearRegression()
coef_predictor.fit(X, y)

pred_coefs = ['pred_' + c for c in COEFS]
df_coefs[pred_coefs] = coef_predictor.predict(X)


# ======================================================================
# Predict coefficients for current era. Make final predictions
# ======================================================================

coef_predictions = coef_predictor.predict(np.array([[era]]))
w = np.array(coef_predictions[0][0:-1])
b = coef_predictions[0][-1]

df_liv = read_data('live', X_COLS)
df_liv[Y_PRED] = df_liv[X_COLS] @ w + b


# ======================================================================
# Plots for coefs as a function of the era
# ======================================================================

for c in tqdm(COEFS):
    fig, ax = plt.subplots()
    ax.plot(df_coefs[ERA], df_coefs[c], label='coefs')
    ax.plot(df_coefs[ERA], df_coefs['pred_' + c], label='linreg')
    ax.set_xlabel('era')
    ax.set_ylabel('coef')
    ax.set_title(f'coef for {c} as function of era')
    ax.legend()

    fig.savefig(f'model-1/figures/{c}.png')
    plt.close(fig)