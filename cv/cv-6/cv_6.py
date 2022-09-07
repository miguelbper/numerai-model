# imports
from lightgbm import LGBMRegressor
import os
import sys
sys.path.append(os.path.abspath('.'))
from utils import *
from math import prod
from itertools import product
from tqdm import tqdm


# options for CV
x_cols = X_COLS
y_cols = Y_COLS
n_splits = 4


# dataset and CV splitter
X, y, e = read_Xye('full', x_cols, y_cols)
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


# ranges
ranges_aux = [
    range(0,  5, 5), # 0
    range(0, 20, 5), # 1
    range(0, 10, 5), # 2
    range(0, 10, 5), # 3
    range(0, 30, 5), # 4
    range(0, 10, 5), # 5
    range(0, 10, 5), # 6
    range(0, 10, 5), # 7
    range(0, 45, 5), # 8
    range(0, 10, 5), # 9
]
ranges = [[x / 100 for x in ran] for ran in ranges_aux]
len_iter = prod([len(ran) for ran in ranges_aux])


# scores dict
results = dict()
for i in range(10):
    results[f'w[{i}]'] = []
for j in range(n_splits):
    results[f'split_{j}'] = []

# cv loop
for j, (trn, val) in enumerate(spl.split(X, y, e)):
    print(f'cv iter {j}')
    X_trn = X.iloc[trn]
    X_val = X.iloc[val]
    y_trn = y.iloc[trn]
    y_val = y.iloc[val]
    e_trn = e.iloc[trn]
    e_val = e.iloc[val]
    
    model_path = f'./cv/cv-6/model_{j}.pkl'
    try:
        model = joblib.load(model_path)
    except:
        model = LGBMRegressor(**params)
        model = EraSubsampler(model, n_subsamples=4)
        model = MultiOutputRegressor(model)
        model.fit(X_trn, y_trn, eras=e_trn)
        joblib.dump(model, model_path)

    y_true = y_val[Y_TRUE]
    y_pred = model.predict(X_val)

    for w in tqdm(product(*ranges), total=len_iter):
        w = np.array(w)
        w[0] = 1 - sum(w[1:])

        for i in range(10):
            if j == 0:
                results[f'w[{i}]'].append(w[i])

        score = corr(y_true, y_pred @ w, rank_b=e_val)
        results[f'split_{j}'].append(score)

res = pd.DataFrame(results)
res['mean'] = res[[f'split_{j}' for j in range(n_splits)]].mean(axis=1)
res.to_excel('./cv/cv-6/res.xlsx')