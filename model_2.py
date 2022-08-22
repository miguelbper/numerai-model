# ======================================================================
# imports
# ======================================================================

from numerapi import NumerAPI
from lightgbm import LGBMRegressor
import joblib
import gc
from utils import *

pth = 'model-2'

# ======================================================================
# download data
# ======================================================================

pub, sec = joblib.load('keys.pkl')

napi = NumerAPI(pub, sec)
round = napi.get_current_round()
era = round + 695
# napi.download_dataset('v4/features.json', 'data/features.json')
# napi.download_dataset('v4/train_int8.parquet', 'data/train.parquet')
# napi.download_dataset('v4/validation_int8.parquet', 'data/validation.parquet')
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

if False:

    x_cols = FEAT_L
    eras = None
    n_splits = 4

    df_1 = read_data('train', X_COLS, Y_COLS)
    df_2 = read_data('validation', X_COLS, Y_COLS)
    df_2 = df_2[df_2[DATA] == 'validation']
    df = pd.concat([df_1, df_2])
    del df_1, df_2
    gc.collect()

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

    ran_ns = range(4, 13)

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

            model_name = f'{pth}/saved-variables/cv_era_{i}_{j}.pkl'
            try:
                model = joblib.load(model_name)
            except:
                model = EraSubsampler(LGBMRegressor(**params), n_subsamples=i)
                model.fit(X_trn, y_trn, eras=e_trn)
                joblib.dump(model, model_name)

            y_name = f'{pth}/saved-variables/y_era_{i}_{j}.pkl'
            try:
                y_val_pred = joblib.load(y_name)
            except:
                y_val_pred = model.predict(X_val)
                joblib.dump(y_val_pred, y_name)

            c = corr(y_val, y_val_pred, rank_b=e_val)
            cv_era[f'split_{j + 1}'].append(c)

    cv_era = pd.DataFrame(cv_era)
    cv_era['mean'] = cv_era[[f'split_{j+1}' for j in range(n_splits)]].mean(axis=1)
    cv_era.to_excel(f'{pth}/cross-validation/cv_era.xlsx')
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

    ran_w1 = [x / 100 for x in range(0, 30, 5)]
    ran_w2 = [x / 100 for x in range(-40, 5, 5)]
    ran_w4 = [x / 100 for x in range(0, 40, 5)]
    ran_w8 = [x / 100 for x in range(0, 55, 5)]
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

        model_name = f'{pth}/saved-variables/cv_targets_{j}.pkl'
        try:
            model = joblib.load(model_name)
        except IOError:
            model = LGBMRegressor(**params)
            model = EraSubsampler(model, n_subsamples=n_subsamples)
            model = MultiOutputRegressor(model)
            model.fit(X_trn, y_trn, eras=e_trn)
            joblib.dump(model, model_name)

        y_name = f'{pth}/saved-variables/y_targets_{j}.pkl'
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
    cv_tar.to_excel(f'{pth}/cross-validation/cv_tar.xlsx')

    ar_max = cv_tar['mean'].argmax()
    w0 = cv_tar['w[0]'][ar_max]
    w1 = cv_tar['w[1]'][ar_max]
    w2 = cv_tar['w[2]'][ar_max]
    w4 = cv_tar['w[4]'][ar_max]
    w8 = cv_tar['w[8]'][ar_max]
    w = np.array([w0, w1, w2, 0, w4, 0, 0, 0, w8, 0])
    print(f'w = {w}')
    # w = [0.5, 0.1, -0.1, 0, 0.15, 0, 0, 0, 0.35, 0]

else:
    n_subsamples = 7
    w = np.array([0.5, 0.1, -0.1, 0, 0.15, 0, 0, 0, 0.35, 0])

y_inds = [0, 1, 2, 4, 8]
y_cols = [Y_COLS[i] for i in y_inds]
w = np.array([w[i] for i in y_inds])

# ======================================================================
# final model
# ======================================================================

model = LGBMRegressor(**params)
model = EraSubsampler(model, n_subsamples=n_subsamples)
model = MultiOutputTrainer(model, weights=w)


# ----------------------------------------------------------------------
# train
# ----------------------------------------------------------------------

model_name = f'{pth}/saved-variables/model.pkl'
try:
    model = joblib.load(model_name)
except:
    df_1 = read_data('train', X_COLS, y_cols)
    df_2 = read_data('validation', X_COLS, y_cols)
    df_2 = df_2[df_2[DATA] == 'validation']
    df = pd.concat([df_1, df_2])
    del df_1, df_2
    gc.collect()

    model.fit(df[X_COLS], df[y_cols], eras=df[ERA])
    joblib.dump(model, model_name)


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------

if False:
    df = read_data('validation', X_COLS, y_cols)
    df[Y_PRED] = model.predict(df[X_COLS])
    df[Y_RANK] = df[Y_PRED].rank(pct=True)
    df[Y_RANK].to_csv(f'{pth}/predictions/val_{round}_{now_dt()}.csv')


# ----------------------------------------------------------------------
# live
# ----------------------------------------------------------------------

pred_name = f'{pth}/predictions/liv_{round}.csv'

df = read_data('live', X_COLS, y_cols)
df[Y_PRED] = model.predict(df[X_COLS])
df[Y_RANK] = df[Y_PRED].rank(pct=True)
df[Y_RANK].to_csv(pred_name)

model_id = napi.get_models()['mbp_2']
napi.upload_predictions(pred_name, model_id=model_id)