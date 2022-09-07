# ======================================================================
# imports
# ======================================================================

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from copy import deepcopy
from math import ceil
from numerapi import NumerAPI

# ======================================================================
# column names
# ======================================================================

# TODO: download features.json in utils
with open('data/features.json', 'r') as f:
    feature_metadata = json.load(f)

FEATURES_L = list(feature_metadata['feature_stats'].keys())
FEATURES_M = feature_metadata['feature_sets']['medium']
FEATURES_S = feature_metadata['feature_sets']['small']
# FEATURES_2 = feature_metadata['feature_sets']['v2_equivalent_features']
# FEATURES_3 = feature_metadata['feature_sets']['v3_equivalent_features']
# FEATURES_N = feature_metadata['feature_sets']['fncv3_features']

# https://forum.numer.ai/t/removing-dangerous-features/5627
INVALID_FEATURES = [
    'feature_palpebral_univalve_pennoncel',    # i =  193
    'feature_unsustaining_chewier_adnoun',     # i =  208
    'feature_brainish_nonabsorbent_assurance', # i =  403
    'feature_coastal_edible_whang',            # i =  418
    'feature_disprovable_topmost_burrower',    # i =  613
    'feature_trisomic_hagiographic_fragrance', # i =  628
    'feature_queenliest_childing_ritual',      # i =  823
    'feature_censorial_leachier_rickshaw',     # i =  838
    'feature_daylong_ecumenic_lucina',         # i = 1033
    'feature_steric_coxcombic_relinquishment', # i = 1048
]

FEAT_L = [f for f in FEATURES_L if f not in INVALID_FEATURES]
FEAT_M = [f for f in FEATURES_M if f not in INVALID_FEATURES]
FEAT_S = [f for f in FEATURES_S if f not in INVALID_FEATURES]

ERA = 'era'
DATA = 'data_type'
X_COLS = FEAT_L
Y_TRUE = 'target_nomi_v4_20'
Y_PRED = 'target_prediction'
Y_RANK = 'prediction' 
Y_FULL = [
    'target_nomi_v4_20',
    'target_nomi_v4_60',
    'target_jerome_v4_20',
    'target_jerome_v4_60',
    'target_janet_v4_20',
    'target_janet_v4_60',
    'target_ben_v4_20',
    'target_ben_v4_60',
    'target_alan_v4_20',
    'target_alan_v4_60',
    'target_paul_v4_20',
    'target_paul_v4_60',
    'target_george_v4_20',
    'target_george_v4_60',
    'target_william_v4_20',
    'target_william_v4_60',
    'target_arthur_v4_20',
    'target_arthur_v4_60',
    'target_thomas_v4_20',
    'target_thomas_v4_60',
]
Y_COLS = [y for y in Y_FULL if y.endswith('20')]

COLUMNS = [ERA, DATA] + X_COLS + Y_COLS

del feature_metadata
del f


# ======================================================================
# datasets
# ======================================================================

def update_dataset(dataset):
    ''' Downloads, cleans and writes dataset.

    Args: 
        dataset: a string taking the values 'train', 'validation' or
        'live'. 
    '''

    print(f'\nUpdating dataset: {dataset}')

    # download and load dataset
    napi = NumerAPI()
    round = napi.get_current_round()
    era = round + 695
    name = f'live_{round}' if dataset == 'live' else dataset
    ds_path = f'data/{name}.parquet'
    napi.download_dataset(f'v4/{dataset}_int8.parquet', ds_path)
    df = pd.read_parquet(ds_path)

    # write era
    if dataset == 'live':
        df[ERA] = era

    # era to int
    df[ERA] = df[ERA].astype('int32')

    # fill nans X
    if df[X_COLS].isnull().values.any():
        df[X_COLS] = df[X_COLS].fillna(value=2)

    # fill nans y
    sr = df[DATA].isin(['train', 'validation'])
    if df.loc[sr, Y_FULL].isnull().values.any():
        df.loc[sr, Y_FULL] = df.loc[sr, Y_FULL].fillna(value=0.5)

    # write cleaned dataset
    df.to_parquet(f'data/_{name}.parquet')


def read_df(dataset, x_cols, y_cols, eras=None):
    ''' Reads dataframe from file.
    
    Args:
        dataset: A string taking the values 'train', 'validation', 
            'live' or 'full'. Here, 'full' means that the returned 
            dataset is train + validation (the biggest dataset where we 
            have labels).
        x_cols: The feature columns that should be read.
        y_cols: The target columns that should be read.
        eras: The eras (rows) of the dataframe that should be read. By 
            default, read all of them (eras=None).

    Returns:
        A pandas dataframe containing the dataset. 
    '''

    # definitions
    napi = NumerAPI()
    round = napi.get_current_round()
    name = f'live_{round}' if dataset == 'live' else dataset
    cols = [ERA, DATA] + x_cols + y_cols

    # load dataframe
    if dataset == 'full':
        df0 = pd.read_parquet(f'data/_train.parquet', columns=cols)
        df1 = pd.read_parquet(f'data/_validation.parquet', columns=cols)
        df1 = df1[df1[DATA] == 'validation']
        df = pd.concat([df0, df1])
    else:
        df = pd.read_parquet(f'data/_{name}.parquet', columns=cols)

    # select eras
    if eras is not None:
        df = df[df[ERA].isin(eras)]

    return df


def read_Xy(dataset, x_cols, y_cols, eras=None):
    ''' Reads X, y from file.
    
    Args:
        dataset: A string taking the values 'train', 'validation', 
            'live' or 'full'. Here, 'full' means that the returned 
            dataset is train + validation (the biggest dataset where we 
            have labels).
        x_cols: The feature columns that should be read.
        y_cols: The target columns that should be read.
        eras: The eras (rows) of the dataframe that should be read. By 
            default, read all of them (eras=None).

    Returns:
        X: pandas dataframe
        y: pandas dataframe or series
    '''
    df = read_df(dataset, x_cols, y_cols, eras=eras)
    y_cols = y_cols[0] if len(y_cols) == 1 else y_cols
    X = df[x_cols]
    y = df[y_cols]
    return X, y


def read_Xye(dataset, x_cols, y_cols, eras=None):
    ''' Reads X, y and eras from file.
    
    Args:
        dataset: A string taking the values 'train', 'validation', 
            'live' or 'full'. Here, 'full' means that the returned 
            dataset is train + validation (the biggest dataset where we 
            have labels).
        x_cols: The feature columns that should be read.
        y_cols: The target columns that should be read.
        eras: The eras (rows) of the dataframe that should be read. By 
            default, read all of them (eras=None).

    Returns:
        X: pandas dataframe
        y: pandas dataframe or series
        e: pandas series
    '''
    df = read_df(dataset, x_cols, y_cols, eras=eras)
    y_cols = y_cols[0] if len(y_cols) == 1 else y_cols
    X = df[x_cols]
    y = df[y_cols]
    e = df[ERA]
    return X, y, e


# ======================================================================
# score functions
# ======================================================================

def np_(df):
    return df if isinstance(df, np.ndarray) else df.to_numpy()


def maybe_rank(a, rank):
    if isinstance(rank, np.ndarray) or isinstance(rank, pd.Series):
        return a.groupby(np_(rank)).apply(lambda x: x.rank(pct=True))
    elif rank:
        return a.rank(pct=True)
    else:
        return a


def corr(a, b, rank_a=False, rank_b=False):
    a = np_(maybe_rank(pd.DataFrame(a), rank_a))
    b = np_(maybe_rank(pd.DataFrame(b), rank_b))
    n = 1 if a.ndim == 1 else len(a[0])
    c = np.corrcoef(a, b, rowvar=False)[0:n, n:].squeeze()
    c = c.item() if c.ndim == 0 else c
    return c


# ======================================================================
# classes
# ======================================================================

class EraBooster(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, n_iters, era_fraction, pass_eras=False):
        self.estimator = estimator
        self.n_iters = n_iters
        self.era_fraction = era_fraction
        self.pass_eras = pass_eras

    def fit(self, X, y, eras, **fit_params):
        X, y = check_X_y(X, y, accept_sparse=True)

        u_eras = eras.unique()
        n_eras = len(u_eras)
        m_eras = round(self.era_fraction * n_eras)
        n = self.n_iters
        self.model = [deepcopy(self.estimator) for _ in range(n)]
        predictions = np.zeros((len(X), 0))
        worst_eras = np.arange(len(X))

        for i in range(n):
            X_ = X[worst_eras]
            y_ = y[worst_eras]

            fit_params_0 = dict(fit_params)
            if self.pass_eras:
                fit_params_0['eras'] = eras[worst_eras]

            self.model[i].fit(X_, y_, **fit_params_0)

            y_pred_new = self.model[i].predict(X)
            y_pred_res = np.reshape(y_pred_new, (-1, 1))
            predictions = np.concatenate((predictions, y_pred_res), axis=1)
            y_pred = np.mean(predictions, axis=1)

            def corr_aux(era):
                dfy = pd.DataFrame({'era': eras, 'yt': y, 'yp': y_pred})
                y_true_era = dfy['yt'][dfy['era'] == era]
                y_pred_era = dfy['yp'][dfy['era'] == era]
                return corr(y_true_era, y_pred_era, rank_b=True)
            corrs = [(corr_aux(era), era) for era in u_eras]
                        
            worst_eras = [e for _, e in sorted(corrs)[:m_eras]]
            worst_eras = np.array([i for i, e in enumerate(eras) 
                                   if e in worst_eras])

        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        n = self.n_iters
        y_pred = 0
        for i in range(n):
            y_pred += self.model[i].predict(X)
        y_pred /= n

        return y_pred


class EraSubsampler(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, n_subsamples, pass_eras=False):
        self.estimator = estimator
        self.n_subsamples = n_subsamples
        self.pass_eras = pass_eras

    def fit(self, X, y, eras, **fit_params):
        X, y = check_X_y(X, y, accept_sparse=True)

        e0 = eras.min()
        e1 = eras.max() + 1
        k = self.n_subsamples
        self.model = [deepcopy(self.estimator) for _ in range(k)]
        
        for i in range(k):
            era_indices = eras.isin(np.arange(e0 + i, e1, k))

            fit_params_0 = dict(fit_params)
            if self.pass_eras:
                fit_params_0['eras'] = eras[era_indices]

            self.model[i].fit(X[era_indices], y[era_indices], **fit_params_0)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        k = self.n_subsamples
        y_pred = 0
        for i in range(k):
            y_pred += self.model[i].predict(X)
        y_pred /= k

        return y_pred


class FeatureSubsampler(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, n_features_per_group):
        self.estimator = estimator
        self.n_features_per_group = n_features_per_group

    def fit(self, X, y, **fit_params):
        X, y = check_X_y(X, y, accept_sparse=True)

        n = len(X[0])
        l = self.n_features_per_group
        l = l if l > 0 else n
        k = ceil(n / l)
        self.model = [deepcopy(self.estimator) for _ in range(k)]
        for i in range(k):
            feature_indices = range(i * l, min((i + 1) * l, n))
            self.model[i].fit(X[:, feature_indices], y, **fit_params)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        n = len(X[0])
        l = self.n_features_per_group
        l = l if l > 0 else n
        k = ceil(n / l)
        y_pred = 0
        for i in range(k):
            feature_indices = range(i * l, min((i + 1) * l, n))
            y_pred += self.model[i].predict(X[:, feature_indices])
        y_pred /= k

        return y_pred


class MultiOutputTrainer(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, weights):
        self.estimator = estimator
        self.weights = weights

    def fit(self, X, y, **fit_params):
        self.model = MultiOutputRegressor(self.estimator)
        self.model.fit(X, y, **fit_params)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return self.model.predict(X) @ self.weights


# The following class is taken from the Numerai example scripts
class TimeSeriesSplitGroups(_BaseKFold):
    def __init__(self, n_splits=5):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_list = np.unique(groups)
        n_groups = len(group_list)
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have n_folds ={0} greater"
                 " than the n_samples: {1}.").format(n_folds, n_groups))
        indices = np.arange(n_samples)
        test_size = (n_groups // n_folds)
        test_starts = range(test_size + n_groups % n_folds,
                            n_groups, test_size)
        test_starts = list(test_starts)[::-1]
        for test_start in test_starts:
            yield (indices[groups.isin(group_list[:test_start])],
                   indices[groups.isin(group_list[test_start
                                                  :test_start + test_size])])