import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from copy import deepcopy
from math import ceil

# ------------
# column names

with open('data/features.json', 'r') as f:
    feature_metadata = json.load(f)

FEATURES_L = list(feature_metadata['feature_stats'].keys())
FEATURES_M = feature_metadata['feature_sets']['medium']
FEATURES_S = feature_metadata['feature_sets']['small']
# FEATURES_2 = feature_metadata['feature_sets']['v2_equivalent_features']
# FEATURES_3 = feature_metadata['feature_sets']['v3_equivalent_features']
# FEATURES_N = feature_metadata['feature_sets']['fncv3_features']

# https://forum.numer.ai/t/removing-dangerous-features/5627
INVALID_FEATURES = ['feature_palpebral_univalve_pennoncel',
                    'feature_unsustaining_chewier_adnoun',
                    'feature_brainish_nonabsorbent_assurance',
                    'feature_coastal_edible_whang',
                    'feature_disprovable_topmost_burrower',
                    'feature_trisomic_hagiographic_fragrance',
                    'feature_queenliest_childing_ritual',
                    'feature_censorial_leachier_rickshaw',
                    'feature_daylong_ecumenic_lucina',
                    'feature_steric_coxcombic_relinquishment']
# i =  193, f = feature_palpebral_univalve_pennoncel
# i =  208, f = feature_unsustaining_chewier_adnoun
# i =  403, f = feature_brainish_nonabsorbent_assurance
# i =  418, f = feature_coastal_edible_whang
# i =  613, f = feature_disprovable_topmost_burrower
# i =  628, f = feature_trisomic_hagiographic_fragrance
# i =  823, f = feature_queenliest_childing_ritual
# i =  838, f = feature_censorial_leachier_rickshaw
# i = 1033, f = feature_daylong_ecumenic_lucina
# i = 1048, f = feature_steric_coxcombic_relinquishment

FEAT_L = [f for f in FEATURES_L if f not in INVALID_FEATURES]
FEAT_M = [f for f in FEATURES_M if f not in INVALID_FEATURES]
FEAT_S = [f for f in FEATURES_S if f not in INVALID_FEATURES]

ERA = 'era'
DATA = 'data_type'
X_COLS = FEAT_L
Y_COLS = joblib.load('data/target_names.pkl')
Y_TRUE = 'target_nomi_v4_20'
Y_PRED = 'target_prediction'
Y_RANK = 'prediction' 

COLUMNS = [ERA, DATA] + X_COLS + Y_COLS

del feature_metadata
del f

# ---------------
# score functions

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

# -------
# classes

class FeatureSubsampler(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, n_features_per_group=208):
        self.estimator = estimator
        self.n_features_per_group = n_features_per_group

    def fit(self, X, y, **fit_params):
        X, y = check_X_y(X, y, accept_sparse=True)
        n = len(X[0])
        l = self.n_features_per_group
        l = l if l > 0 else n
        k = ceil(n / l)
        self.model = [deepcopy(self.estimator) for i in range(k)]
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


class EraSubsampler(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, n_subsamples=4):
        self.estimator = estimator
        self.n_subsamples = n_subsamples

    def fit(self, X, y, eras):
        X, y = check_X_y(X, y, accept_sparse=True)
        e0 = eras.min()
        e1 = eras.max() + 1
        k = self.n_subsamples
        self.model = [deepcopy(self.estimator) for i in range(k)]
        for i in range(k):
            self.model[i].fit(X[eras.isin(np.arange(e0 + i, e1, k))], 
                              y[eras.isin(np.arange(e0 + i, e1, k))])
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


class MultiTargetTrainer(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, targets=None, n_jobs=None):
        self.estimator = estimator
        self.targets = targets
        self.n_jobs = n_jobs

    def fit(self, X, y, **fit_params):
        self.model = MultiOutputRegressor(self.estimator, n_jobs=self.n_jobs)
        self.model.fit(X, y, **fit_params)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        y_preds = self.model.predict(X)
        indices = self.targets
        indices = indices if indices is not None else range(len(y_preds[0]))
        return np.average(y_preds[:, indices], axis=1)


# alternative way of implementing FeatureNeutralizer:
# instead of giving groups as separate argument, give them inside matrix X
# adv: more compatible with sklearn
# disadv: can't use GridSearchCV anyway (unpractical), bandaid, 
# less clarity when using class
class FeatureNeutralizer(BaseEstimator):
    def __init__(self, estimator, n_features, alpha):
        self.estimator = estimator
        self.n_features = n_features
        self.alpha = alpha

    def fit(self, X, y, **fit_params):
        # X, y = check_X_y(X, y, accept_sparse=True)
        self.estimator.fit(X, y, **fit_params)
        self.is_fitted_ = True
        return self
    
    def compute_y_pred(self, X):
        # checks
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        # computations
        self.y_pred = self.estimator.predict(X)

    def compute_y_linr(self, X, groups):
        # checks
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        if not hasattr(self, 'y_pred'):
            self.compute_y_pred(X)
        if self.n_features == 0:
            self.y_linr = 0
            return
        # computations
        y_pred = self.y_pred
        groups = np_(groups)
        # n riskiest features
        exposures = corr(X, y_pred, rank_a=groups, rank_b=groups)
        riskiest = [(v, i) for i, v in enumerate(exposures)]
        riskiest = sorted(riskiest, reverse=True)
        riskiest = riskiest[0:self.n_features]
        riskiest = [i for _, i in riskiest]
        # auxiliary function
        def aux_linreg(df):
            X_ = df[df.columns[0:-1]]
            y_ = df[df.columns[-1]]
            model = LinearRegression()
            model.fit(X_, y_)
            return pd.Series(model.predict(X_) - model.intercept_)
        # result
        R = X[:, riskiest]
        df_Ry = pd.DataFrame(np.hstack((R, np.atleast_2d(y_pred).T)))
        self.y_linr = df_Ry.groupby(groups).apply(aux_linreg).to_numpy()

    def predict(self, X, groups):
        # checks
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        if not hasattr(self, 'y_pred'):
            self.compute_y_pred(X)
        if not hasattr(self, 'y_linr'):
            self.compute_y_linr(X, groups)
        # computations
        return self.y_pred - self.alpha * self.y_linr

    # cannot import score from regressor mixin
    # because predict needs a groups parameter
    def score(self, X, y, groups, sample_weight=None):
        y_pred = self.predict(X, groups)
        return r2_score(y, y_pred, sample_weight=sample_weight)


# The following class is taken from the examples from Numerai
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