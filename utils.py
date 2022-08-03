import json
import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import MultiOutputRegressor
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

ERA = 'era'
DATA = 'data_type'
X_COLS = FEATURES_L
Y_COLS = joblib.load('data/target_names.pkl')
Y_TRUE = 'target_nomi_v4_20'
Y_PRED = 'target_prediction'
Y_RANK = 'prediction' 

COLUMNS = [ERA, DATA] + X_COLS + Y_COLS

del feature_metadata
del f

# ---------------
# score functions



# -------
# classes

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
    def __init__(self, estimator, n_subsamples):
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
    def __init__(self, estimator, features=None, n_jobs=None):
        self.estimator = estimator
        self.features = features
        self.n_jobs = n_jobs

    def fit(self, X, y, **fit_params):
        self.model = MultiOutputRegressor(self.estimator, n_jobs=self.n_jobs)
        self.model.fit(X, y, **fit_params)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        y_preds = self.model.predict(X)
        indices = self.features
        indices = indices if indices is not None else range(len(y_preds[0]))
        return np.average(y_preds[:, indices], axis=1)


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
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds, n_groups))
        indices = np.arange(n_samples)
        test_size = (n_groups // n_folds)
        test_starts = range(test_size + n_groups % n_folds,
                            n_groups, test_size)
        test_starts = list(test_starts)[::-1]
        for test_start in test_starts:
            yield (indices[groups.isin(group_list[:test_start])],
                   indices[groups.isin(group_list[test_start:test_start + test_size])])