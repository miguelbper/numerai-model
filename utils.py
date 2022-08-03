import json
import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from copy import deepcopy

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

# The following class is taken from the examples from Numerai

# Because the TimeSeriesSplit class in sklearn does not use groups and won't 
# respect era boundries, we implement a version that will
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