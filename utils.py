import json
import joblib
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
import numpy as np

# ------------
# column names

with open('data/features.json', 'r') as f:
    feature_metadata = json.load(f)

FEATURES_L = list(feature_metadata['feature_stats'].keys())
# FEATURES_M = feature_metadata['feature_sets']['medium']
# FEATURES_S = feature_metadata['feature_sets']['small']
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