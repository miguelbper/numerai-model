from catboost import CatBoostRegressor
import os
import sys
sys.path.append(os.path.abspath('.'))
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from utils import *


class CatBoostCV(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators, max_depth, frac_leaves):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.frac_leaves = frac_leaves

    def fit(self, X, y, **fit_params):
        X, y = check_X_y(X, y, accept_sparse=True)

        params = {
            'n_estimators': self.n_estimators,
            'learning_rate': 20 / self.n_estimators,
            'max_depth': self.max_depth,
            'logging_level': 'Silent',
            'task_type': 'GPU', 
            'devices': '0',
        }
        if self.frac_leaves:
            params['grow_policy'] = 'Lossguide'
            params['num_leaves'] = 5 * 2**(self.max_depth - 3)

        self.model = CatBoostRegressor(**params)
        self.model.fit(X, y, **fit_params)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return self.model.predict(X)


param_grid = {
    'max_depth': [5, 7, 9],
    'frac_leaves': [True, False],
}

n_splits = 4
cv_splitter = TimeSeriesSplitGroups(n_splits=n_splits)

def corr_scorer(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]
scorer = make_scorer(corr_scorer, greater_is_better=True)

model = GridSearchCV(
    estimator=CatBoostCV(n_estimators=2000, max_depth=5, frac_leaves=1), 
    param_grid=param_grid,
    scoring=scorer,
    cv=cv_splitter,
    verbose=3,
)

X, y, e = read_Xye('train', X_COLS, [Y_TRUE])
model.fit(X, y, groups=e)
joblib.dump(model, './cv/cv-2/model.pkl')

cv_res = pd.DataFrame(model.cv_results_)
cv_res.to_excel('./cv/cv-2/cv_results.xlsx')