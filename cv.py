from math import prod
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from utils import *
from lightgbm import LGBMRegressor


class CVEraBoosterLGBM(BaseEstimator, RegressorMixin):
    def __init__(self, n_est, n_iters, max_depth, frac_leaves, reg_lambda):
        self.n_est = n_est
        self.n_iters = n_iters
        self.max_depth = max_depth
        self.frac_leaves = frac_leaves
        self.reg_lambda = reg_lambda

    def fit(self, X, y, eras, **fit_params):
        X, y = check_X_y(X, y, accept_sparse=True)
        
        params = {
            'n_estimators': self.n_est // self.n_iters,
            'learning_rate': 20 / self.n_est,
            'max_depth': self.max_depth,
            'num_leaves': round(2**self.max_depth * self.frac_leaves),
            'colsample_bytree': 0.1,
            'reg_lambda': self.reg_lambda,
            'device': 'gpu',
        }
        self.model = EraBooster(LGBMRegressor(**params), 
                                n_iters=self.n_iters, 
                                era_fraction=0.5)
        self.model.fit(X, y, eras, **fit_params)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return self.model.predict(X)


param_grid = {
    'n_iters': [1, 10, 100],
    'max_depth': [5, 7, 9],
    'frac_leaves': [1, 0.6],
    'reg_lambda': [0, 10],
}

n_splits = 4
cv_splitter = TimeSeriesSplitGroups(n_splits=n_splits)

def corr_scorer(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]
scorer = make_scorer(corr_scorer, greater_is_better=True)

model = CVEraBoosterLGBM(
    n_est=2000, 
    n_iters=2, 
    max_depth=2, 
    frac_leaves=2, 
    reg_lambda=2,
)
model = GridSearchCV(
    estimator=model, 
    param_grid=param_grid,
    scoring=scorer,
    cv=cv_splitter,
    verbose=3,
)


model_path = 'cv/model_gs.pkl'
try:
    model = joblib.load(model_path)
except:
    X, y, e = read_Xye('full', X_COLS, [Y_TRUE], np.arange(219, 1020, 4))
    model.fit(X, y, eras=e, groups=e)
    joblib.dump(model, model_path)

cv_res = pd.DataFrame(model.cv_results_)
cv_res.to_excel('cv/cv_results.xlsx')


# ======================================================================
# weights for MultiTargetTrainer
# ======================================================================

ranges_aux = [
    range(0, 5, 5),   # 0
    range(0, 20, 5),  # 1
    range(-30, 5, 5), # 2
    range(-5, 10, 5), # 3
    range(0, 30, 5),  # 4
    range(-5, 10, 5), # 5
    range(-5, 10, 5), # 6
    range(-5, 10, 5), # 7
    range(0, 45, 5),  # 8
    range(-5, 10, 5), # 9
]
ranges = [[x / 100 for x in ran] for ran in ranges_aux]
len_iter = prod([len(ran) for ran in ranges_aux])

# ranges = [
#     [x / 100 for x in range(0, 5, 5)],   # 0
#     [x / 100 for x in range(0, 20, 5)],  # 1
#     [x / 100 for x in range(-30, 5, 5)], # 2
#     [x / 100 for x in range(-5, 10, 5)], # 3
#     [x / 100 for x in range(0, 30, 5)],  # 4
#     [x / 100 for x in range(-5, 10, 5)], # 5
#     [x / 100 for x in range(-5, 10, 5)], # 6
#     [x / 100 for x in range(-5, 10, 5)], # 7
#     [x / 100 for x in range(0, 45, 5)],  # 8
#     [x / 100 for x in range(-5, 10, 5)], # 9
# ]
