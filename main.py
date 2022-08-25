from lightgbm import LGBMRegressor
from utils import *


update_dataset('live')

# ======================================================================
# define models
# ======================================================================

# ----------------------------------------------------------------------
# definitions
# ----------------------------------------------------------------------

lgbm0 = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 5,
    'num_leaves': 2**5,
    'colsample_bytree': 0.1,
    'device': 'gpu',
}

lgbm1 = {
    'n_estimators': 20000,
    'learning_rate': 0.001,
    'max_depth': 6,
    'num_leaves': 2**6,
    'colsample_bytree': 0.1,
    'device': 'gpu',
}


# ----------------------------------------------------------------------
# mbp_r
# ----------------------------------------------------------------------

mbp_r = Model(
    estimator=RandomRegressor(),
    name='mbp_r',
    dataset='train',
    x_cols=FEAT_S,
    y_cols=[Y_TRUE],
    predict_only=True,
)


# ----------------------------------------------------------------------
# mbp_vn
# ----------------------------------------------------------------------

mbp_vn = Model(
    estimator=EraSubsampler(LGBMRegressor(**lgbm0), n_subsamples=4),
    name='mbp_vn',
    dataset='full',
    x_cols=X_COLS,
    y_cols=[Y_TRUE],
    eras=np.arange(200, 1000),
    pass_eras=True,
    predict_only=True,
)


# ----------------------------------------------------------------------
# mbp_pr
# ----------------------------------------------------------------------

mbp_pr = Model(
    estimator=EraSubsampler(LGBMRegressor(**lgbm1), n_subsamples=4),
    name='mbp_pr',
    dataset='full',
    x_cols=X_COLS,
    y_cols=[Y_TRUE],
    eras=np.arange(200, 1000),
    pass_eras=True,
    predict_only=True,
)


# ----------------------------------------------------------------------
# mbp_mo
# ----------------------------------------------------------------------

w = [0.5, 0.1, -0.1, 0, 0.15, 0, 0, 0, 0.35, 0]
y_inds = [0, 1, 2, 4, 8]
y_cols = [Y_COLS[i] for i in y_inds]
w = np.array([w[i] for i in y_inds])

m = LGBMRegressor(**lgbm0)
m = EraSubsampler(m, n_subsamples=4)
m = MultiOutputTrainer(m, weights=w)

mbp_mo = Model(
    estimator=m,
    name='mbp_mo',
    dataset='full',
    x_cols=X_COLS,
    y_cols=y_cols,
    eras=np.arange(200, 1000),
    pass_eras=True,
    predict_only=True,
)


# ----------------------------------------------------------------------
# mbp_eb
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# mbp_er
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# mbp_cr
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# mbp_s
# ----------------------------------------------------------------------


# ======================================================================
# train and submit
# ======================================================================

models = [
    mbp_r,
    mbp_vn,
    mbp_pr,
    mbp_mo,
]

for model in models:
    model.train()
    model.submit()