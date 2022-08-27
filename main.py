from lightgbm import LGBMRegressor
from utils import *


update_dataset('live')

# ======================================================================
# define models
# ======================================================================

# ----------------------------------------------------------------------
# definitions
# ----------------------------------------------------------------------

params_0 = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 5,
    'num_leaves': 2**5,
    'colsample_bytree': 0.1,
    'device': 'gpu',
}

params_1 = {
    'n_estimators': 20000,
    'learning_rate': 0.001,
    'max_depth': 6,
    'num_leaves': 2**6,
    'colsample_bytree': 0.1,
    'device': 'gpu',
}

era_1 = 1019
era_0 = era_1 - 800

# ----------------------------------------------------------------------
# mbp_rnd (random predictions)
# ----------------------------------------------------------------------

mbp_rnd = Model(
    estimator=RandomRegressor(),
    name='mbp_rnd',
    dataset='train',
    x_cols=FEAT_S,
    y_cols=[Y_TRUE],
    # predict_only=True,
)


# ----------------------------------------------------------------------
# mbp_sim (simple model)
# ----------------------------------------------------------------------

mbp_sim = Model(
    estimator=EraSubsampler(LGBMRegressor(**params_0), n_subsamples=4),
    name='mbp_sim',
    dataset='full',
    x_cols=X_COLS,
    y_cols=[Y_TRUE],
    eras=np.arange(era_0, era_1),
    pass_eras=True,
    predict_only=True,
)


# ----------------------------------------------------------------------
# mbp_626 (choose parameters recommended in the example script)
# ----------------------------------------------------------------------

mbp_626 = Model(
    estimator=EraSubsampler(LGBMRegressor(**params_1), n_subsamples=4),
    name='mbp_626',
    dataset='full',
    x_cols=X_COLS,
    y_cols=[Y_TRUE],
    eras=np.arange(era_0, era_1),
    pass_eras=True,
    # predict_only=True,
)


# ----------------------------------------------------------------------
# mbp_mor (train on many targets)
# ----------------------------------------------------------------------

w = [0.5, 0.1, -0.1, 0, 0.15, 0, 0, 0, 0.35, 0]
y_inds = [0, 1, 2, 4, 8]
y_cols = [Y_COLS[i] for i in y_inds]
w = np.array([w[i] for i in y_inds])

m = LGBMRegressor(**params_0)
m = EraSubsampler(m, n_subsamples=4)
m = MultiOutputTrainer(m, weights=w)

mbp_mor = Model(
    estimator=m,
    name='mbp_mor',
    dataset='full',
    x_cols=X_COLS,
    y_cols=y_cols,
    eras=np.arange(era_0, era_1),
    pass_eras=True,
    predict_only=True,
)


# ----------------------------------------------------------------------
# mbp_eb (do era boosting)
# ----------------------------------------------------------------------

params_ebs = dict(params_0)
params_ebs['n_estimators'] = 200

model_ebs = LGBMRegressor(**params_ebs)
model_ebs = EraBooster(model_ebs, n_iters=10, percent_eras=0.5)
model_ebs = EraSubsampler(model_ebs, n_subsamples=4)

mbp_ebs = Model(
    estimator=model_ebs,
    name='mbp_ebs',
    dataset='full',
    x_cols=X_COLS,
    y_cols=[Y_TRUE],
    eras=np.arange(era_0, era_1),
    pass_eras=True,
    pass_eras_boost=True,
    predict_only=True,
)


# ----------------------------------------------------------------------
# mbp_er (do era regression)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# mbp_cr (use corr as an objective function)
# ----------------------------------------------------------------------

params_obj = dict(params_0)
params_obj['objective'] = objective_corr

mbp_obj = Model(
    estimator=EraSubsampler(LGBMRegressor(**params_obj), n_subsamples=4),
    name='mbp_obj',
    dataset='full',
    x_cols=X_COLS,
    y_cols=[Y_TRUE],
    eras=np.arange(era_0, era_1),
    pass_eras=True,
    # predict_only=True,
)

# ----------------------------------------------------------------------
# mbp_200 (train on last 200 eras)
# ----------------------------------------------------------------------

mbp_200 = Model(
    estimator=LGBMRegressor(**params_0),
    name='mbp_200',
    dataset='full',
    x_cols=X_COLS,
    y_cols=[Y_TRUE],
    eras=np.arange(era_1 - 200, era_1),
    # predict_only=True,
)


# ----------------------------------------------------------------------
# mbp_s (use every technique I know!)
# ----------------------------------------------------------------------


# ======================================================================
# train and submit
# ======================================================================

models = [
    # mbp_rnd,
    # mbp_sim,
    # mbp_626,
    # mbp_mor,
    # mbp_obj,
    # mbp_200,
    mbp_ebs
]

for model in models:
    model.train()
    model.submit()