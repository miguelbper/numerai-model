from lightgbm import LGBMRegressor
from utils import *


# update_dataset('train')
# update_dataset('validation')
update_dataset('live')

# ======================================================================
# class NumeraiModel
# ======================================================================

class NumeraiModel():
    def __init__(self, estimator, name, dataset, x_cols, y_cols, eras=None,
                 pass_eras=False):
        self.estimator = estimator
        self.name = name
        self.dataset = dataset
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.eras = eras
        self.pass_eras = pass_eras

    def train(self):
        model_path = f'models/{self.name}.pkl'
        try:
            self.estimator = joblib.load(model_path)
            print(f'Loaded model {self.name}')
        except:
            print(f'Model {self.name} is not trained. Training... ', end='')
            X, y, e = read_Xye(self.dataset, self.x_cols, self.y_cols, 
                               self.eras)
            
            fit_params = {'eras': e} if self.pass_eras else dict()

            self.estimator.fit(X, y, **fit_params)
            joblib.dump(self.estimator, model_path)
            print('Done.')

    def predict(self):
        print(f'Predicting model {self.name}... ', end='')
        napi = NumerAPI()
        round = napi.get_current_round()
        pred_path = f'predictions/{self.name}_{round}.csv'

        df = read_df('live', self.x_cols, self.y_cols)
        df[Y_PRED] = self.estimator.predict(df[self.x_cols])
        df[Y_RANK] = df[Y_PRED].rank(pct=True)
        df[Y_RANK].to_csv(pred_path)
        print('Done.')

    def submit(self):
        pub, sec = joblib.load('keys.pkl')
        napi = NumerAPI(pub, sec)
        round = napi.get_current_round()
        pred_path = f'predictions/{self.name}_{round}.csv'

        model_id = napi.get_models()[self.name]
        napi.upload_predictions(pred_path, model_id=model_id)


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

mbp_rnd = NumeraiModel(
    estimator=RandomRegressor(),
    name='mbp_rnd',
    dataset='train',
    x_cols=FEAT_S,
    y_cols=[Y_TRUE],
)


# ----------------------------------------------------------------------
# mbp_sim (simple model)
# ----------------------------------------------------------------------

mbp_sim = NumeraiModel(
    estimator=EraSubsampler(LGBMRegressor(**params_0), n_subsamples=4),
    name='mbp_sim',
    dataset='full',
    x_cols=X_COLS,
    y_cols=[Y_TRUE],
    eras=np.arange(era_0, era_1),
    pass_eras=True,
)


# ----------------------------------------------------------------------
# mbp_626 (choose parameters recommended in the example script)
# ----------------------------------------------------------------------

mbp_626 = NumeraiModel(
    estimator=EraSubsampler(LGBMRegressor(**params_1), n_subsamples=4),
    name='mbp_626',
    dataset='full',
    x_cols=X_COLS,
    y_cols=[Y_TRUE],
    eras=np.arange(era_0, era_1),
    pass_eras=True,
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

mbp_mor = NumeraiModel(
    estimator=m,
    name='mbp_mor',
    dataset='full',
    x_cols=X_COLS,
    y_cols=y_cols,
    eras=np.arange(era_0, era_1),
    pass_eras=True,
)


# ----------------------------------------------------------------------
# mbp_erb (do era boosting)
# ----------------------------------------------------------------------

params_erb = dict(params_0)
params_erb['n_estimators'] = 200

model_erb = LGBMRegressor(**params_erb)
model_erb = EraBooster(model_erb, n_iters=10, era_fraction=0.5)
model_erb = EraSubsampler(model_erb, n_subsamples=4, pass_eras=True)

mbp_erb = NumeraiModel(
    estimator=model_erb,
    name='mbp_erb',
    dataset='full',
    x_cols=X_COLS,
    y_cols=[Y_TRUE],
    eras=np.arange(era_0, era_1),
    pass_eras=True,
)


# ----------------------------------------------------------------------
# mbp_er (do era regression)
# ----------------------------------------------------------------------


# # ----------------------------------------------------------------------
# # mbp_cr (use corr as an objective function)
# # ----------------------------------------------------------------------

# params_obj = dict(params_0)
# params_obj['objective'] = objective_corr

# mbp_obj = Model(
#     estimator=EraSubsampler(LGBMRegressor(**params_obj), n_subsamples=4),
#     name='mbp_obj',
#     dataset='full',
#     x_cols=X_COLS,
#     y_cols=[Y_TRUE],
#     eras=np.arange(era_0, era_1),
#     pass_eras=True,
# )

# ----------------------------------------------------------------------
# mbp_200 (train on last 200 eras)
# ----------------------------------------------------------------------

mbp_200 = NumeraiModel(
    estimator=LGBMRegressor(**params_0),
    name='mbp_200',
    dataset='full',
    x_cols=X_COLS,
    y_cols=[Y_TRUE],
    eras=np.arange(era_1 - 200, era_1),
)


# ----------------------------------------------------------------------
# mbp_s (use every technique I know!)
# ----------------------------------------------------------------------


# ======================================================================
# train and submit
# ======================================================================

models = [
    mbp_rnd,
    mbp_200,
    mbp_sim,
    mbp_626,
    mbp_mor,
    mbp_erb
]

for model in models:
    model.train()
    model.predict()
    model.submit()