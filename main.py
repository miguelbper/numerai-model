import joblib
from utils import *

class Model():
    def __init__(self, estimator, name, dataset, x_cols, y_cols, eras=None):
        self.estimator = estimator
        self.name = name
        self.dataset = dataset
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.eras = eras

    def train(self):
        model_path = f'models/{self.name}.pkl'
        try:
            self.estimator = joblib.load(model_path)
            print(f'Loaded model {self.name}')
        except:
            print(f"Couldn't load model {self.name}. Training model")
            X, y = read_Xy(self.dataset, self.x_cols, self.y_cols, self.eras)
            self.estimator.fit(X, y)
            joblib.dump(self.estimator, model_path)

    def submit(self):
        pub, sec = joblib.load('keys.pkl')
        napi = NumerAPI(pub, sec)
        round = napi.get_current_round()
        pred_path = f'predictions/{self.name}/liv_{round}.csv'

        df = read_df(self.dataset, self.x_cols, self.y_cols, self.eras)
        df[Y_PRED] = self.estimator.predict(df[self.x_cols])
        df[Y_RANK] = df[Y_PRED].rank(pct=True)
        df[Y_RANK].to_csv(pred_path)

        model_id = napi.get_models()[self.name]
        napi.upload_predictions(pred_path, model_id=model_id)
        print(f'Submited predictions for model {self.name}')

models = []

for model in models:
    model.train()
    model.submit()
