# alternative way of implementing FeatureNeutralizer:
# instead of giving groups as separate argument, give them inside X
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
    
    # this function is only meant to be used by predict
    def compute_y_pred(self, X):
        # checks
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        # computations
        self.y_pred = self.estimator.predict(X)

    # this function is only meant to be used by predict
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



# ----------------------------------------------------------------------
# feature neutralization
# ----------------------------------------------------------------------

ran_nf = range(5, 55, 5)
ran_al = [x / 100 for x in range(0, 105, 5)]

cv_neu = {
    'n_features': [],
    'alpha': [],
}
for j in range(n_splits):
    cv_neu[f'split_{j + 1}'] = []

for j, (trn, val) in tqdm(enumerate(spl.split(X, y, e)), 
                        desc='cv split', 
                        total=n_splits,
                        leave=None,
                        position=0):
    X_val = X.iloc[val]
    y_val = Y.iloc[val]
    e_val = e.iloc[val]
    y_val_true = y_val[Y_TRUE]
    y_val_pred = joblib.load(f'model-0/saved-variables/y_targets_{j}.pkl')
    y_val_pred_w = y_val_pred @ w
    
    exposures = corr(X_val, y_val_pred_w, rank_a=e_val, rank_b=e_val)
    riskiest = [(v, i) for i, v in enumerate(exposures)]
    riskiest = sorted(riskiest, reverse=True)
    riskiest = [i for _, i in riskiest]

    for nf in tqdm(ran_nf, desc='n_features', leave=False, position=1):
        riskiest_n = riskiest[0:nf]
        riskiest_n = [x_cols[i] for i in riskiest_n]

        def aux_linreg(df):
            X_ = df[df.columns[0:-1]]
            y_ = df[df.columns[-1]]
            model = LinearRegression()
            model.fit(X_, y_)
            return pd.Series(model.predict(X_) - model.intercept_)

        R = X_val[riskiest_n].to_numpy()
        df_Ry = pd.DataFrame(np.hstack((R, np.atleast_2d(y_val_pred_w).T)))
        y_linr = df_Ry.groupby(e_val.to_numpy()).apply(aux_linreg).to_numpy()

        for al in tqdm(ran_al, desc='alpha', leave=False, position=2):
            y_neut = y_val_pred_w - al * y_linr
            c = corr(y_val_true, y_neut, rank_b=e_val)

            cv_neu[f'split_{j + 1}'].append(c)
            if j == 0:
                cv_neu['n_features'].append(nf)
                cv_neu['alpha'].append(al)
            

cv_neu = pd.DataFrame(cv_neu)
cv_neu['mean'] = cv_neu[[f'split_{j+1}' for j in range(n_splits)]].mean(axis=1)
cv_neu.to_excel('model-0/cross-validation/cv_neu.xlsx')

ar_max = cv_neu['mean'].argmax()
n_features = cv_neu['n_features'][ar_max]
alpha = cv_neu['alpha'][ar_max]
print(f'n_features = {n_features}')
print(f'alpha = {alpha}')
# n_features = 30, alpha = 0.3