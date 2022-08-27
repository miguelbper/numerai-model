from itertools import product
from lightgbm import LGBMRegressor
from utils import *

# ======================================================================
# LightGBM parameters + objective function
# ======================================================================

'''
parameters to keep constant:
    colsample_bytree: 0.1
    device: 'gpu'

parameters to decide:
    (n_estimators, learning_rate) <- {(2000, 0.01), (20000, 0.001)}
    (max_depth, num_leaves) <- {(5, 2**5), (6, 2**6)}
    objective <- {default, 
                  objective_corr (*1), 
                  objective_corr (*10), 
                  objective_corr (*100), 
                  objective_corr_ones}

rounds = 2 * 2 * 5 = 20

note: objective_corr (*100) means that we multiply the learning rate by 
100
'''

# options for CV
x_cols = FEAT_L
eras = np.arange(200, 1000, 4)
n_splits = 4

# dataset and CV splitter
X, y, e = read_Xye('full', x_cols, [Y_TRUE], eras)
spl = TimeSeriesSplitGroups(n_splits=n_splits)

# hyperparameters
estimators_rate_ls = [(100, 0.1), (1000, 0.01), (2000, 0.01), (10000, 0.001)] # [(2000, 0.01), (20000, 0.001)]
depth_leaves_ls = [(5, 2**5), (6, 2**6), (7, 2**7)] # [(5, 2**5), (6, 2**6)]
objective_ls = ['default'] # ['default', 'corr_0', 'corr_1', 'corr_2', 'corr_ones']
# estimators_rate_ls = [(200, 0.01)]
# depth_leaves_ls = [(5, 2**5)]
# objective_ls = ['corr_2', 'corr_ones', 'default']
all_params = list(product(estimators_rate_ls, depth_leaves_ls, objective_ls))
n_params = len(all_params)

# dict to save the scores
corr_dict = {
    'n_estimators': [],
    'learning_rate': [],
    'max_depth': [],
    'num_leaves': [],
    'objective': [],
}
for j in range(n_splits):
    corr_dict[f'corr_trn_{j}'] = []
for j in range(n_splits):
    corr_dict[f'corr_val_{j}'] = []

# CV loop
for i, ((ne, lr), (md, nl), ob) in enumerate(all_params):
    # define hyperparameters
    n_estimators = ne
    max_depth = md
    num_leaves = nl
    learning_rate = lr

    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'colsample_bytree': 0.1,
        'device': 'gpu',
    }

    # objective function
    if ob == 'corr_0': 
        params['objective'] = objective_corr
        ob = 'objective_corr'
    if ob == 'corr_1': 
        learning_rate = learning_rate * 10
        params['objective'] = objective_corr
        ob = 'objective_corr'
    if ob == 'corr_2': 
        learning_rate = learning_rate * 100
        params['objective'] = objective_corr
        ob = 'objective_corr'
    if ob == 'corr_ones': 
        params['objective'] = objective_corr_ones
        ob = 'objective_corr_ones'

    params['learning_rate'] = learning_rate        
    
    # fill dict
    corr_dict['n_estimators'].append(n_estimators)
    corr_dict['learning_rate'].append(learning_rate)
    corr_dict['max_depth'].append(max_depth)
    corr_dict['num_leaves'].append(num_leaves)
    corr_dict['objective'].append(ob)

    # define model
    model = LGBMRegressor(**params)
    
    # print doc
    s = [
        f'i = {i:2d}/{n_params - 1}',
        f'n_estimators = {n_estimators:5d}',
        f'learning_rate = {learning_rate:.3f}',
        f'max_depth = {max_depth}',
        f'num_leaves = {num_leaves}',
        f'obj = {ob}',
    ]
    print(', '.join(s))

    for j, (trn, val) in enumerate(spl.split(X, y, e)):
        print(f'\tj = {j}/3. ', end='')
        # define trn, val datasets
        X_trn = X.iloc[trn]
        X_val = X.iloc[val]
        y_trn = y.iloc[trn]
        y_val = y.iloc[val]
        e_trn = e.iloc[trn]
        e_val = e.iloc[val]

        # define an init_model (case of corr as objective)
        init_model=None
        if ob.startswith('objective_corr'):
            param_init = {
                'n_estimators': 1,
                'learning_rate': 0.01,
                'max_depth': 5,
                'num_leaves': 2**5,
                'colsample_bytree': 0.1,
                'device': 'gpu',
            }
            init_model = LGBMRegressor(**param_init)
            init_model.fit(X_trn, y_trn)

        # train model on trn
        print('Training... ', end='')
        model.fit(X_trn, y_trn, init_model=init_model)

        # predict using model on trn
        print('Predicting (trn)... ', end='')
        y_trn_pred = model.predict(X_trn)

        # compute score on trn
        print('Computing score (trn)... ', end='')
        c = corr(y_trn, y_trn_pred, rank_b=e_trn)
        corr_dict[f'corr_trn_{j}'].append(c)

        # predict using model on val
        print('Predicting (val)... ', end='')
        y_val_pred = model.predict(X_val)

        # compute score on val
        print('Computing score (val)... ')
        c = corr(y_val, y_val_pred, rank_b=e_val)
        corr_dict[f'corr_val_{j}'].append(c)

# save dict to file
corr_df = pd.DataFrame(corr_dict)
corr_df['val_mean'] = corr_df[[f'corr_val_{j}' for j in range(n_splits)]].mean(axis=1)
corr_df['trn_mean'] = corr_df[[f'corr_trn_{j}' for j in range(n_splits)]].mean(axis=1)
corr_df.to_excel('corr_df_2.xlsx')