import joblib
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm
from utils import *
from sys import getsizeof

df = read_data('train', FEAT_L)
X = df[X_COLS]
y = df[Y_TRUE]
e = df[ERA]
del df

exposures = corr(X, y, rank_b=e)

print(f'exposures = {exposures}')

# X_gp = X.groupby(e)
# print(f'size(X_gp) = {getsizeof(X_gp) / 10**6} MB')
# X_rk = X_gp.apply(lambda x: x.rank())
# print(f'size(X_rk) = {getsizeof(X_rk) / 10**6} MB')
# print(f'size(X_rk) = {X_rk.memory_usage(deep=True)}')