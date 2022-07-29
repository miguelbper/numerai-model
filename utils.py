import numpy as np
import json
import re

with open('v4/FEATURES.json', 'r') as f:
    FEATURE_METADATA = json.load(f)
del f

FEATURES_L = list(FEATURE_METADATA['feature_stats'].keys())
FEATURES_M = FEATURE_METADATA['feature_sets']['medium']
FEATURES_S = FEATURE_METADATA['feature_sets']['small']
FEATURES_2 = FEATURE_METADATA['feature_sets']['v2_equivalent_features']
FEATURES_3 = FEATURE_METADATA['feature_sets']['v3_equivalent_features']
FEATURES_N = FEATURE_METADATA['feature_sets']['fncv3_features']

ERA = 'era'
DATA = 'data_type'
TARGET = 'target_nomi_v4_20'

FEATURES = FEATURES_L
N_FEATURES = len(FEATURES)
COLUMNS = [ERA, DATA] + FEATURES + [TARGET]

def string_from_class(c):
    s = str(c)
    s = re.sub(r'[^A-Za-z0-9.]+', '', s)
    s = s.split('.')[-1]
    return s

def era_subsample(e0, e1, i):
    return np.arange(e0 + i, e1, 4)

def feature_group(j):
    return [FEATURES[k] for k in range(j * 210, min((j + 1) * 210, N_FEATURES))]

def X_block(df, i, j):
    e0 = df['era'][0]
    e1 = df['era'][-1]
    return df[df.era.isin(era_subsample(e0, e1, i))][feature_group(j)].to_numpy()

def X_cols(df, j):
    return df[feature_group(j)].to_numpy()

def y_rows(df, i):
    e0 = df['era'][0]
    e1 = df['era'][-1]
    return df[df.era.isin(era_subsample(e0, e1, i))][TARGET].to_numpy()

def X_(df, eras = None):
    if eras == None:
        return df[FEATURES].to_numpy()
    else:
        e0 = df['era'][0]
        e1 = df['era'][-1]
        return df[df.era.isin(era_subsample(e0, e1, eras))][FEATURES].to_numpy()

def y_(df, eras = None):
    if eras == None:
        return df[TARGET].to_numpy()
    else:
        e0 = df['era'][0]
        e1 = df['era'][-1]
        return df[df.era.isin(era_subsample(e0, e1, eras))][TARGET].to_numpy()

def rank_pct(x):
    return x.rank(pct=True, method="first")

def blocks(df, i, j):
    e0 = df['era'][0]
    e1 = df['era'][-1]
    df = df[df.era.isin(era_subsample(e0, e1, i))]
    X = df[feature_group(j)].to_numpy()
    y = df[TARGET].to_numpy()
    return X, y

def average_of_arrays(arrays):
    return np.mean(np.array(arrays), axis=0)