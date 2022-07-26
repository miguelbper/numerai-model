import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from itertools import product

file = 'train_int8'

feature_metadata = json.load(open('v4/features.json', 'r'))
features = list(feature_metadata['feature_stats'].keys())
target = 'target_nomi_v4_20'
cols = ['era', 'data_type'] + features + [target] 

df = pd.read_parquet(f'v4/{file}.parquet', columns=cols)
df['era'] = df['era'].astype('int32')

first_era = df['era'][0]
last_era = df['era'][-1]
num_features = len(features)

i = 1
for e, f in product(range(4), range(6)):
    eras_subset = np.arange(first_era + e, last_era, 4)
    features_subset = [features[j] for j in range(f * 210, min((f + 1) * 210, num_features))]
    
    df_write = df[df.era.isin(eras_subset)]
    X = df_write[features_subset]
    y = df_write[target].to_frame()
    
    X.to_parquet(f'blocks/{file}_X_{e}{f}.parquet')
    y.to_parquet(f'blocks/{file}_y_{e}{f}.parquet')
    
    print(f'{i} / 24it. e = {e}, f = {f}, X.shape = {X.shape}, y.shape = {y.shape}')
    i += 1