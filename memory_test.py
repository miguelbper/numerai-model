import pandas as pd
import json
import time

with open('v4/features.json', 'r') as f:
    feature_metadata = json.load(f)

features_all = list(feature_metadata['feature_stats'].keys())
# features_small = feature_metadata['feature_sets']['small']
# features_medium = feature_metadata['feature_sets']['medium']
# features_v2 = feature_metadata['feature_sets']['v2_equivalent_features']
# features_v3 = feature_metadata['feature_sets']['v3_equivalent_features']
# features_fncv3 = feature_metadata['feature_sets']['fncv3_features']

features = features_all
target = 'target_nomi_v4_20'

read_columns = ['era', 'data_type'] + features + [target] 

def memory_test(df):
    print('Im a function which does nothing')

df = pd.read_parquet('v4/train_int8.parquet', columns=read_columns)

time.sleep(10)
print('slept 10')

memory_test(df)
memory_test(df)
memory_test(df)
memory_test('dickbutttt')

print('sleeping 100')
time.sleep(100)