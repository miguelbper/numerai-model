import joblib
import numpy as np
import pandas as pd
from pprint import pprint
# ar = np.random.rand(6*10**6, 9)
# print(f'ar = {ar}')
# print(f'size = {ar.nbytes / 10**9} GB')

ran_w1 = [x / 100 for x in range(0, 25, 5)] # np.arange(0.1, 0.2, 0.01)
ran_w2 = [x / 100 for x in range(-35, 5, 5)] # np.arange(0, -0.3, -0.05)
ran_w4 = [x / 100 for x in range(0, 25, 5)] # np.arange(0.1, 0.2, 0.01)
ran_w8 = [x / 100 for x in range(20, 45, 5)] # np.arange(0.3, 0.4, 0.01)

print(ran_w1)
print(ran_w2)
print(ran_w4)
print(ran_w8)