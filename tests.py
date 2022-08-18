import joblib
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm
from utils import *

y_inds = [0, 1, 2, 4, 8]
y_cols = [Y_COLS[i] for i in y_inds]

print(y_cols)

w = np.array([0.6, 0.1, -0.25, 0, 0.2, 0, 0, 0, 0.35, 0])
w = np.array([w[i] for i in y_inds])
print(w)