import joblib
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm
from utils import *
from sys import getsizeof

pth = 5

x = ('2PAF5GDR53PRF4QNEMJHLFU7NMRZGH3B', 'I35VQMGMTTIRWTEGBEUU2BWW6PBLUHKFU76HYPBKHHAQYDPV52QXPCI65N3DCX74')
joblib.dump(x, 'keys.pkl')