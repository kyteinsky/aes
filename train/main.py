import sys
sys.path.append('../') # importing modules from parent dir

import pandas as pd
from config import *
from preprocess import *
import torch as t
from model import Model

## ----- DATALOADING ----- ##
# Loading all csv files into pandas -- no 'iv' here
data_ds = pd.read_csv(dataset_dir + 'data_1.csv', usecols=list(range(1,17,1)))
key_values_ds = pd.read_csv(dataset_dir + 'key_values_1.csv', usecols=list(range(1,17,1)))

# to torch tensors
x = t.tensor(key_values_ds.values, dtype=t.int32)
y = t.tensor(data_ds.values, dtype=t.int32)

dataset = t.utils.data.DataLoader(+)
## ------------------------ ##


# set the device to be used
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

model = Model() # get the model instance

