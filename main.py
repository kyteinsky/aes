import pandas as pd
from config import *

data_ds = pd.read_csv(dataset_dir + 'data_1.csv', usecols=list(range(1,17,1)))

print(data_ds.head())
