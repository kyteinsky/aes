import pandas as pd
import config

data_ds = pd.loadcsv(dataset_dir + 'data_1.csv', usecols=list(range(1,16,1)))

data_ds.head()