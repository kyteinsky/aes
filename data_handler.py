import torch as t
import pandas as pd
from torch.utils.data import Dataset
from train.config import batch_size
from pandas.core.common import flatten


def int_2_bin(xy):
    li = []
    for x in xy:
        x = ['{:08b}'.format(int(i)) for i in x]
        li.append([list(map(int, list(i))) for i in x])
    return li

class dataset(Dataset):

    def __init__(self, data_csv, enc_csv):
        self.usecols = list(range(17))[1:]
        # self.usecols = list(range(16))
        self.enc_csv = enc_csv
        self.data_csv = data_csv

        len_enc = sum(1 for line in open(self.enc_csv))
        self.l = list(range(len_enc))
        batch_list = [self.l[i:i + batch_size] for i in range(0, len(self.l), batch_size)]
        self.length = len(batch_list)

        # self.enc_df = iter(pd.read_csv(enc_csv, usecols=usecols, chunksize=batch_size))
        # self.data_df = iter(pd.read_csv(data_csv, usecols=usecols, chunksize=batch_size))

        # self.x = t.tensor(enc_df.values, dtype=t.float32)/255.0
        # self.y = t.tensor(int_2_bin(data_df.values), dtype=t.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        batch_list = [self.l[i:i + batch_size] for i in range(0, len(self.l), batch_size)]
        batch_list.remove(batch_list[index])
        
        data_df = pd.read_csv(self.data_csv, usecols=self.usecols, nrows=batch_size, skiprows=list(flatten(batch_list)), header=None)
        enc_df = pd.read_csv(self.enc_csv, usecols=self.usecols, nrows=batch_size, skiprows=list(flatten(batch_list)), header=None)

        x = t.tensor(enc_df.values, dtype=t.float32)/255.0
        y = t.tensor(int_2_bin(data_df.values), dtype=t.float32)

        return x, y




# from textwrap import wrap

# def hex_2_bin(x):
#     return str(bin(int(str(x), 16)))[2:]

# class Dataset(torch.utils.data.Dataset):

#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def __len__(self):
#         return len(self.list_IDs)

#     def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#         ID = self.list_IDs[index]

#         # Load data and get label
#         X = torch.load('data/' + ID + '.pt')
#         y = self.labels[ID]

#         return X, y

### --- OLD --- ### ---------------

# def int_2_bin(xy):
#     li = []
#     for x in xy:
#         x = ['{:08b}'.format(int(i)) for i in x]
#         li.append([list(map(int, list(i))) for i in x])
#     return li
### -------------------------------------
