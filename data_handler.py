import torch as t
import pandas as pd
from torch.utils.data import Dataset


def int_2_bin(xy):
    li = []
    for x in xy:
        x = ['{:08b}'.format(int(i)) for i in x]
        li.append([list(map(int, list(i))) for i in x])
    return li

class dataset(Dataset):

    def __init__(self, data_csv, enc_csv):
        usecols = list(range(17))[1:]
        enc_df = pd.read_csv(enc_csv, usecols=usecols)
        data_df = pd.read_csv(data_csv, usecols=usecols)

        self.x = t.tensor(enc_df.values, dtype=t.float32)/255.0
        self.y = t.tensor(int_2_bin(data_df.values), dtype=t.float32)

        self.length = len(data_df)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index]





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
