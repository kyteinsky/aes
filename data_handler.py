import torch as t
import pandas as pd
from torch.utils.data import Dataset
from train.config import batch_size
from pandas.core.common import flatten


def toTensor(x):
    return t.tensor(x, dtype=t.float32)

def scale(x):
    return x/255.0

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

        data_df = pd.read_csv(self.data_csv, header=None, usecols=self.usecols)
        enc_df = pd.read_csv(self.enc_csv, header=None, usecols=self.usecols)

        self.x = scale(toTensor(enc_df.values))
        self.y = toTensor(int_2_bin(data_df.values))

        self.length = self.x.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index]
