import torch as t
from torch.utils.data import DataLoader, random_split
from data_handler import *
from train.config import dataset_dir, split


## ----- DATALOADING ----- ##
ds = dataset(dataset_dir+'data_1.csv', dataset_dir+'enc_values_1.csv')

train_ds, val_ds = random_split(ds, [int(split*len(ds)), int(float('%.1f'%(1-split))*len(ds))], generator=t.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, shuffle=True)
val_loader = DataLoader(val_ds, shuffle=True)

for x, y in train_loader:
    print(x)
    break
