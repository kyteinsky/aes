import sys
sys.path.insert(0,'.')

import pandas as pd
from config import *
from data_handler import *
import torch as t
from model import Model, loss_fn
from data_handler import *

'''
## ----- DATALOADING ----- ##
# Loading all csv files into pandas -- no 'iv' here
data_ds = pd.read_csv(dataset_dir + 'data_1.csv', usecols=list(range(1,17,1)))
key_values_ds = pd.read_csv(dataset_dir + 'key_values_1.csv', usecols=list(range(1,17,1)))

# to torch tensors
x = t.tensor(key_values_ds.values, dtype=t.int32)
y = t.tensor(data_ds.values, dtype=t.int32)

# dataset = t.utils.data.DataLoader()
## ------------------------ ##
'''

# set the device to be used
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

net = Model()#.to(device) # get the model instance
# print(net)

input = t.tensor([[[
    [23,124,66,94],
    [244,62,85,188],
    [45,173,48,87],
    [90,81,211,64]
]]], dtype=t.float32)
print(input)
out = net(input)
print(out)

loss = loss_fn(out, 
        t.tensor([[0, 1, 1, 1, 0, 1, 1, 0], 
                    [0, 1, 0, 0, 1, 0, 1, 1], 
                    [0, 0, 1, 1, 0, 0, 0, 0], 
                    [0, 1, 0, 0, 0, 1, 1, 0], 
                    [0, 1, 1, 1, 1, 1, 0, 1], 
                    [0, 0, 1, 0, 0, 1, 0, 1], 
                    [0, 0, 1, 1, 1, 1, 0, 1], 
                    [0, 1, 1, 0, 1, 1, 1, 0], 
                    [0, 1, 1, 1, 1, 1, 0, 0], 
                    [1, 1, 1, 1, 0, 1, 1, 0], 
                    [0, 0, 1, 0, 1, 0, 0, 1], 
                    [0, 1, 1, 1, 1, 1, 0, 0], 
                    [0, 1, 0, 0, 0, 1, 0, 1], 
                    [0, 1, 0, 1, 1, 0, 0, 1], 
                    [0, 1, 0, 1, 0, 0, 1, 1], 
                    [0, 1, 0, 0, 0, 0, 0, 0]], requires_grad=True))
                    

print('loss =',loss)
t.nn.MSELoss()
