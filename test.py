# from torchviz import make_dot
# from model import Model
import torch as t
from data_handler import dataset
from train.config import dataset_dir, batch_size
from torch.utils.data import random_split, DataLoader

# x = t.randn(1,16)

# model = Model()
# out = model(x)
# # print(list(model.named_parameters()))
# make_dot(out, params=dict(model.named_parameters()))

## ----- DATALOADING ----- ##
ds = dataset(dataset_dir+'data_1.csv', dataset_dir+'enc_values_1.csv')

train_ds, val_ds = random_split(ds, [int(0.8*len(ds)), int(0.2*len(ds))], generator=t.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=1)
## ------------------------ ##


# def int_2_bin(xy):
#     li = []
#     for x in xy:
#         x = ['{:08b}'.format(int(i)) for i in x]
#         li.append([list(map(int, list(i))) for i in x])
#     return li

# def binary(x, bits):
#     mask = 2**t.arange(bits).to(x.device, x.dtype)
#     return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


# val_iter = iter(val_loader)
for i, (x, y) in enumerate(train_loader):
    # vx, vy = next(val_iter)
    # print(vy[i])
    # print(int_2_bin(vy[i]))
    # print(int(x[0][0]*255))
    # print(binary(t.tensor(x.clone().detach()[0][0]*255, dtype=t.int32), 8))
    
    print(x)
    break
