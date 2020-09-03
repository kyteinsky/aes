import sys
sys.path.insert(0,'.')

from config import *
from data_handler import *
import torch as t
from model import Model
from torch.utils.data import DataLoader, random_split

# from pytorch_model_summary import summary

# set the device to be used
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
device = 'cpu'


## ----- DATALOADING ----- ##
ds = dataset(dataset_dir+'data_1.csv', dataset_dir+'enc_values_1.csv')

train_ds, val_ds = random_split(ds, [int(0.8*len(ds)), int(0.2*len(ds))], generator=t.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=1)
## ------------------------ ##


net = Model().to(device) # get the model instance
print(net)
# print()
# print(summary(net, t.zeros((1, 1, 16)), show_input=True))


criterion = t.nn.MSELoss().to(device)
optimizer = t.optim.Adam(net.parameters(), lr=lr)

## ------ TRAINING LOOP ------ ##
n_total_steps = len(train_loader)
val_iter =  iter(val_loader)
n_val_sets = len(val_loader)

flatten = t.nn.Flatten()
net.train()

for epoch in range(epochs):
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        for k in range(n_total_steps):
            # forward pass
            outputs = net(x[k])
            loss = criterion(outputs, y[k])
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            val_loss = t.tensor(0).to(device)
            with t.no_grad():
                if (i+1)*(k+1) % 4 == 0:
                    vx, vy = next(val_iter)
                    vx, vy = vx.to(device), vy.to(device)
                    val_loss = criterion(flatten(net(vx[k])), t.tensor(flatten(int_2_bin(vy[k])), dtype=t.float32))

            if (i+1) % 1 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], step [{i+1}/{n_total_steps}], loss= [{loss.item():.4f}], val_loss= [{val_loss:.4f}]')
        
print('Finished training')
