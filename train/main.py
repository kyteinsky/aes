import sys
sys.path.insert(0,'.')

from config import *
from data_handler import *
import torch as t
from model import Model
from torch.utils.data import DataLoader, random_split
from time import time
import wandb
wandb.init(project="sea")


# from pytorch_model_summary import summary

# set the device to be used
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
# device = 'cpu'


## ----- DATALOADING ----- ##
ds = dataset(dataset_dir+'data_1.csv', dataset_dir+'enc_values_1.csv')

train_ds, val_ds = random_split(ds, [int(split*len(ds)), int(float('%.1f'%(1-split))*len(ds))], generator=t.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=1)
## ------------------------ ##


net = Model().to(device) # get the model instance
# print(net)
wandb.watch(net)
# print()
# print(summary(net, t.zeros((1, 1, 16)), show_input=True))


criterion = t.nn.L1Loss().to(device)
optimizer = t.optim.Adam(net.parameters(), lr=lr)

val_loss = t.tensor(0).to(device)
net.train()

## -- had mistakenly ignored passing batches, instead was passing in single values, now all good!
st = time()
## ------ TRAINING LOOP ------ ## 
for epoch in range(epochs):
    val_iter =  iter(val_loader)
    for i, (x, y) in enumerate(train_loader):
        with t.no_grad():
            global x1, y1
            x1 = x.to(device)
            y1 = y.reshape(-1, 128).to(device)
        

        # forward pass
        outputs = net(x1)
        loss = criterion(outputs, y1)
        loss = loss * t.exp(loss)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % int(len(train_loader)/len(val_loader)) == 0:
            with t.no_grad():
                vx, vy = next(val_iter)
                # print(type(vy))
                vx, vy = vx.to(device)/255.0, vy.to(device)
                vx = net(vx)
                val_loss = criterion(vx.view(-1), vy.view(-1))
                val_loss = val_loss * t.exp(val_loss)

        if (i+1) % 1 == 0:
            # c = int((((epoch+1)*((i+1)+epoch*len(train_loader)))/((epochs+1)*(len(train_loader))))*20)
            print('Epoch [{epoch+1}/{epochs}], step [{i+1}/{len(train_loader)}], loss= [{loss.item():.4f}], val_loss= [{val_loss:.4f}]', end='\r')
            wandb.log({"Loss": loss.item(), "Validation Loss": val_loss})
# ['+'#'*(c)+' '*(20-c)+f'] == 

print()
print(f'Finished training in {time() - st} s')


# Save model to wandb
import os
t.save(net.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
