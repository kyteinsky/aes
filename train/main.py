import sys
sys.path.insert(0,'.')

from config import *
from data_handler import *
import torch as t
from model import Model
from torch.utils.data import DataLoader, random_split
from time import time
from tqdm import trange
if wandb:
    import wandb
    wandb.init(project="sea-custom")


# from pytorch_model_summary import summary

# set the device to be used
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
# device = 'cpu'


## ----- DATALOADING ----- ##
train_ds = dataset(dataset_dir+'train_data_1.csv', dataset_dir+'train_enc_values_1.csv')
val_ds = dataset(dataset_dir+'val_data_1.csv', dataset_dir+'val_enc_values_1.csv')
# train_ds, val_ds = random_split(ds, [int(split*len(ds)), int(float('%.1f'%(1-split))*len(ds))], generator=t.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_ds, shuffle=True, batch_size=batch_size)
## ------------------------ ##


net = Model().to(device) # get the model instance
# print(net)
if wandb:
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
    for i, (x1, y1) in enumerate(train_loader):
        x1 = x1.to(device)
        y1 = y1.reshape(-1, 128).to(device)
        


        # forward pass
        outputs = net(x1)
        loss = criterion(outputs, y1)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % int(len(train_loader)/len(val_loader)) == 0:
            with t.no_grad():
                vx, vy = next(val_iter)
                vx, vy = vx.to(device)/255.0, vy.to(device)
                vx = net(vx)
                val_loss = criterion(vx.view(-1), vy.view(-1))

        if (i+1) % 1 == 0:
            # c = int((((epoch+1)*((i+1)+epoch*len(train_loader)))/((epochs+1)*(len(train_loader))))*20)
            print(f'Epoch [{epoch+1}/{epochs}], step [{i+1}/{len(train_loader)}], loss= [{loss.item():.4f}], val_loss= [{val_loss:.4f}]')#, end='\r')
            if wandb:
                wandb.log({"Loss": loss.item(), "Validation Loss": val_loss})
    #     break
    # break
# ['+'#'*(c)+' '*(20-c)+f'] == 

print()
print(f'Finished training in {time() - st} s')

value = input('Do you want to save model? (y/N): ')
if value == 'y' or value == 'Y': 
    t.save(net.state_dict(), './')
    print('model saved some where! safe now.')

if wandb:
    # Save model to wandb
    import os
    t.save(net.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

