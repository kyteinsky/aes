import torch.nn as nn
import torch as t
from train.config import loop

# def loss_fn(pred, true):
#     loss = t.tensor(0, dtype=t.float32, requires_grad=True)
#     pred = t.flatten(pred)
#     true = t.flatten(true)
#     for _,i in enumerate(pred):
#         loss = t.add(loss, 1) if true[_] != i else t.add(loss, 0)
#     return loss

class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()

        self.conv_ob = nn.Conv2d(1,1,3,padding=1)
        self.batch_norm = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(loop, 60)
        self.lin2 = nn.Linear(60, 16)
        self.lin3 = nn.Linear(16, 8)
        self.norm_lin = nn.BatchNorm1d(60)

    def forward(self, x):
        x = t.reshape(x, (1,1,4,4))
        
        # 'loop' times convolution stage
        x = self.rep(x, loop)
        
        # flatten and split
        x = self.flatten(x) # size of x is (100, 16)
        x = t.reshape(x, (16, 100))

        # linear stack
        x = map(self.regr, x)
        x = list(x)

        # x = list(map(self.bin_2_hex, x))

        y = t.empty((16,8))
        for _,i in enumerate(x):
            y[_] = i
        
        return y


    def regr(self, x):
        x = self.lin1(x)
        x = self.swish(x) # activation
        x.unsqueeze_(0)
        # x = self.norm_lin(x) # batch norm
        x = (x-0.5)/0.5
        x = self.lin2(x)
        x = self.swish(x) # activation
        x = self.lin3(x)
        x.squeeze_()
        # x = self.bin_activ(x)
        return x



    def rep(self, x, n):
        kk = t.tensor([])
        for i in range(n):
            # a branch
            a = self.conv_ob(x)

            # b branch
            b = self.conv_ob(x)
            b = self.swish(b)
            b = self.batch_norm(b)
            
            # geometric mean with sign
            x = self.sqrt(t.mul(a,b))

            kk = t.cat((kk, x))

        return kk # shape = (loop, 1, 4, 4)

    def sqrt(self, hh):
        gg = t.zeros(1,1,4,4)
        for i in range(hh.size()[-1]):
            for j in range(hh.size()[-2]):
                gg[0][0][j][i] = t.sqrt(abs(hh[0][0][j][i]))
                gg[0][0][j][i] = gg[0][0][j][i] * (-1 if hh[0][0][j][i] < 0 else 1)
        return gg

    def swish(self, x):
        return x * self.sigmoid(x)

    # def bin_activ(self, x):
    #     x = x.detach().numpy()
    #     return [1 if i > 0 else 0 for i in x]
    
    def bin_2_hex(self, x):
        return int(''.join(list(map(str, x))), 2)

# print(loss_fn())