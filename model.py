import torch.nn as nn
import torch as t
from train.config import loop

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.conv_ob = nn.Conv2d(1,loop,3,padding=1)
        # self.pool = nn.AdaptiveAvgPool3d((2,2))
        self.pool = nn.AvgPool2d(2)
        self.batch_norm = nn.BatchNorm2d(loop)
        self.sigmoid = nn.Sigmoid()
        self.lin1 = nn.Linear(loop*16, 1200)
        self.lin2 = nn.Linear(1200, 741)
        self.norm_lin = nn.LayerNorm(741)
        self.lin3 = nn.Linear(741, 240)
        self.lin4 = nn.Linear(240, 128)

    def forward(self, x):
        x = x.reshape(-1,1,4,4)

        # branch a
        a = self.conv_ob(x)
        a = self.swish(a)
        a = self.batch_norm(a)

        # branch b
        b = self.pool(x)
        b = b.reshape(-1, 1, 1, 4)

        # multiplication
        z = t.mul(a, b)
        z = z.reshape(-1, a[0].view(-1).shape[0])

        z = self.regr(z)

        return z


    def regr(self, x):
        x = self.swish(self.lin1(x))
        x = self.lin2(x)
        x = self.norm_lin(x)
        x = self.swish(self.lin3(x))
        x = self.lin4(x)
        return x


    def swish(self, x):
        return x * self.sigmoid(x)
