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
        self.lin1 = nn.Linear(loop*16, 2048)
        self.norm_lin1 = nn.LayerNorm(2048)
        self.lin2 = nn.Linear(2048, 1024)
        self.norm_lin2 = nn.LayerNorm(1024)
        self.lin3 = nn.Linear(1024, 512)
        self.norm_lin3 = nn.LayerNorm(512)
        self.lin4 = nn.Linear(512, 128)

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

        # z = self.binarizer(z)

        return z


    def regr(self, x):
        x = self.swish(self.lin1(x))
        # x = self.norm_lin1(x)
        x = t.relu(self.lin2(x))
        x = self.norm_lin2(x)
        x = self.swish(self.lin3(x))
        # x = self.norm_lin3(x)
        x = self.lin4(x)
        return x


    def swish(self, x):
        return x * t.sigmoid(x)

    def binarizer(self, x):
        return 1/(1+(t.exp(-1000*x+500)))

# net = Model()
# inp = t.randn(2,1,4,4)
# print(inp)
# print(net(inp).shape)
