from torchviz import make_dot
from model import Model
import torch as t

x = t.randn(1,16)

model = Model()
out = model(x)
# print(list(model.named_parameters()))
make_dot(out, params=dict(model.named_parameters()))
