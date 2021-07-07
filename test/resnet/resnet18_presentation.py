"""BackPACK extension of resnet18.

2021-07-08: Code presented to Phillipp.
"""
from torch import rand, rand_like
from torchvision.models import resnet18
from torch.nn import MSELoss

from backpack import backpack, extend
from backpack.extensions import DiagGGNExact

resnet18 = extend(resnet18(num_classes=100).eval(), use_converter=True)

x = rand(8, 3, 224, 224, requires_grad=True)
y_predict = resnet18(x)

loss = extend(MSELoss())(y_predict, rand_like(y_predict))

with backpack(DiagGGNExact()):
    loss.backward()
for name, param in resnet18.named_parameters():
    print(name)
    print(param.grad.shape)
    print(param.diag_ggn_exact.shape)
