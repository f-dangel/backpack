import torch
from torch.nn import Linear, ReLU, Conv2d, CrossEntropyLoss, Sequential
from bpexts.gradient.layers import Flatten
from bpexts.gradient import bpexts, extend as xtd

import bpexts.gradient.extensions as ext

N, C, H, W = 2, 3, 4, 4
D = C * H * W

X = torch.randn(N, C, H, W)
Y = torch.randint(high=2, size=(N,))

model = Sequential(xtd(Conv2d(3, 2, 2)), xtd(ReLU()), Flatten(), xtd(Linear(18, 2)))
lossfunc = xtd(CrossEntropyLoss())

loss = lossfunc(model(X), Y)

with bpexts(ext.DIAG_GGN):
    loss.backward()
