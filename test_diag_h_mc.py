from torch import manual_seed, rand, randn
from torch.nn import Linear, MSELoss, Sequential, Sigmoid

import backpack

manual_seed(0)

N = 10
D_in = 4
D_hidden = 3
D_out = 2

X = randn(N, D_in, requires_grad=True)  # to force backpropagate call
y = randn(N, D_out)

model = backpack.extend(
    Sequential(
        Linear(D_in, D_hidden),
        Sigmoid(),
        Linear(D_hidden, D_hidden),
        Sigmoid(),
        Linear(D_hidden, D_out),
    )
)
for p in model.parameters():
    p.data = rand(p.shape)

loss_fn = backpack.extend(MSELoss(reduction="mean"))

with backpack.backpack(backpack.extensions.DiagHessian()):
    loss = loss_fn(model(X), y)
    loss.backward()

diag_h = [p.diag_h for p in model.parameters()]
print(diag_h)

mc = "Normal"
mc = "Bernoulli"

with backpack.backpack(
    backpack.extensions.secondorder.DiagHessianMC(mc_samples=100000, mc=mc)
):
    loss = loss_fn(model(X), y)
    loss.backward()

diag_h_mc = [p.diag_h_mc for p in model.parameters()]
print(diag_h_mc)

num_params = sum(p.numel() for p in model.parameters())

# print(sum((h1 - h2).abs().sum() for h1, h2 in zip(diag_h, diag_h_mc)) / num_params)

tr_h = sum(d.sum() for d in diag_h)
tr_h_mc = sum(d.sum() for d in diag_h_mc)
# print(abs(tr_h - tr_h_mc) / num_params)

print(tr_h)
print(tr_h_mc)
print((1 - abs(tr_h - tr_h_mc) / tr_h) * 100, "%")
print(
    "Squared error:", sum(((d1 - d2) ** 2).sum() for d1, d2 in zip(diag_h_mc, diag_h))
)
