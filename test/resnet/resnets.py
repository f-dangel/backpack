"""Contains example from Katharina Ott."""

import torch
from torch.nn import MSELoss, Sequential, Tanh

from backpack import backpack, extend
from backpack.custom_module.branching import ActiveIdentity, Parallel
from backpack.custom_module.scale_module import ScaleModule
from backpack.extensions import KFAC, DiagGGNExact

# parameter
dt = 0.1
in_dim, hidden_dim = (2, 10)

# define net and loss function
lin1 = torch.nn.Linear(in_dim, hidden_dim)
lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
lin3 = torch.nn.Linear(hidden_dim, in_dim)
activation_function_1 = Tanh()
activation_function_2 = Tanh()
net = Sequential(lin1, activation_function_1, lin2, activation_function_2, lin3)
net = extend(net)
loss_function = extend(MSELoss())

# define input and solution
x = torch.tensor([[1.0, 2.0]], requires_grad=True)
solution = torch.tensor([[1.0, 1.0]])

# version from Katharina
logits = x + net(x) * dt
loss = loss_function(logits, solution)

print("\nAlternative network.")
# the BackPACK equivalent
net_euler = Parallel(
    Sequential(net, ScaleModule(weight=dt)),
    ActiveIdentity(),
)
net_euler = extend(net_euler)
logits_alt = net_euler(x)
loss_alt = loss_function(logits_alt, solution)

print("Do the logits match?", torch.allclose(logits, logits_alt))
print("Do the losses match?", torch.allclose(loss, loss_alt))

with backpack(KFAC(), DiagGGNExact()):
    loss_alt.backward()
for name, param in net.named_parameters():
    print(name)
    print(param.grad)
    print(param.kfac)
    print(param.diag_ggn_exact)
