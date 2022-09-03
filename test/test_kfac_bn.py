import torch
from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential, BatchNorm1d
from matplotlib import pyplot as plt

from backpack import backpack, extend
from backpack.extensions import KFAC, SqrtGGNExact
from backpack.utils.examples import load_one_batch_mnist

def visualize_hessian(H, param_names, param_length, fig_path, vmin=None, vmax=None):
    '''
    Args:
        H(torch.Tensor): Hessian matrix ([M, M])
        param_names(List[str]): list of param names
        param_length(List[int]): list of param lengths
        fig_path(str): path to save the figure

    Returns:
        H_min(float): min of H
        H_max(float): max of H
    '''
    plt.figure(figsize=(10,10))
    plt.imshow(H.cpu().numpy(), vmin=vmin, vmax=vmax, origin='upper')
    acc = -0.5
    all_ = H.shape[0]
    for name, l in zip(param_names, param_length):
        plt.plot([0-0.5, all_], [acc, acc], 'b-', linewidth=2)
        plt.plot([acc, acc], [0-0.5, all_], 'b-', linewidth=2)
        acc+= l
    plt.xlim([-0.5, all_-0.5])
    plt.ylim([all_-0.5, -0.5])
    plt.colorbar()
    plt.savefig(fig_path, bbox_inches='tight')
    return H.min(), H.max()

X, y = load_one_batch_mnist(batch_size=512)
model = Sequential(Flatten(), Linear(784, 3), BatchNorm1d(3), Linear(3, 10))
lossfunc = CrossEntropyLoss()
model = extend(model.eval())
lossfunc = extend(lossfunc)

loss = lossfunc(model(X), y)
with backpack(KFAC(mc_samples=1000), SqrtGGNExact()):
    loss.backward()

for name, param in model.named_parameters():
    GGN_VT = param.sqrt_ggn_exact.reshape(-1, param.numel())
    GGN = GGN_VT.t() @ GGN_VT
    KFAC_ = torch.kron(param.kfac[0], param.kfac[1]) if len(param.kfac) == 2 \
        else param.kfac[0]
    visualize_hessian(GGN, [name], [param.numel()], f"./{name}_GGN.png")
    visualize_hessian(KFAC_, [name], [param.numel()], f"./{name}_KFAC.png")
    print(name, torch.norm(GGN-KFAC_, 2).item())

# Check handeling the train mode situation
model = extend(model.train())
loss = lossfunc(model(X), y)
try:
    with backpack(KFAC(mc_samples=1000), SqrtGGNExact()):
        loss.backward()
except NotImplementedError:
    print("PASS. It raises NotImplementedError when model is in the training mode.")