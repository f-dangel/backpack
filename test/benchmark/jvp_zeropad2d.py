from torch import randn
from torch.nn import ZeroPad2d

from backpack import extend


def data(device="cpu"):
    N, C, Hin, Win = 100, 10, 32, 32
    padding = [1, 2, 3, 4]
    Hout = Hin + padding[2] + padding[3]
    Wout = Win + padding[0] + padding[1]

    X = randn(N, C, Hin, Win, requires_grad=True, device=device)
    module = extend(ZeroPad2d(padding)).to(device=device)
    out = module(X)

    vout = randn(N, C, Hin, Win, device=device)
    vin = randn(N, C, Hout, Wout, device=device)

    return {
        "X": X,
        "module": module,
        "output": out,
        "vout_ag": vout,
        "vout_bp": vout.view(N, -1, 1),
        "vin_ag": vin,
        "vin_bp": vin.view(N, -1, 1),
    }
