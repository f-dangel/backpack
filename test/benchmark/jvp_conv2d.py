from torch import randn
from torch.nn import Conv2d

from backpack import extend


def data_conv2d(device="cpu"):
    N, Cin, Hin, Win = 100, 10, 32, 32
    Cout, KernelH, KernelW = 25, 5, 5

    X = randn(N, Cin, Hin, Win, requires_grad=True, device=device)
    module = extend(Conv2d(Cin, Cout, (KernelH, KernelW))).to(device=device)
    out = module(X)

    Hout = Hin - (KernelH - 1)
    Wout = Win - (KernelW - 1)
    vin = randn(N, Cout, Hout, Wout, device=device)
    vout = randn(N, Cin, Hin, Win, device=device)

    return {
        "X": X,
        "module": module,
        "output": out,
        "vout_ag": vout,
        "vout_bp": vout.view(N, -1, 1),
        "vin_ag": vin,
        "vin_bp": vin.view(N, -1, 1),
    }
