from torch import randn
from torch.nn import Linear

from backpack import extend


def data_linear(device="cpu"):
    N, D1, D2 = 100, 64, 256

    X = randn(N, D1, requires_grad=True, device=device)
    linear = extend(Linear(D1, D2).to(device=device))
    out = linear(X)

    vin = randn(N, D2, device=device)
    vout = randn(N, D1, device=device)

    return {
        "X": X,
        "module": linear,
        "output": out,
        "vout_ag": vout,
        "vout_bp": vout.unsqueeze(2),
        "vin_ag": vin,
        "vin_bp": vin.unsqueeze(2),
    }
