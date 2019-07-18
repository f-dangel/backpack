from torch import randn
from torch.nn import Linear
from backpack import extend


def data():
    N, D1, D2 = 10, 10, 10

    X = randn(N, D1, requires_grad=True)
    linear = extend(Linear(D1, D2))
    out = linear(X)

    vin = randn(N, D2)
    vout = randn(N, D1)

    return {
        "X": X,
        "module": linear,
        "output": out,
        "vout_ag": vout,
        "vout_bp": vout.unsqueeze(2),
        "vin_ag": vin,
        "vin_bp": vin.unsqueeze(2),
    }
