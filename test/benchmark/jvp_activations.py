from torch import randn
from backpack import extend


def data(module_class):
    N, D = 100, 200

    X = randn(N, D, requires_grad=True)
    module = extend(module_class())
    out = module(X)

    v = randn(N, D)

    return {
        "X": X,
        "module": module,
        "output": out,
        "vout_ag": v,
        "vout_bp": v.unsqueeze(2),
        "vin_ag": v,
        "vin_bp": v.unsqueeze(2),
    }
