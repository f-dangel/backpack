from torch import randn

from backpack import extend


def data(module_class, device="cpu"):
    N, D = 100, 200

    X = randn(N, D, requires_grad=True, device=device)
    module = extend(module_class()).to(device=device)
    out = module(X)

    v = randn(N, D, device=device)

    return {
        "X": X,
        "module": module,
        "output": out,
        "vout_ag": v,
        "vout_bp": v.unsqueeze(2),
        "vin_ag": v,
        "vin_bp": v.unsqueeze(2),
    }
