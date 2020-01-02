from functools import partial

from backpack.core.layers import Conv2dConcat, LinearConcat
from torch import randn, manual_seed
from backpack import extend
from torch.nn import AvgPool2d, Conv2d, Linear, MaxPool2d, ZeroPad2d
from torch.nn import Dropout, ReLU, Tanh, Sigmoid

N = 100
D, D1, D2 = 200, 64, 256
C, Cin, Cout = 10, 10, 25
Hin, Win = 32, 32
K = 4
padding = [1, 2, 3, 4]
Hout_pool = int(Hin / K)
Wout_pool = int(Win / K)
Hout_conv = Hin - (K - 1)
Wout_conv = Win - (K - 1)
Hout_pad = Hin + padding[2] + padding[3]
Wout_pad = Win + padding[0] + padding[1]


def __make_dict(X, mod, vin, vout):
    extend(mod)
    return {
        "X": X,
        "module": mod,
        "out": mod(X),
        "vin_ag": vin,
        "vin_bp": vin.reshape(N, -1, 1),
        "vout_ag": vout,
        "vout_bp": vout.reshape(N, -1, 1),
    }


def data_activations(module_class, seed=0, device="cpu"):
    manual_seed(seed)
    X = randn(N, D, requires_grad=True, device=device)
    module = module_class().to(device=device)
    v = randn(N, D, device=device)
    return __make_dict(X, module, v, v)


data_sigmoid = partial(data_activations, module_class=Sigmoid)
data_tanh = partial(data_activations, module_class=Tanh)
data_relu = partial(data_activations, module_class=ReLU)
data_dropout = partial(data_activations, module_class=Dropout)


def data_avgpool2d(seed=0, device="cpu"):
    manual_seed(seed)
    X = randn(N, C, Hin, Win, requires_grad=True, device=device)
    module = AvgPool2d(K).to(device=device)
    vout = randn(N, C, Hin, Win, device=device)
    vin = randn(N, C, Hout_pool, Wout_pool, device=device)
    return __make_dict(X, module, vin, vout)


def data_conv2d(seed=0, device="cpu"):
    manual_seed(seed)
    X = randn(N, Cin, Hin, Win, requires_grad=True, device=device)
    module = Conv2d(Cin, Cout, (K, K)).to(device=device)
    vin = randn(N, Cout, Hout_conv, Wout_conv, device=device)
    vout = randn(N, Cin, Hin, Win, device=device)
    return __make_dict(X, module, vin, vout)


def data_conv2dconcat(seed=0, device="cpu"):
    manual_seed(seed)
    X = randn(N, Cin, Hin, Win, requires_grad=True, device=device)
    module = Conv2dConcat(Cin, Cout, (K, K)).to(device=device)
    vin = randn(N, Cout, Hout_conv, Wout_conv, device=device)
    vout = randn(N, Cin, Hin, Win, device=device)
    return __make_dict(X, module, vin, vout)


def data_linear(seed=0, device="cpu"):
    manual_seed(seed)
    X = randn(N, D1, requires_grad=True, device=device)
    module = Linear(D1, D2).to(device=device)
    vin = randn(N, D2, device=device)
    vout = randn(N, D1, device=device)
    return __make_dict(X, module, vin, vout)


def data_linearconcat(seed=0, device="cpu"):
    manual_seed(seed)
    X = randn(N, D1, requires_grad=True, device=device)
    module = LinearConcat(D1, D2).to(device=device)
    vin = randn(N, D2, device=device)
    vout = randn(N, D1, device=device)
    return __make_dict(X, module, vin, vout)


def data_maxpool2d(seed=0, device="cpu"):
    manual_seed(seed)
    X = randn(N, C, Hin, Win, requires_grad=True, device=device)
    module = MaxPool2d(K).to(device=device)
    vout = randn(N, C, Hin, Win, device=device)
    vin = randn(N, C, Hout_pool, Wout_pool, device=device)
    return __make_dict(X, module, vin, vout)


def data_zeropad2d(seed=0, device="cpu"):
    manual_seed(seed)
    X = randn(N, C, Hin, Win, requires_grad=True, device=device)
    module = ZeroPad2d(padding).to(device=device)
    vout = randn(N, C, Hin, Win, device=device)
    vin = randn(N, C, Hout_pad, Wout_pad, device=device)
    return __make_dict(X, module, vin, vout)
