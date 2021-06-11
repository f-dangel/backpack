import torch
from torch import Tensor, allclose, autograd, rand
from torch.nn import BatchNorm1d

from backpack import extend
from backpack.core.derivatives.batchnorm1d import BatchNorm1dDerivatives


def test_jac_t_mat_prod_compare():
    V: int = 1
    N: int = 8
    C: int = 2
    # L: int = 10
    module: BatchNorm1d = BatchNorm1d(num_features=C)
    module = extend(module)
    _input: Tensor = rand(N, C)
    _input.requires_grad = True
    _out = module(_input)
    _grad_output: Tensor = rand(V, N, C)

    _derivatives: BatchNorm1dDerivatives = BatchNorm1dDerivatives()
    jac_t_mat_prod_old: Tensor = _derivatives._jac_t_mat_prod(
        module, None, None, _grad_output
    )
    jac_t_mat_prod_new: Tensor = _derivatives._jac_t_mat_prod_alternative(
        module, None, None, _grad_output
    )

    assert allclose(jac_t_mat_prod_old, jac_t_mat_prod_new, atol=1e-5)


def test_jac_t_mat_prod_autograd():
    has_l_axis = True
    V: int = 1
    N: int = 8
    C: int = 1
    L: int = 2
    module: BatchNorm1d = BatchNorm1d(num_features=C, momentum=1)
    module = extend(module)
    _input: Tensor = rand(N, C, L) if has_l_axis else rand(N, C)
    _input.requires_grad = True
    _out = module(_input)
    _grad_output: Tensor = rand(V, N, C, L) if has_l_axis else rand(V, N, C)

    _derivatives: BatchNorm1dDerivatives = BatchNorm1dDerivatives()
    jac_t_mat_prod: Tensor = _derivatives._jac_t_mat_prod_alternative(
        module, None, None, _grad_output
    )
    derivative_autograd: Tensor = torch.stack(
        autograd.grad(
            outputs=[_out],
            inputs=[_input],
            grad_outputs=[_grad_output[v] for v in range(V)],
        )
    )

    print("new jac_t_mat_prod", jac_t_mat_prod.shape)
    print("autograd", derivative_autograd.shape)
    assert allclose(jac_t_mat_prod, derivative_autograd, atol=1e-5)
    raise NotImplementedError("!!!!!!!!!!!!!!!!! WORKED !!!!!!!!!!!!!!!!!!!")


def test_jac_t_mat_prod_old_autograd():
    V: int = 1
    N: int = 8
    C: int = 2
    # L: int = 10
    module: BatchNorm1d = BatchNorm1d(num_features=C)
    module = extend(module)
    _input: Tensor = rand(N, C)
    _input.requires_grad = True
    _out = module(_input)
    _grad_output: Tensor = rand(V, N, C)

    _derivatives: BatchNorm1dDerivatives = BatchNorm1dDerivatives()
    jac_t_mat_prod: Tensor = _derivatives._jac_t_mat_prod(
        module, None, None, _grad_output
    )
    derivative_autograd: Tensor = torch.stack(
        autograd.grad(
            outputs=[_out],
            inputs=[_input],
            grad_outputs=[_grad_output[v] for v in range(V)],
        )
    )

    print("old jac_t_mat_prod", jac_t_mat_prod.shape)
    print("autograd", derivative_autograd.shape)
    assert allclose(jac_t_mat_prod, derivative_autograd, atol=1e-5)


def test_batch_norm_n_l_equivalence():
    V: int = 1
    N: int = 3
    C: int = 2
    L: int = 5
    module: BatchNorm1d = BatchNorm1d(num_features=C)
    _input0: Tensor = rand(N, C, L, requires_grad=True)
    _out0 = module(_input0)
    _input1: Tensor = _input0.reshape(N * L, C, 1)
    _out1 = module(_input1)

    assert allclose(_out0, _out1.reshape(N, C, L), atol=1e-5)


def test_batch_norm_forward():
    V: int = 1
    N: int = 3
    C: int = 2
    L: int = 5
    module: BatchNorm1d = BatchNorm1d(num_features=C)
    _input0: Tensor = rand(N, C, L, requires_grad=True)
    _out0 = module(_input0)
    mean = _input0.mean(dim=(0, 2))
    variance = _input0.var(dim=(0, 2), unbiased=False)
    out1: Tensor = (_input0 - mean[None, :, None])
    out1 = out1 / (variance[None, :, None] + module.eps).sqrt()
    out1 = out1 * module.weight[None, :, None]
    out1 = out1 + module.bias[None, :, None]

    assert allclose(_out0, out1)
