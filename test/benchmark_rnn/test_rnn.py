"""Benchmarks rnn."""
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from backpack import extend
from backpack.core.derivatives.rnn import RNNDerivatives

N = 128  # batch size
seq_len = 100
input_size = 50
hidden_size = 30
inputs = torch.rand(seq_len, N, input_size)
inputs.requires_grad = True
grad_outputs = torch.rand(seq_len, N, hidden_size)


def _setup_rnn_autograd():
    _setup_rnn_autograd.rnn = torch.nn.RNN(input_size, hidden_size)
    _setup_rnn_autograd.outputs, _ = _setup_rnn_autograd.rnn(inputs)


def _make_rnn_autograd():
    return torch.autograd.grad(
        [_setup_rnn_autograd.outputs],
        [_setup_rnn_autograd.rnn.bias_ih_l0],
        grad_outputs=[grad_outputs],
    )[0]


def _setup_rnn_backpack():
    _setup_rnn_backpack.derivatives = RNNDerivatives()
    _setup_rnn_backpack.rnn = extend(torch.nn.RNN(input_size, hidden_size))
    outputs, _ = _setup_rnn_backpack.rnn(inputs)


def _make_rnn_backpack():
    return _setup_rnn_backpack.derivatives.bias_ih_l0_jac_t_mat_prod(
        _setup_rnn_backpack.rnn, None, None, grad_outputs, sum_batch=True
    )


def test_rnn_autograd(benchmark: BenchmarkFixture):
    """Tests the speed of the derivatives of autograd.

    Args:
        benchmark: benchmark
    """
    benchmark.pedantic(_make_rnn_autograd, setup=_setup_rnn_autograd, rounds=10)


def test_rnn_backpack(benchmark: BenchmarkFixture):
    """Tests the speed of the derivatives of backpack.

    Args:
        benchmark: benchmark
    """
    benchmark.pedantic(_make_rnn_backpack, setup=_setup_rnn_backpack, rounds=10)
