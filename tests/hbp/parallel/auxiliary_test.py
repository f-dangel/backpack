"""
Parameter splitting requires some nasty modifications with pointers
to chunks of tensors and gradient computation for these chunks.
This file is used by the author to explore gradient computation in
PyTorch.
"""

from torch.nn import Module
from torch.nn.parameter import Parameter
from torch import cat, tensor, allclose


class Layer(Module):
    """Small testing module to understand computation of gradients
    when using concatenated tensors."""

    def __init__(self):
        super().__init__()
        # weight matrix rows
        w1 = Parameter(tensor([[5., 4.]]))
        self.w1 = w1
        w2 = Parameter(tensor([[2., 1]]))
        self.w2 = w2
        # concatenated weight matrix
        w = cat([w1, w2], 0)
        self.w = w

    def forward(self, input):
        """Matric multiplication by w."""
        return self.w.matmul(input)


def test_understanding_grad_computation():
    """Test understanding of gradient computation."""
    layer = Layer()
    x = tensor([1., 2.])
    y = layer(x)
    loss = y.sum()

    assert allclose(y, tensor([13., 4.]))
    assert allclose(loss, tensor([17.]))

    loss.backward()

    assert len(list(layer.parameters())) == 2
    for name, param in layer.named_parameters():
        print(name)
        print(param)
        print(param.grad)
        assert param.grad is not None
        print(param.storage().data_ptr())

    assert layer.w.grad is None
