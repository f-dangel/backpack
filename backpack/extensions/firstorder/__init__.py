"""
First order backPACK extensions.

Those extension do not backpropagate additional information, and their
computational overhead is small.

They make it easier to extract more information from the gradient being already
backpropagated through the computational graph, such as
- `BatchGrad`: The individual gradients, rather than the sum over the samples
- `SumGradSquared`: The second moment of the individual gradient
- `Variance`: The variance of the individual gradients
- `BatchL2Grad`: The L2 norm of the individual gradients
"""

from .batch_l2_grad import BatchL2Grad
from .batch_grad import BatchGrad
from .sum_grad_squared import SumGradSquared
from .variance import Variance


