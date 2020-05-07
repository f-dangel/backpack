"""First order extensions.
===================================

First-order extensions make it easier to extract information from the gradients
being already backpropagated through the computational graph.
They do not backpropagate additional information, and have small overhead.
The implemented extensions are

- :func:`BatchGrad <backpack.extensions.BatchGrad>`
  The individual gradients, rather than the sum over the samples
- :func:`SumGradSquared <backpack.extensions.SumGradSquared>`
  The second moment of the individual gradient
- :func:`Variance <backpack.extensions.Variance>`
  The variance of the individual gradients
- :func:`BatchL2Grad <backpack.extensions.BatchL2Grad>`
  The L2 norm of the individual gradients



"""

from .batch_grad import BatchGrad
from .batch_l2_grad import BatchL2Grad
from .sum_grad_squared import SumGradSquared
from .variance import Variance

__all__ = ["BatchL2Grad", "BatchGrad", "SumGradSquared", "Variance"]
