Good to know
====================================


We try to make BackPACK easy to use,
but some conventions differ from standard PyTorch and can be a bit obscure.
Here are some common pitfalls and recommendations.


Check that BackPACK does what you think it should do
-----------------------------------------------------

Most of the quantities provided by BackPACK can also be computed with the
standard automatic differentiation in PyTorch, although not efficiently.
For example, the individual gradients given by
:py:class:`BatchGrad <backpack.extensions.BatchGrad>`
can be computed by doing a ``for loop`` over each example, and doing a forward
and a backward pass over each sample individually.
This is slow, but can be used to check that the values returned by BackPACK
match what you expect them to be.

While we test many a use-case and try to write solid code, unexpected
behavior (such as some listed on this page) or bugs are not impossible.
We recommend that you check that the outputs match your expectations,
especially if you're using non-default values on slightly more unusual parameters
like ``groups`` for convolutions or pooling operators.
BackPACK will try to raise warnings if you use modules or parameters
in obvious unsupported ways, but it can't anticipate everything.


Aggregating quantities and :py:meth:`zero_grad() <torch.nn.Module.zero_grad>`
-----------------------------------------------------------------------------

When computing multiple gradients, one after the other, PyTorch accumulates
all the gradients seen so far in the :py:attr:`.grad <torch.Tensor.grad>` field
by summing the result of each :py:meth:`.backward() <torch.Tensor.backward>` call.
To compute a new gradient, it is standard practice to call
:py:meth:`zero_grad() <torch.nn.Module.zero_grad>`
on the model or optimizer to reset the accumulation.
This means that to compute the gradient of a sum, you can compute one element
of the sum, do a :py:meth:`.backward() <torch.Tensor.backward>`,
and iterate through the next one, and :py:attr:`.grad <torch.Tensor.grad>`
will hold the sum of the gradient (which is also the gradient of the sum).

Because the quantities returned by BackPACK are not necessarily additive
(the variance over two batches is not the sum of the variance on each batch),
BackPACK does not aggregate like PyTorch.
Every call to :py:meth:`.backward() <torch.Tensor.backward>`,
inside a :py:func:`with backpack(...): <backpack.backpack>`,
reset the corresponding field, and the fields returned by BackPACK
are not affected by :py:meth:`zero_grad() <torch.nn.Module.zero_grad>`.

:py:func:`extend <backpack.extend()>`-ing for first and second-order extension
------------------------------------------------------------------------------------------------

The :ref:`intro example <How to use BackPACK>` shows how to make a model
using a :py:class:`torch.nn.Sequential` module
and how to :py:func:`extend() <backpack.extend>` the model and the loss function,
but this setup is only really necessary for
:ref:`second order quantities <Second order extensions>`.
For those, BackPACK needs to know about the structure of the whole network
to propagate additional information.

:ref:`First order extensions <First order extensions>` are more flexible,
and the only :py:class:`torch.nn.Module` that need to be extended
are modules with parameters, to extract more information,
as the gradients are already propagated by PyTorch.
For every operations that is not parametrized, you can use standard operations
from the :std:doc:`torch.nn.functional <nn.functional>` module or standard
tensor operations. This makes it possible to use first order extensions
for ResNets (see :ref:`this example <First order extensions with a ResNet>`).


Not (yet) supported models
----------------------------------

The second-order extensions for BackPACK don't support (yet) residual networks,
and no extension support recurrent architectures.
We're working on how to handle those, as well as adding more
:ref:`layers <Supported models>`.
Along those lines, some things that will (most likely) not work with BackPACK,
but that we're trying to build support for:

- Inplace operations (e.g., using ``inplace=True`` for activation functions like
  :py:class:`torch.nn.ReLU`.
- Reusing the same parameters or module multiple time in the computation graph.

  For second order extensions, this also holds for any module,
  whether or not they have parameters.
  This sadly mean that BackPACK can't compute the individual gradients or
  second-order information of a L2-regularized loss, for example.



