Supported models
====================================

BackPACK expects models to be
`sequences <https://pytorch.org/docs/stable/nn.html#sequential>`_
of `PyTorch NN modules <https://pytorch.org/docs/stable/nn.html>`_.
For example,

.. code-block:: python

	model = torch.nn.Sequential(
		torch.nn.Linear(784, 64),
		torch.nn.ReLU(),
		torch.nn.Linear(64, 10)
	)

**If you overwrite any** :code:`forward()` **function** (for example in ResNets
and RNNs), the additional backward pass to compute second-order quantities will
break. You can ask BackPACK to inspect the graph and try converting it
into a compatible structure by setting :code:`use_converter=True` in
:py:func:`extend <backpack.extend()>`.

This page lists the layers currently supported by BackPACK.

.. raw:: html

	<hr/>

For first-order extensions
--------------------------------------

BackPACK can extract more information about the gradient with respect to the
parameters of the following layers;

- :py:class:`torch.nn.Linear`
- :py:class:`torch.nn.Conv1d`,
  :py:class:`torch.nn.Conv2d`,
  :py:class:`torch.nn.Conv3d`
- :py:class:`torch.nn.ConvTranspose1d`,
  :py:class:`torch.nn.ConvTranspose2d`,
  :py:class:`torch.nn.ConvTranspose3d`
- :py:class:`torch.nn.BatchNorm1d` (evaluation mode),
  :py:class:`torch.nn.BatchNorm2d` (evaluation mode),
  :py:class:`torch.nn.BatchNorm3d` (evaluation mode)
- :py:class:`torch.nn.Embedding`
- :py:class:`torch.nn.RNN`, :py:class:`torch.nn.LSTM`

Some layers (like :code:`torch.nn.BatchNormNd` in training mode) mix samples and
lead to ill-defined first-order quantities.

-----

For second-order extensions
--------------------------------------

BackPACK needs to know how to backpropagate additional information for
second-order quantities. This is implemented for:

+-------------------------------+-----------------------------------------------+
| **Parametrized layers**       | :py:class:`torch.nn.Conv1d`,                  |
|                               | :py:class:`torch.nn.Conv2d`,                  |
|                               | :py:class:`torch.nn.Conv3d`                   |
|                               +-----------------------------------------------+
|                               | :py:class:`torch.nn.ConvTranspose1d`,         |
|                               | :py:class:`torch.nn.ConvTranspose2d`,         |
|                               | :py:class:`torch.nn.ConvTranspose3d`          |
|                               +-----------------------------------------------+
|                               | :py:class:`torch.nn.Linear`                   |
|                               +-----------------------------------------------+
|                               | :py:class:`torch.nn.BatchNorm1d`,             |
|                               | :py:class:`torch.nn.BatchNorm2d`,             |
|                               | :py:class:`torch.nn.BatchNorm3d`              |
|                               +-----------------------------------------------+
|                               | :py:class:`torch.nn.Embedding`                |
|                               +-----------------------------------------------+
|                               | :py:class:`torch.nn.RNN`,                     |
|                               | :py:class:`torch.nn.LSTM`                     |
+-------------------------------+-----------------------------------------------+
| **Loss functions**            | :py:class:`torch.nn.MSELoss`                  |
|                               +-----------------------------------------------+
|                               | :py:class:`torch.nn.CrossEntropyLoss`         |
+-------------------------------+-----------------------------------------------+
| **Layers without parameters** | :py:class:`torch.nn.MaxPool1d`,               |
|                               | :py:class:`torch.nn.MaxPool2d`,               |
|                               | :py:class:`torch.nn.MaxPool3d`                |
|                               +-----------------------------------------------+
|                               | :py:class:`torch.nn.AvgPool1d`,               |
|                               | :py:class:`torch.nn.AvgPool2d`,               |
|                               | :py:class:`torch.nn.AvgPool3d`                |
|                               +-----------------------------------------------+
|                               | :py:class:`torch.nn.AdaptiveAvgPool1d`,       |
|                               | :py:class:`torch.nn.AdaptiveAvgPool2d`,       |
|                               | :py:class:`torch.nn.AdaptiveAvgPool3d`        |
|                               +-----------------------------------------------+
|                               | :py:class:`torch.nn.ZeroPad2d`,               |
|                               +-----------------------------------------------+
|                               | :py:class:`torch.nn.Dropout`                  |
|                               +-----------------------------------------------+
|                               | :py:class:`torch.nn.ReLU`,                    |
|                               | :py:class:`torch.nn.Sigmoid`,                 |
|                               | :py:class:`torch.nn.Tanh`,                    |
|                               | :py:class:`torch.nn.LeakyReLU`,               |
|                               | :py:class:`torch.nn.LogSigmoid`,              |
|                               | :py:class:`torch.nn.ELU`,                     |
|                               | :py:class:`torch.nn.SELU`                     |
|                               +-----------------------------------------------+
|                               | :py:class:`torch.nn.Identity`                 |
|                               +-----------------------------------------------+
|                               | :py:class:`torch.nn.Flatten`                  |
+-------------------------------+-----------------------------------------------+
