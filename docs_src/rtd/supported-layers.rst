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

This page lists the layers currently supported by BackPACK.


**Do not rewrite the** :code:`forward()` **function of the** :code:`Sequential` **or the inner modules!**
If the forward is not standard, the additional backward pass to compute second-order quantities will not match the actual function.
First-order extensions that extract information might work outside of this framework, but it is not tested.

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

First-order extensions should support any module as long as they do not have parameters,
but some layers lead to the concept of "individual gradient for a sample in a minibatch"
to be ill-defined, as they introduce dependencies across examples
(like :py:class:`torch.nn.BatchNorm`).

-----

For second-order extensions
--------------------------------------

BackPACK needs to know how to propagate second-order information.
This is implemented for:

+-------------------------------+---------------------------------------+
| **Parametrized layers**       | :py:class:`torch.nn.Conv2d`           |
|                               +---------------------------------------+
|                               | :py:class:`torch.nn.Linear`           |
+-------------------------------+---------------------------------------+
| **Loss functions**            | :py:class:`torch.nn.MSELoss`          |
|                               +---------------------------------------+
|                               | :py:class:`torch.nn.CrossEntropyLoss` |
+-------------------------------+---------------------------------------+
| **Layers without parameters** | :py:class:`torch.nn.MaxPool2d`        |
|                               | :py:class:`torch.nn.AvgPool2d`        |
|                               +---------------------------------------+
|                               | :py:class:`torch.nn.Dropout`          |
|                               +---------------------------------------+
|                               | :py:class:`torch.nn.ReLU`             |
|                               | :py:class:`torch.nn.Sigmoid`          |
|                               | :py:class:`torch.nn.Tanh`             |
+-------------------------------+---------------------------------------+

The other convolution layers (``Conv1d``, ``Conv3d``, and ``ConvTransposeNd``)
are not yet supported.
