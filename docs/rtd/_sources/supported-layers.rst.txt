Supported models
====================================

BackPACK expects models to be 
`sequences <https://pytorch.org/docs/stable/nn.html#sequential>`_ 
of `PyTorch NN modules <https://pytorch.org/docs/stable/nn.html>`_.
For example, 

.. code-block:: python

	model = torch.nn.Sequential(
		torch.nn.Linear(764, 64),
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

You can use any layers, as long as they do not have parameters.
BackPACK can extract more information about the gradient w.r.t. the parameters of those layers:

* `Conv2d <https://pytorch.org/docs/stable/nn.html#conv2d>`_
* `Linear <https://pytorch.org/docs/stable/nn.html#linear>`_

**Some layers lead to the concept of "inidividual gradient for a sample in a minibatch" to be ill-defined.**
This is the case for Batch Normalization layers, for example.

.. raw:: html 
	
	<hr/>

For second-order extensions
--------------------------------------

BackPACK needs to know how to compute an additional backward pass.
In addition to the parametrized layers above, this implemented for the following layers:

**Loss functions**

* `MSELoss <https://pytorch.org/docs/stable/nn.html#mseloss>`_
* `CrossEntropyLoss <https://pytorch.org/docs/stable/nn.html#crossentropyloss>`_

**Layers without parameters**

* `MaxPool2d <https://pytorch.org/docs/stable/nn.html#maxpool2d>`_
* `AvgPool2d <https://pytorch.org/docs/stable/nn.html#avgpool2d>`_
* `Dropout <https://pytorch.org/docs/stable/nn.html#dropout>`_
* `ReLU <https://pytorch.org/docs/stable/nn.html#relu>`_
* `Sigmoid <https://pytorch.org/docs/stable/nn.html#sigmoid>`_
* `Tanh <https://pytorch.org/docs/stable/nn.html#tanh>`_


Custom layers
--------------------------------------

:code:`torch.nn.functional.flatten` can not be used in this setup because it is a function, not a module.
Use :code:`backpack.core.layers.Flatten` instead.
