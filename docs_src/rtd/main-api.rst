How to use BackPACK
====================================

The use BackPACK with your setup, you first need to :py:meth:`backpack.extend` the model and the loss function
and register the extension you want to use with :py:meth:`backpack.backpack`
before calling the :code:`backward()` function 

Extending the model and loss function
--------------------------------------------

.. code-block:: python

	import torch 
	
	model = torch.nn.Sequential(
		torch.nn.Linear(764, 64),
		torch.nn.ReLU(),
		torch.nn.Linear(64, 10)
	)
	lossfunc = torch.nn.CrossEntropyLoss()
	
	model = extend(model)
	lossfunc = extend(lossfunc)

See :ref:`Supported models` for the list of supported layers.


.. autofunction:: backpack.extend

Calling the extension
---------------------------------

.. code-block:: python

	from backpack import backpack
	from backpack.extensions import KFAC
	from utils import load_data
	
	X, y = load_data()
	
	loss = lossfunc(model(X), y)
	
	with backpack(KFAC()):
		loss.backward()
		
		for param in model.parameters():
			print(param.grad)
			print(param.kfac)
			
See :ref:`Extensions` for the list of available extensions and how to access the quantities.

.. autofunction:: backpack.backpack

