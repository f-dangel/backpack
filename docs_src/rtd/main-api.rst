How to use BackPACK
====================================

If you haven't already installed it,

.. code:: bash

	pip install backpack-for-pytorch


To use BackPACK with your setup,
you will need to :func:`extend <backpack.extend>` the model and the loss function
and register the extension you want to use with :func:`backpack <backpack.backpack>`
before calling ``backward()``.

Extending the model and loss function
--------------------------------------------

The :func:`extend(module) <backpack.extend>` function
tells backpack which parts of the model will be used
(and that it needs to track of to compute additional quantities).

.. code-block:: python

    import torch
    from backpack import extend
    from utils import load_data

    X, y = load_data()

    model = torch.nn.Sequential(
        torch.nn.Linear(784, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)
    )
    lossfunc = torch.nn.CrossEntropyLoss()

    model = extend(model)
    lossfunc = extend(lossfunc)


Calling the extension
---------------------------------

To activate an extension during the backward pass,
call ``backward()`` within a ``with backpack(extension):`` block;

.. code-block:: python

    from backpack import backpack
    from backpack.extensions import KFAC

    loss = lossfunc(model(X), y)

    with backpack(KFAC()):
        loss.backward()

    for param in model.parameters():
        print(param.grad)
        print(param.kfac)

See :ref:`Available Extensions` for other quantities,
and the :ref:`Supported models`.

-----

.. autofunction:: backpack.extend
.. autofunction:: backpack.backpack

