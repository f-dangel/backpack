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

The :func:`extend(torch.nn.Module) <backpack.extend>` function
tells BackPACK what part of the computation graph needs to be tracked.
If your model is a :py:class:`torch.nn.Sequential` and you use one of the
:py:class:`torch.nn` loss functions;

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

To activate an extension, call ``backward()`` inside a
``with backpack(extension):`` block;

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

