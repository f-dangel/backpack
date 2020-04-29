.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_basic_usage_example_all_in_one.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_basic_usage_example_all_in_one.py:


Example using all extensions
==============================

Basic example showing how compute the gradient,
and and other quantities with BackPACK,
on a linear model for MNIST.

Let's start by loading some dummy data and extending the model


.. code-block:: default


    from backpack.utils.examples import load_one_batch_mnist
    from torch.nn import CrossEntropyLoss, Flatten, Linear, Sequential
    from backpack import backpack, extend
    from backpack.extensions import KFAC, KFLR, KFRA
    from backpack.extensions import DiagGGNExact, DiagGGNMC, DiagHessian
    from backpack.extensions import BatchGrad, SumGradSquared, Variance, BatchL2Grad

    X, y = load_one_batch_mnist(batch_size=512)

    model = Sequential(Flatten(), Linear(784, 10),)
    lossfunc = CrossEntropyLoss()

    model = extend(model)
    lossfunc = extend(lossfunc)








First order extensions
----------------------

Batch gradients


.. code-block:: default


    loss = lossfunc(model(X), y)
    with backpack(BatchGrad()):
        loss.backward()

    for name, param in model.named_parameters():
        print(name)
        print(".grad.shape:             ", param.grad.shape)
        print(".grad_batch.shape:       ", param.grad_batch.shape)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1.weight
    .grad.shape:              torch.Size([10, 784])
    .grad_batch.shape:        torch.Size([512, 10, 784])
    1.bias
    .grad.shape:              torch.Size([10])
    .grad_batch.shape:        torch.Size([512, 10])




Variance


.. code-block:: default


    loss = lossfunc(model(X), y)
    with backpack(Variance()):
        loss.backward()

    for name, param in model.named_parameters():
        print(name)
        print(".grad.shape:             ", param.grad.shape)
        print(".variance.shape:         ", param.variance.shape)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1.weight
    .grad.shape:              torch.Size([10, 784])
    .variance.shape:          torch.Size([10, 784])
    1.bias
    .grad.shape:              torch.Size([10])
    .variance.shape:          torch.Size([10])




Second moment/sum of gradients squared


.. code-block:: default


    loss = lossfunc(model(X), y)
    with backpack(SumGradSquared()):
        loss.backward()

    for name, param in model.named_parameters():
        print(name)
        print(".grad.shape:             ", param.grad.shape)
        print(".sum_grad_squared.shape: ", param.sum_grad_squared.shape)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1.weight
    .grad.shape:              torch.Size([10, 784])
    .sum_grad_squared.shape:  torch.Size([10, 784])
    1.bias
    .grad.shape:              torch.Size([10])
    .sum_grad_squared.shape:  torch.Size([10])




L2 norm of individual gradients


.. code-block:: default


    loss = lossfunc(model(X), y)
    with backpack(BatchL2Grad()):
        loss.backward()

    for name, param in model.named_parameters():
        print(name)
        print(".grad.shape:             ", param.grad.shape)
        print(".batch_l2.shape:         ", param.batch_l2.shape)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1.weight
    .grad.shape:              torch.Size([10, 784])
    .batch_l2.shape:          torch.Size([512])
    1.bias
    .grad.shape:              torch.Size([10])
    .batch_l2.shape:          torch.Size([512])




It's also possible to ask for multiple quantities at once


.. code-block:: default


    loss = lossfunc(model(X), y)
    with backpack(BatchGrad(), Variance(), SumGradSquared(), BatchL2Grad()):
        loss.backward()

    for name, param in model.named_parameters():
        print(name)
        print(".grad.shape:             ", param.grad.shape)
        print(".grad_batch.shape:       ", param.grad_batch.shape)
        print(".variance.shape:         ", param.variance.shape)
        print(".sum_grad_squared.shape: ", param.sum_grad_squared.shape)
        print(".batch_l2.shape:         ", param.batch_l2.shape)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1.weight
    .grad.shape:              torch.Size([10, 784])
    .grad_batch.shape:        torch.Size([512, 10, 784])
    .variance.shape:          torch.Size([10, 784])
    .sum_grad_squared.shape:  torch.Size([10, 784])
    .batch_l2.shape:          torch.Size([512])
    1.bias
    .grad.shape:              torch.Size([10])
    .grad_batch.shape:        torch.Size([512, 10])
    .variance.shape:          torch.Size([10])
    .sum_grad_squared.shape:  torch.Size([10])
    .batch_l2.shape:          torch.Size([512])




Second order extensions
--------------------------

Diagonal of the Gauss-Newton and its Monte-Carlo approximation


.. code-block:: default


    loss = lossfunc(model(X), y)
    with backpack(DiagGGNExact(), DiagGGNMC(mc_samples=1)):
        loss.backward()

    for name, param in model.named_parameters():
        print(name)
        print(".grad.shape:             ", param.grad.shape)
        print(".diag_ggn_mc.shape:      ", param.diag_ggn_mc.shape)
        print(".diag_ggn_exact.shape:   ", param.diag_ggn_exact.shape)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1.weight
    .grad.shape:              torch.Size([10, 784])
    .diag_ggn_mc.shape:       torch.Size([10, 784])
    .diag_ggn_exact.shape:    torch.Size([10, 784])
    1.bias
    .grad.shape:              torch.Size([10])
    .diag_ggn_mc.shape:       torch.Size([10])
    .diag_ggn_exact.shape:    torch.Size([10])




KFAC, KFRA and KFLR


.. code-block:: default


    loss = lossfunc(model(X), y)
    with backpack(KFAC(mc_samples=1), KFLR(), KFRA()):
        loss.backward()

    for name, param in model.named_parameters():
        print(name)
        print(".grad.shape:             ", param.grad.shape)
        print(".kfac (shapes):          ", [kfac.shape for kfac in param.kfac])
        print(".kflr (shapes):          ", [kflr.shape for kflr in param.kflr])
        print(".kfra (shapes):          ", [kfra.shape for kfra in param.kfra])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1.weight
    .grad.shape:              torch.Size([10, 784])
    .kfac (shapes):           [torch.Size([10, 10]), torch.Size([784, 784])]
    .kflr (shapes):           [torch.Size([10, 10]), torch.Size([784, 784])]
    .kfra (shapes):           [torch.Size([10, 10]), torch.Size([784, 784])]
    1.bias
    .grad.shape:              torch.Size([10])
    .kfac (shapes):           [torch.Size([10, 10])]
    .kflr (shapes):           [torch.Size([10, 10])]
    .kfra (shapes):           [torch.Size([10, 10])]




Diagonal Hessian


.. code-block:: default


    loss = lossfunc(model(X), y)
    with backpack(DiagHessian()):
        loss.backward()

    for name, param in model.named_parameters():
        print(name)
        print(".grad.shape:             ", param.grad.shape)
        print(".diag_h.shape:           ", param.diag_h.shape)




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1.weight
    .grad.shape:              torch.Size([10, 784])
    .diag_h.shape:            torch.Size([10, 784])
    1.bias
    .grad.shape:              torch.Size([10])
    .diag_h.shape:            torch.Size([10])





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  1.571 seconds)


.. _sphx_glr_download_basic_usage_example_all_in_one.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: example_all_in_one.py <example_all_in_one.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: example_all_in_one.ipynb <example_all_in_one.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
