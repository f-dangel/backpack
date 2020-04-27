Good to know
====================================

.. warning::

	Work in progress

Backpack does not do aggregating like pytorch for the gradient. Calling backward with backpack multiple time resets the quantities, it does not add them.

Not everything needs to be extended for first order extensions. Only modules with parameters need to be.

The Sequential module format is only necessary for second-order extensions, and needs to be respected in that case, otherwise bad things happen. But for first-order extension, you can rewrite the forward pass of your thingy as long as the module with parameters are extended.

Background: The Sequential defines the computational graph assumed by BackPACK for 2nd-order backpropagation.

Note: BackPACK cannot (yet) fully support residual neural networks!

You should not use the Diagonal Hessian. It’s there if you need it, but it’s not going to be efficient. Our implementation is made to work best when the network is simple, but is going to struggle if you have many nonlinearities. The goal is to make it possible, not efficient. That’s why many people have been working on approximations that work. Use those.

The first order extensions will change depending on the scaling, including mean or sum. Double check what you’re doing.

Generally, we recommend double checking the output of Backpack. It’s a relatively new library and while we’re trying hard, bugs might lurk. Check that Backpack is computing what you think it is computing and let us know if something is weird. (There might be utility scripts for that in the near future)

This especially applies to exotic hyperparameters (e.g. dilation, groups for convolution or pooling)

Please do not use the same module (e.g. `torch.nn.Sigmoid`) multiple times. We do not know what is going to happen in that case. The results will be wrong with high probability.

Please do not use `inplace=True` for activation functions such as `torch.nn.Sigmoid`. We have no experience/tests how/if this is going to affect the computation.
