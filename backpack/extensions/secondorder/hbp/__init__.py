from torch.nn import (
    AvgPool2d,
    Conv2d,
    CrossEntropyLoss,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    MSELoss,
    ReLU,
    Sigmoid,
    Tanh,
    ZeroPad2d,
)

from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.curvature import Curvature
from backpack.extensions.secondorder.hbp.hbp_options import (
    BackpropStrategy,
    ExpectationApproximation,
    LossHessianStrategy,
)

from . import activations, conv2d, dropout, flatten, linear, losses, padding, pooling


class HBP(BackpropExtension):
    def __init__(
        self,
        curv_type,
        loss_hessian_strategy,
        backprop_strategy,
        ea_strategy,
        savefield="hbp",
    ):
        self.curv_type = curv_type
        self.loss_hessian_strategy = loss_hessian_strategy
        self.backprop_strategy = backprop_strategy
        self.ea_strategy = ea_strategy

        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.HBPMSELoss(),
                CrossEntropyLoss: losses.HBPCrossEntropyLoss(),
                Linear: linear.HBPLinear(),
                MaxPool2d: pooling.HBPMaxpool2d(),
                AvgPool2d: pooling.HBPAvgPool2d(),
                ZeroPad2d: padding.HBPZeroPad2d(),
                Conv2d: conv2d.HBPConv2d(),
                Dropout: dropout.HBPDropout(),
                Flatten: flatten.HBPFlatten(),
                ReLU: activations.HBPReLU(),
                Sigmoid: activations.HBPSigmoid(),
                Tanh: activations.HBPTanh(),
            },
        )

    def get_curv_type(self):
        return self.curv_type

    def get_loss_hessian_strategy(self):
        return self.loss_hessian_strategy

    def get_backprop_strategy(self):
        return self.backprop_strategy

    def get_ea_strategy(self):
        return self.ea_strategy


class KFAC(HBP):
    """
    Approximate Kronecker factorization of the Generalized Gauss-Newton/Fisher
    using Monte-Carlo sampling.

    Stores the output in :code:`kfac` as a list of Kronecker factors.

    - If there is only one element, the item represents the GGN/Fisher
      approximation itself.
    - If there are multiple elements, they are arranged in the order such
      that their Kronecker product represents the Generalized Gauss-Newton/Fisher
      approximation.
    - The dimension of the factors depends on the layer, but the product
      of all row dimensions (or column dimensions) yields the dimension of the
      layer parameter.

    .. note::
        The literature uses column-stacking as vectorization convention,
        but ``torch`` defaults to a row-major storing scheme of tensors.
        The order of factors might differs from the presentation in the literature.

    Implements the procedures described by

    - `Optimizing Neural Networks with Kronecker-factored Approximate Curvature
      <http://proceedings.mlr.press/v37/martens15.html>`_
      by James Martens and Roger Grosse, 2015.

    - `A Kronecker-factored approximate Fisher matrix for convolution layers
      <http://proceedings.mlr.press/v48/grosse16.html>`_
      by Roger Grosse and James Martens, 2016
    """

    def __init__(self, mc_samples=1):
        self._mc_samples = mc_samples
        super().__init__(
            curv_type=Curvature.GGN,
            loss_hessian_strategy=LossHessianStrategy.SAMPLING,
            backprop_strategy=BackpropStrategy.SQRT,
            ea_strategy=ExpectationApproximation.BOTEV_MARTENS,
            savefield="kfac",
        )

    def get_num_mc_samples(self):
        return self._mc_samples


class KFRA(HBP):
    """
    Approximate Kronecker factorization of the Generalized Gauss-Newton/Fisher
    using the full Hessian of the loss function w.r.t. the model output
    and averaging after every backpropagation step.

    Stores the output in :code:`kfra` as a list of Kronecker factors.

    - If there is only one element, the item represents the GGN/Fisher
      approximation itself.
    - If there are multiple elements, they are arranged in the order such
      that their Kronecker product represents the Generalized Gauss-Newton/Fisher
      approximation.
    - The dimension of the factors depends on the layer, but the product
      of all row dimensions (or column dimensions) yields the dimension of the
      layer parameter.

    .. note::

        The literature uses column-stacking as vectorization convention.
        This is in contrast to the default row-major storing scheme of tensors
        in :code:`torch`. Therefore, the order of factors differs from the
        presentation in the literature.

    - `Practical Gauss-Newton Optimisation for Deep Learning
      <http://proceedings.mlr.press/v70/botev17a.html>`_
      by Aleksandar Botev, Hippolyt Ritter and David Barber, 2017.

    Extended for convolutions following

    - `A Kronecker-factored approximate Fisher matrix for convolution layers
      <http://proceedings.mlr.press/v48/grosse16.html>`_
      by Roger Grosse and James Martens, 2016
    """

    def __init__(self):
        super().__init__(
            curv_type=Curvature.GGN,
            loss_hessian_strategy=LossHessianStrategy.SUM,
            backprop_strategy=BackpropStrategy.BATCH_AVERAGE,
            ea_strategy=ExpectationApproximation.BOTEV_MARTENS,
            savefield="kfra",
        )


class KFLR(HBP):
    """
    Approximate Kronecker factorization of the Generalized Gauss-Newton/Fisher
    using the full Hessian of the loss function w.r.t. the model output.

    Stores the output in :code:`kflr` as a list of Kronecker factors.

    - If there is only one element, the item represents the GGN/Fisher
      approximation itself.
    - If there are multiple elements, they are arranged in the order such
      that their Kronecker product represents the Generalized Gauss-Newton/Fisher
      approximation.
    - The dimension of the factors depends on the layer, but the product
      of all row dimensions (or column dimensions) yields the dimension of the
      layer parameter.

    .. note::

        The literature uses column-stacking as vectorization convention.
        This is in contrast to the default row-major storing scheme of tensors
        in :code:`torch`. Therefore, the order of factors differs from the
        presentation in the literature.

    Implements the procedures described by

    - `Practical Gauss-Newton Optimisation for Deep Learning
      <http://proceedings.mlr.press/v70/botev17a.html>`_
      by Aleksandar Botev, Hippolyt Ritter and David Barber, 2017.

    Extended for convolutions following

    - `A Kronecker-factored approximate Fisher matrix for convolution layers
      <http://proceedings.mlr.press/v48/grosse16.html>`_
      by Roger Grosse and James Martens, 2016
    """

    def __init__(self):
        super().__init__(
            curv_type=Curvature.GGN,
            loss_hessian_strategy=LossHessianStrategy.EXACT,
            backprop_strategy=BackpropStrategy.SQRT,
            ea_strategy=ExpectationApproximation.BOTEV_MARTENS,
            savefield="kflr",
        )
