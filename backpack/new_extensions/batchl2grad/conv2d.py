from backpack.utils.utils import einsum
from backpack.utils import conv as convUtils
from backpack.new_extensions.firstorder import FirstOrderExtension


class BatchL2Conv2d(FirstOrderExtension):
    def __init__(self):
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, grad_input, grad_output, backproped):
        return (grad_output[0].sum(3).sum(2)**2).sum(1)

    def weight(self, ext, module, grad_input, grad_output, backproped):
        X, dE_dY = convUtils.get_weight_gradient_factors(
            module.input0, grad_output[0], module)
        return einsum('bml,bkl,bmi,bki->b', (dE_dY, X, dE_dY, X))


class BatchL2Conv2dConcat(FirstOrderExtension):
    def __init__(self):
        super().__init__(params=["weight"])

    def weight(self, ext, module, grad_input, grad_output, backproped):
        X, dE_dY = convUtils.get_weight_gradient_factors(
            module.input0, grad_output[0], module)

        if module.has_bias():
            X = module.append_ones(X)

        return einsum('bml,bkl,bmi,bki->b', (dE_dY, X, dE_dY, X))
