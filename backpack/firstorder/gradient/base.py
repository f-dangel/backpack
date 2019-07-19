from ..firstorder import FirstOrderExtension


class GradBase(FirstOrderExtension):
    def __init__(self,
                 module,
                 extension,
                 derivatives,
                 params=None,
                 req_inputs=None,
                 req_output=False):
        super().__init__(
            module,
            extension,
            params=params,
            req_inputs=req_inputs,
            req_output=req_output,
            derivatives=derivatives)

    # TODO: Same code as for batch gradient, but with sum_batch = True
    def bias(self, module, grad_input, grad_output):
        batch = grad_output[0].shape[0]
        grad_out_vec = grad_output[0].contiguous().view(batch, -1)

        bias_grad = self.derivatives.bias_jac_t_mat_prod(
            module, grad_input, grad_output, grad_out_vec, sum_batch=True)

        shape = module.bias.shape
        return bias_grad.view(shape)

    # TODO: Same code as for batch gradient, but with sum_batch = True
    def weight(self, module, grad_input, grad_output):
        batch = grad_output[0].shape[0]
        grad_out_vec = grad_output[0].contiguous().view(batch, -1)

        weight_grad = self.derivatives.weight_jac_t_mat_prod(
            module, grad_input, grad_output, grad_out_vec, sum_batch=True)

        shape = module.weight.shape
        return weight_grad.view(shape)
