from ..firstorder import FirstOrderExtension


class BatchGradBase(FirstOrderExtension):
    SUM_OVER_BATCH = False

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

    def bias(self, module, grad_input, grad_output):
        batch = grad_output[0].shape[0]
        grad_out_vec = grad_output[0].contiguous().view(batch, -1)

        bias_grad = self.derivatives.bias_jac_t_mat_prod(
            module,
            grad_input,
            grad_output,
            grad_out_vec,
            sum_batch=self.SUM_OVER_BATCH)

        shape = module.bias.shape
        if self.SUM_OVER_BATCH is False:
            shape = (batch, ) + shape

        return bias_grad.view(shape)

    def weight(self, module, grad_input, grad_output):
        batch = grad_output[0].shape[0]
        grad_out_vec = grad_output[0].contiguous().view(batch, -1)

        weight_grad = self.derivatives.weight_jac_t_mat_prod(
            module,
            grad_input,
            grad_output,
            grad_out_vec,
            sum_batch=self.SUM_OVER_BATCH)

        shape = module.weight.shape
        if self.SUM_OVER_BATCH is False:
            shape = (batch, ) + shape

        return weight_grad.view(shape)
