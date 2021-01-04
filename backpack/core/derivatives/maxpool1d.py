from backpack.core.derivatives.maxpoolnd import MaxPoolNDDerivatives


class MaxPool1DDerivatives(MaxPoolNDDerivatives):
    def __init__(self):
        super().__init__(N=1)

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        raise NotImplementedError
