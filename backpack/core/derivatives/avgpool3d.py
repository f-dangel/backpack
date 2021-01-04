"""The code relies on the insight that average pooling can be understood as
convolution over single channels with a constant kernel."""
from backpack.core.derivatives.avgpoolnd import AvgPoolNDDerivatives


class AvgPool3DDerivatives(AvgPoolNDDerivatives):
    def __init__(self):
        super().__init__(N=3)
    
    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        raise NotImplementedError