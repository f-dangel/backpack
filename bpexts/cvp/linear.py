"""Curvature-vector products for linear layer."""

from torch.nn import Linear
from ..hbp.module import hbp_decorate
from ..utils import einsum


class CVPLinear(hbp_decorate(Linear)):
    """Linear layer with recursive Hessian-vector products."""
    # override
    @classmethod
    def from_torch(cls, torch_layer):
        if not isinstance(torch_layer, Linear):
            raise ValueError("Expecting torch.nn.Linear, got {}".format(
                torch_layer.__class__))
        # create instance
        linear = cls(
            in_features=torch_layer.in_features,
            out_features=torch_layer.out_features,
            bias=torch_layer.bias is not None)
        # copy parameters
        linear.weight = torch_layer.weight
        linear.bias = torch_layer.bias
        return linear

    # override
    def hbp_hooks(self):
        """Install hooks to track quantities required for CVP."""
        self.register_exts_forward_pre_hook(self.store_input)

    # --- hooks ---
    @staticmethod
    def store_input(module, input):
        """Save input to layer.

        Intended use as pre-forward hook.
        Initialize module buffer 'layer_input'.
        """
        if not len(input) == 1:
            raise ValueError('Cannot handle multi-input scenario')
        layer_input = input[0].detach()
        module.register_exts_buffer('layer_input', layer_input)

    # --- end of hooks ---

    # --- Hessian-vector product with the input Hessian ---
    # override
    def input_hessian(self, output_hessian, modify_2nd_order_terms='none'):
        """Return CVP with respect to the input."""

        def _input_hessian_vp(v):
            """Multiplication by the Hessian w.r.t. the input."""
            return self._input_jacobian_transpose(
                output_hessian(self._input_jacobian(v))).detach()

        return _input_hessian_vp

    def _input_jacobian(self, v):
        """Apply the Jacobian with respect to the input."""
        batch, _ = tuple(self.layer_input.size())
        assert tuple(v.size()) == (batch * self.in_features, )
        result = einsum('bj,ij->bi',
                        (v.view(batch, self.in_features), self.weight))
        assert tuple(result.size()) == (batch, self.out_features)
        return result.view(-1)

    def _input_jacobian_transpose(self, v):
        """Apply the transposed Jacobian with respect to the input."""
        batch, _ = tuple(self.layer_input.size())
        assert tuple(v.size()) == (batch * self.out_features, )
        result = einsum('bi,ij->bj',
                        (v.view(batch, self.out_features), self.weight))
        assert tuple(result.size()) == (batch, self.in_features)
        return result.view(-1)

    # --- Hessian-vector products with the parameter Hessians ---
    # override
    def parameter_hessian(self, output_hessian):
        """Initialize VPs with the layer parameter Hessian."""
        if self.bias is not None:
            self.init_bias_hessian(output_hessian)
        self.init_weight_hessian(output_hessian)

    # --- bias term ---
    def init_bias_hessian(self, output_hessian):
        """Initialize bias Hessian-vector product."""

        def _bias_hessian_vp(v):
            """Multiplication by the bias Hessian."""
            return self._bias_jacobian_transpose(
                output_hessian(self._bias_jacobian(v)))

        self.bias.hvp = _bias_hessian_vp

    def _bias_jacobian(self, v):
        """Apply the Jacobian with respect to the bias."""
        batch, _ = tuple(self.layer_input.size())
        assert tuple(v.size()) == (self.out_features, )
        result = v.view(1, self.out_features)
        result = result.expand(batch, self.out_features)
        assert tuple(result.size()) == (batch, self.out_features)
        return result.contiguous().view(-1)

    def _bias_jacobian_transpose(self, v):
        """Apply the transposed Jacobian with respect to the bias."""
        batch, _ = tuple(self.layer_input.size())
        assert tuple(v.size()) == (batch * self.out_features, )
        result = v.view(batch, self.out_features).sum(0).view(-1)
        assert tuple(result.size()) == (self.out_features, )
        return result

    def init_weight_hessian(self, output_hessian):
        """Initialize weight Hessian-vector product."""

        def _weight_hessian_vp(v):
            """Multiplication by the weight Hessian."""
            return self._weight_jacobian_transpose(
                output_hessian(self._weight_jacobian(v)))

        self.weight.hvp = _weight_hessian_vp

    def _weight_jacobian(self, v):
        """Apply the Jacobian with respect to the weights."""
        batch, _ = tuple(self.layer_input.size())
        assert tuple(v.size()) == (self.weight.numel(), )
        result = v.view(1, self.out_features, self.in_features)
        result = result.expand(batch, self.out_features, self.in_features)
        assert tuple(result.size()) == (batch, self.out_features,
                                        self.in_features)
        result = einsum('bj,bij->bi', (self.layer_input, result))
        assert tuple(result.size()) == (batch, self.out_features)
        return result.view(-1)

    def _weight_jacobian_transpose(self, v):
        """Apply the transposed Jacobian with respect to the weights."""
        batch, _ = tuple(self.layer_input.size())
        assert tuple(v.size()) == (batch * self.out_features, )
        result = einsum('bj,bi->ji',
                        (v.view(batch, self.out_features), self.layer_input))
        assert tuple(result.size()) == (self.out_features, self.in_features)
        return result.view(-1)
