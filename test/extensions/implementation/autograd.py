"""Autograd implementation of BackPACK's extensions."""
from math import isclose
from test.extensions.implementation.base import ExtensionsImplementation
from typing import Iterator, List, Union

from torch import Tensor, autograd, backends, cat, stack, var, zeros, zeros_like
from torch.nn.utils.convert_parameters import parameters_to_vector

from backpack.hessianfree.ggnvp import ggn_vector_product, ggn_vector_product_from_plist
from backpack.hessianfree.rop import R_op
from backpack.utils.convert_parameters import vector_to_parameter_list


class AutogradExtensions(ExtensionsImplementation):
    """Extension implementations with autograd."""

    def batch_grad(
        self, subsampling: Union[List[int], None]
    ) -> List[Tensor]:  # noqa: D102
        N = self.problem.get_batch_size()
        samples = list(range(N)) if subsampling is None else subsampling

        gradients_list = []
        for b in range(N):
            _, _, loss = self.problem.forward_pass(subsampling=[b])
            gradients = autograd.grad(loss, self.problem.trainable_parameters())
            gradients_list.append(gradients)

        batch_grads = [
            zeros(len(samples), *p.size()).to(self.problem.device)
            for p in self.problem.trainable_parameters()
        ]
        factor = self.problem.compute_reduction_factor()

        for out_idx, sample in enumerate(samples):
            for param_idx, sample_g in enumerate(gradients_list[sample]):
                batch_grads[param_idx][out_idx, :] = sample_g.detach() * factor

        return batch_grads

    def batch_l2_grad(self) -> List[Tensor]:  # noqa: D102
        return [
            (g**2).flatten(start_dim=1).sum(1)
            for g in self.batch_grad(subsampling=None)
        ]

    def sgs(self) -> List[Tensor]:  # noqa: D102
        return [(g**2).sum(0) for g in self.batch_grad(subsampling=None)]

    def variance(self) -> List[Tensor]:  # noqa: D102
        return [
            var(g, dim=0, unbiased=False) for g in self.batch_grad(subsampling=None)
        ]

    def _get_diag_ggn(self, loss: Tensor, output: Tensor) -> List[Tensor]:
        diag_ggn_flat = cat(
            [col[[i]] for i, col in enumerate(self._ggn_columns(loss, output))]
        )
        return vector_to_parameter_list(
            diag_ggn_flat, list(self.problem.trainable_parameters())
        )

    def diag_ggn(self) -> List[Tensor]:  # noqa: D102
        try:
            _, output, loss = self.problem.forward_pass()
            return self._get_diag_ggn(loss, output)
        except RuntimeError:
            # torch does not implement cuda double-backwards pass on RNNs and
            # recommends this workaround
            with backends.cudnn.flags(enabled=False):
                _, output, loss = self.problem.forward_pass()
                return self._get_diag_ggn(loss, output)

    def diag_ggn_exact_batch(self) -> List[Tensor]:  # noqa: D102
        try:
            return self._diag_ggn_exact_batch()
        except RuntimeError:
            # torch does not implement cuda double-backwards pass on RNNs and
            # recommends this workaround
            with backends.cudnn.flags(enabled=False):
                return self._diag_ggn_exact_batch()

    def _diag_ggn_exact_batch(self):
        # batch_diag_ggn has entries [sample_idx][param_idx]
        batch_diag_ggn = []
        for b in range(self.problem.get_batch_size()):
            _, output, loss = self.problem.forward_pass(subsampling=[b])
            diag_ggn = self._get_diag_ggn(loss, output)
            batch_diag_ggn.append(diag_ggn)

        factor = self.problem.compute_reduction_factor()

        # params_batch_diag_ggn has entries [param_idx][sample_idx]
        params_batch_diag_ggn = list(zip(*batch_diag_ggn))
        return [stack(param) * factor for param in params_batch_diag_ggn]

    def _get_diag_h(self, loss):
        def extract_ith_element_of_diag_h(i, p, df_dx):
            v = zeros_like(p).flatten()
            v[i] = 1.0
            vs = vector_to_parameter_list(v, [p])

            Hvs = R_op(df_dx, [p], vs)
            Hv = cat([g.flatten() for g in Hvs])

            return Hv[i]

        diag_hs = []
        for p in list(self.problem.trainable_parameters()):
            diag_h_p = zeros_like(p).flatten()

            df_dx = autograd.grad(loss, [p], create_graph=True, retain_graph=True)
            for parameter_index in range(p.numel()):
                diag_value = extract_ith_element_of_diag_h(parameter_index, p, df_dx)
                diag_h_p[parameter_index] = diag_value

            diag_hs.append(diag_h_p.view(p.size()))
        return diag_hs

    def diag_h(self) -> List[Tensor]:  # noqa: D102
        _, _, loss = self.problem.forward_pass()
        return self._get_diag_h(loss)

    def diag_h_batch(self) -> List[Tensor]:  # noqa: D102
        batch_diag_h = []
        for b in range(self.problem.get_batch_size()):
            _, _, loss = self.problem.forward_pass(subsampling=[b])
            diag_h = self._get_diag_h(loss)
            batch_diag_h.append(diag_h)

        factor = self.problem.compute_reduction_factor()

        params_batch_diag_h = list(zip(*batch_diag_h))
        return [stack(param) * factor for param in params_batch_diag_h]

    def ggn(self, subsampling: List[int] = None) -> Tensor:  # noqa: D102
        _, output, loss = self.problem.forward_pass(subsampling=subsampling)
        ggn = stack(list(self._ggn_columns(loss, output)), dim=1)

        # correct normalization constant for 'mean' reduction
        if subsampling is not None:
            factor = self.problem.compute_reduction_factor()
            if not isclose(factor, 1.0):
                ggn *= len(subsampling) * factor
            elif isclose(factor, 1.0) and self.problem.get_batch_size() == 1:
                ggn *= len(subsampling)

        return ggn

    def _ggn_columns(self, loss: Tensor, output: Tensor) -> Iterator[Tensor]:
        params = list(self.problem.trainable_parameters())
        num_params = sum(p.numel() for p in params)
        model = self.problem.model

        for i in range(num_params):
            # GGN-vector product with i.th unit vector yields the i.th row
            e_i = zeros(num_params).to(self.problem.device)
            e_i[i] = 1.0

            # convert to model parameter shapes
            e_i_list = vector_to_parameter_list(e_i, params)
            ggn_i_list = ggn_vector_product(loss, output, model, e_i_list)

            yield parameters_to_vector(ggn_i_list)

    def diag_ggn_mc(self, mc_samples) -> List[Tensor]:  # noqa: D102
        raise NotImplementedError

    def diag_ggn_mc_batch(self, mc_samples: int) -> List[Tensor]:  # noqa: D102
        raise NotImplementedError

    def ggn_mc(self, mc_samples: int, chunks: int = 1):  # noqa: D102
        raise NotImplementedError

    @staticmethod
    def ggn_param_block(outputs: Tensor, loss: Tensor, param: Tensor) -> Tensor:
        """Calculate the GGN block for a single parameter.

        Args:
            outputs: Model output.
            loss: Loss value.
            param: Parameter of the GGN block.

        Returns:
            Parameter GGN of shape ``[param.numel(), param.numel()]``.
        """
        columns = []

        for i in range(param.numel()):
            one_hot = zeros_like(param).flatten()
            one_hot[i] = 1.0
            one_hot = one_hot.reshape_as(param)

            (column,) = ggn_vector_product_from_plist(loss, outputs, [param], one_hot)
            columns.append(column.flatten())
        return stack(columns)

    def ggn_blocks(self) -> List[Tensor]:
        """Calculate the GGN blocks for all trainable parameters.

        Returns:
            GGN blocks for each trainable parameter.
        """
        _, outputs, loss = self.problem.forward_pass()

        return [
            self.ggn_param_block(outputs, loss, param)
            for param in self.problem.trainable_parameters()
        ]

    def kfac(self, mc_samples: int = 1) -> List[List[Tensor]]:  # noqa: D102
        raise NotImplementedError

    def kflr(self) -> List[List[Tensor]]:  # noqa: D102
        raise NotImplementedError

    def kfra(self) -> List[List[Tensor]]:  # noqa: D102
        raise NotImplementedError
