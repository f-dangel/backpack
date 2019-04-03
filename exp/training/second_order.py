"""Training procedure using a second-order method."""

import torch
from .training import Training
from bpexts.hbp.loss import batch_summed_hessian


class SecondOrderTraining(Training):
    """Handle logging in training procedure with 2nd-order optimizers."""

    def __init__(self,
                 model,
                 loss_function,
                 optimizer,
                 data_loader,
                 logdir,
                 num_epochs,
                 modify_2nd_order_terms,
                 logs_per_epoch=10,
                 device=torch.device('cpu'),
                 input_shape=(-1, )):
        """Train a model, log loss values into logdir.

        See `Training` class in `training.py`

        Parameters:
        -----------
        modify_2nd_order_terms : string ('none', 'clip', 'sign', 'zero')
                String specifying the strategy for dealing with 2nd-order
                module effects in Hessian backpropagation
        """
        super().__init__(
            model,
            loss_function,
            optimizer,
            data_loader,
            logdir,
            num_epochs,
            logs_per_epoch=logs_per_epoch,
            device=device,
            input_shape=input_shape)
        self.modify_2nd_order = modify_2nd_order_terms

    # override
    def backward_pass(self, outputs, loss):
        """Perform backward pass for gradients and Hessians.

        Parameters:
        -----------
        outputs : (torch.Tensor)
            Tensor of size (batch_size, ...) storing the model outputs
        loss : (torch.Tensor)
            Scalar value of the loss function evaluated on the outputs
        """
        # Hessian of loss function w.r.t. outputs
        output_hessian = batch_summed_hessian(loss, outputs)
        # compute gradients
        super().backward_pass(outputs, loss)
        # backward Hessian
        self.model.backward_hessian(
            output_hessian,
            compute_input_hessian=False,
            modify_2nd_order_terms=self.modify_2nd_order)
