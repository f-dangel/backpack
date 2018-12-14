"""Class for handling training and logging metrics."""

from tensorboard_logger import Logger
from os import path
from tqdm import tqdm
import torch

import enable_import_bpexts
import bpexts


class FirstOrderTraining(object):
    """Handle logging during training procedure with 1st-order optimizers."""
    def __init__(self, model, loss_function, optimizer,
                 data_loader, logdir):
        """Train a model, log loss values into logdir.

        Parameters:
        -----------
        model : (torch.nn.Module)
            Trainable model
        loss_function : (torch.nn.Functional / torch.nn.Module)
            Scalar loss function computed on top of the model output
        optimizer : (torch.optim.Optimizer)
            Optimizer used for training
        batch_size (int): Number of samples in a batch
        num_epochs (int): Epochs to train the model
        reuse (int): Number of iterations to reuse the Hessian
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.logger = self.create_logger(logdir)

    def log_values(self, summary, step):
        """Log all key-value pairs in `summary` at `step`.

        Parameters:
        -----------
        summary : (dict)
            Dictionary with scalar items step : (int)
            Step for logging (must be int)
        """
        for key, value in summary.items():
            self.logger.log_value(key, value, step)

    def run(self,
            num_epochs,
            device=torch.device('cpu'),
            logs_per_epoch=10):
        """Run training, log values to logdir.

        Parameters:
        -----------
        num_epochs : (int)
            Number of epochs to train on
        device : (torch.Device)
           Device to run the training on
        logs_per_epoch : (int)
            Number of datapoints written into logdir per epoch
        """
        self.model = self.model.to(device)
        training_set = self.data_loader.train_loader()
        num_batches = len(training_set)
        log_every = max(num_batches // logs_per_epoch, 1)

        train_samples_seen = 0
        progressbar = tqdm(range(num_epochs))
        for epoch in progressbar:
            for idx, (inputs, labels) in enumerate(training_set):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, loss = self.forward_pass(inputs, labels)
                # backward pass and update step
                self.optimizer.zero_grad()
                self.backward_pass(outputs, loss)
                self.optimizer.step()
                batch_size = inputs.size()[0]
                train_samples_seen += batch_size

                # logging of metrics
                if idx % log_every == 0:
                    batch_loss = loss.item()
                    batch_acc = self.compute_accuracy(outputs, labels)
                    summary = {'batch_loss': batch_loss,
                               'batch_acc': batch_acc}
                    self.log_values(summary, train_samples_seen)

                    status = 'epoch [{}/{}] step [{}/{}]'.format(epoch + 1,
                                                                 num_epochs,
                                                                 idx + 1,
                                                                 num_batches)
                    status += ' batch_loss: {:.5f} batch_acc: {:.5f}'.format(
                              batch_loss, batch_acc)
                    progressbar.set_description(status)

    def forward_pass(self, inputs, labels):
        """Perform forward pass, return model output and loss.

        Parameters:
        -----------
        inputs : (torch.Tensor)
            Tensor of size (batch_size, ...) containing the inputs
        labels : (torch.Tensor)
            Tensor of size (batch_size, 1) holding the labels

        Returns:
        --------
        outputs : (torch.Tensor)
            Tensor of size (batch_size, ...) storing the model outputs
        loss : (torch.Tensor)
            Scalar value of the loss function evaluated on the outputs
        """
        # reshape and load to device
        batch_size = inputs.size()[0]
        inputs = inputs.view(batch_size, -1)
        # forward pass
        outputs = self.model(inputs)
        loss = self.loss_function(outputs, labels)
        return outputs, loss

    def backward_pass(self, outputs, loss):
        """Perform backward pass.

        Parameters:
        -----------
        outputs : (torch.Tensor)
            Tensor of size (batch_size, ...) storing the model outputs
        loss : (torch.Tensor)
            Scalar value of the loss function evaluated on the outputs
        """
        loss.backward()

    def create_logger(self, logdir):
        """Instantiate and return a logger for logdir.

        Parameters:
        -----------
        logdir : (str)
            Path to the log directory

        Raises:
        -------
        (FileExistsError)
            If logdir already exists.
        """
        # if path.isdir(logdir):
        #    raise FileExistsError('Logdir {} already exists!'.format(logdir))
        return Logger(logdir)

    def load_training_set(self):
        """Return the train_loader."""
        return self.data_loader.train_loader()

    def load_test_set(self):
        """Return the test_loader."""
        return self.data_loader.test_loader()

    @staticmethod
    def compute_accuracy(outputs, labels):
        """Compute accuracy given the networks' outputs and labels.

        Parameters:
        -----------
        outputs : (torch.Tensor)
            Outputs of the model with dimensions (batch_size, ...)
        labels : (torch.Tensor)
            Target labels of dimension (batch_size, 1)

        Returns:
        --------
        accuracy : (float)
            Ratio of correctly classified inputs
        """
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        return correct / total


class SecondOrderTraining(FirstOrderTraining):
    """Handle logging in training procedure with 2nd-order optimizers."""
    _modify_2nd_order = None

    def run(self,
            num_epochs,
            modify_2nd_order_terms,
            device=torch.device('cpu'),
            logs_per_epoch=10):
        """Run training, log values to logdir.

        Parameters:
        -----------
        num_epochs : (int)
            Number of epochs to train on
        modify_2nd_order_terms : (str) ('none', 'clip', 'sign', 'zero')
            String specifying the strategy for dealing with 2nd-order
            terms during Hessian backpropagation
        device : (torch.Device)
           Device to run the training on
        logs_per_epoch : (int)
            Number of datapoints written into logdir per epoch
        """
        try:
            # modify Hessian backward mode
            self.__class__._modify_2nd_order = modify_2nd_order_terms
            super().run(num_epochs,
                        device=device,
                        logs_per_epoch=logs_per_epoch)
        except Exception as e:
            raise e
        finally:
            # reset Hessian backward mode
            self.__class._modify_2nd_order = None

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
        output_hessian = bpexts.hessian.loss.batch_summed_hessian(loss, outputs)
        # compute gradients
        super().backward_pass(outputs, loss)
        loss.backward()

        # backward Hessian
        mod = self.__class__._modify_2nd_order_terms
        self.model.backward_hessian(output_hessian,
                                    compute_input_hessian=False,
                                    modify_2nd_order_terms=mod)

 
    # TODO
    # def loss_and_accuracy_on_test_set(self):
    #    """Evaluate loss and accuracy on the entire test set.
    #
    #    Returns:
    #    --------
    #    loss, accuracy : (float, float)
    #        Loss evaluated on the entire test set and ratio
    #        of correctly classified test examples
    #    """
    #    test_loader = self.load_test_set()
    #    if not len(test_loader) == 1:
    #        raise NotImplementedError('Test loss/accuracy currently only'
    #                                  ' supported in unbatched mode')
    #    with torch.no_grad():
    #        (inputs, labels) = next(iter(test_loader))
    #        total = labels.size(0)
    #        outputs = self.model(inputs.view(total, -1))
    #        loss = self.loss_function(outputs, labels).item()
    #        _, predicted = torch.max(outputs.data, 1)
    #        return loss, self.compute_accuracy(outputs, labels)
