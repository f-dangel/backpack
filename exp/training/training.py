"""Class for handling training and logging metrics."""

from abc import ABC, abstractmethod
from os import path
from tqdm import tqdm
import torch
import pandas
from .logger import Logger


class Training(ABC):
    """Handle training and logging."""
    def __init__(self,
                 model,
                 loss_function,
                 optimizer,
                 data_loader,
                 logdir,
                 num_epochs,
                 logs_per_epoch=10,
                 device=torch.device('cpu'),
                 ):
        """Train a model, log loss values into logdir.

        Parameters:
        -----------
        model : (torch.nn.Module)
            Trainable model
        loss_function : (torch.nn.Functional / torch.nn.Module)
            Scalar loss function computed on top of the model output
        optimizer : (torch.optim.Optimizer)
            Optimizer used for training
        data_loader : (DatasetLoader)
            Provides loaders for train and test set, see `load_dataset.py`
        logdir : (str)
            Path to the log directory
        num_epochs : (int)
            Number of epochs to train on
        logs_per_epoch : (int)
            Number of datapoints written into logdir per epoch
        device : (torch.Device)
           Device to run the training on
        """
        self.model = model.to(device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.logdir = logdir
        self.num_epochs = num_epochs
        self.logs_per_epoch = logs_per_epoch
        self.device = device
        # initialize during run or point_to
        self.logger = None

    def run(self, convert_to_csv=True):
        """Run training, log values to logdir and convert to csv.
        
        Parameters:
        -----------
        convert_to_csv : (bool)
            Convert the tensorboard logging file into a csv file
        """
        if self.logger is None:
            self.point_logger_to_subdir('')
        training_set = self.data_loader.train_loader()
        num_batches = len(training_set)
        log_every = max(num_batches // self.logs_per_epoch, 1)

        train_samples_seen = 0
        progressbar = tqdm(range(self.num_epochs))
        for epoch in progressbar:
            for idx, (inputs, labels) in enumerate(training_set):
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
                    test_loss, test_acc = self.test_loss_and_accuracy()

                    summary = {'batch_loss': batch_loss,
                               'batch_acc': batch_acc,
                               'test_loss': test_loss,
                               'test_acc': test_acc}
                    self.logger.log_scalar_values(summary, train_samples_seen)

                    status = 'epoch | step [{}/{}|{}/{}]'\
                             .format(epoch + 1,
                                     self.num_epochs,
                                     idx + 1,
                                     num_batches)
                    status += ' | batch_loss: {:.5f} batch_acc: {:.5f}'\
                              .format(batch_loss,
                                      batch_acc)
                    status += ' | test_loss: {:.5f} test_acc: {:.5f}'\
                              .format(test_loss,
                                      test_acc)
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
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
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
        total = labels.size()[0]
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        return correct / total

    def load_training_set(self):
        """Return the train_loader."""
        return self.data_loader.train_loader()

    def load_test_set(self):
        """Return the test_loader."""
        return self.data_loader.test_loader()

    def point_logger_to_subdir(self, subdir):
        """Point logger to subdirectory.

        Parameters:
        -----------
        subdir : (str)
            name of the subdirectory logging directory
        """
        self.logger = Logger(self.logdir, subdir)
        self.logger.print_tensorboard_instruction()

    def logger_subdir_path(self, subdir):
        """Return path to subdirectory of logger."""
        return Logger.subdir_path(self.logdir, subdir)

    def test_loss_and_accuracy(self):
        """Evaluate mean loss and accuracy on the entire test set.
    
        Returns:
        --------
        loss, accuracy : (float, float)
            Mean loss evaluated on the entire test set and ratio
            of correctly classified test examples
        """
        loss, correct = 0., 0.
        with torch.no_grad():
            for (inputs, labels) in self.load_test_set():
                batch = labels.size()[0]
                outputs, b_loss = self.forward_pass(inputs, labels)
                loss += batch * b_loss.item()
                correct += batch * self.compute_accuracy(outputs, labels)
        loss /= self.data_loader.test_set_size
        correct /= self.data_loader.test_set_size
        return loss, correct
