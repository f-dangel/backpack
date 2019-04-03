"""Investigations of the Hessiasn of a FCNN on MNIST.

Plot the Hessians during HBP for an MNIST image as .pdf file."""

import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from exp.models.fcnn import fcnn
from exp.loading.load_mnist import MNISTLoader

from exp03_c1d1_hessian import (
    hbp_passed_hessians, parameter_hessians_layerwise, imshow_hessian,
    save_input_images, save_passed_hessians, save_parameter_hessians_layerwise)
from bpexts.utils import (set_seeds, boxed_message)
from exp.utils import (directory_in_data, directory_in_fig)
from os import path, makedirs, remove
from exp.training.runner import TrainingRunner
from exp.training.first_order import FirstOrderTraining
from torch.optim import SGD
import glob

# directory names for figures and training logging
dirname = 'exp04_fcnn_hessian'
fig_dir = directory_in_fig(dirname)
data_dir = directory_in_data(dirname)


def fcnn_model():
    """Create the model used in this experiment."""
    return fcnn(input_size=(1, 28, 28), hidden_dims=[15], num_outputs=10)


def trained_fcnn_model():
    """Return trained FCNN model on MNIST."""
    # clean data from previous runs
    old_logs = glob.glob(path.join(data_dir, "events.out.tfevents.*"))
    for file in old_logs:
        print(boxed_message("Remove old log {}".format(file)))
        remove(file)

    # set up training function
    model = fcnn_model()
    loss_function = CrossEntropyLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_loader = MNISTLoader(500, 500)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    num_epochs, logs_per_epoch = 10, 5
    # initialize training
    training = FirstOrderTraining(
        model,
        loss_function,
        optimizer,
        data_loader,
        data_dir,
        num_epochs,
        logs_per_epoch=logs_per_epoch,
        device=device,
        input_shape=(1, 28, 28))

    # run training
    training.run()
    return training.model.cpu()


def main():
    """Create figures of quantities involved in the HBP of ``c1d1```."""
    # create figure directory
    print(boxed_message("Creating figures in {}".format(fig_dir)))
    makedirs(fig_dir, exist_ok=True)

    # load inputs
    set_seeds(0)
    model = fcnn_model()
    loss_fn = CrossEntropyLoss()
    train_loader = MNISTLoader(1, 1).train_loader()
    input = next(iter(train_loader))
    input[0].requires_grad = True

    # Untrained model
    #################

    # save images of the input
    save_input_images(input[0], fig_dir)

    # save Hessians backpropagated through the graph
    save_passed_hessians(
        model, loss_fn, input, fig_dir, prefix="untrained_passed_hessian")

    # save layerwise parameter Hessians
    save_parameter_hessians_layerwise(
        model, loss_fn, input, fig_dir, prefix="untrained_parameter_hessian")

    # Trained model
    ###############
    set_seeds(0)
    trained_model = trained_fcnn_model()

    # save Hessians backpropagated through the graph
    save_passed_hessians(
        trained_model,
        loss_fn,
        input,
        fig_dir,
        prefix="trained_passed_hessian")

    # save layerwise parameter Hessians
    save_parameter_hessians_layerwise(
        trained_model,
        loss_fn,
        input,
        fig_dir,
        prefix="trained_parameter_hessian")


if __name__ == "__main__":
    main()
