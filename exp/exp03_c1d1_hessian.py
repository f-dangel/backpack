"""Investigation of the Hessians of the c1d1 CNN.

Plot the Hessians during HBP for an MNIST image as .pdf files.
"""

import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from exp.models.convolution import c1d1
from exp.loading.load_mnist import MNISTLoader
from bpexts.hessian.exact import exact_hessian
from bpexts.utils import (set_seeds, boxed_message)
from exp.utils import (directory_in_data, directory_in_fig)
from os import path, makedirs, remove
from exp.training.runner import TrainingRunner
from exp.training.first_order import FirstOrderTraining
from torch.optim import SGD
import glob

# directory names for figures and training logging
dirname = 'exp03_c1d1_hessian'
fig_dir = directory_in_fig(dirname)
data_dir = directory_in_data(dirname)


def all_children(model):
    """Yield children and sub-children of the entire model.

    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model

    Yields:
    -------
    torch.nn.Module
        The modules contained within ``model``, and their children
    """
    for mod in model.children():
        # there are sub-children
        if len(list(mod.children())) != 0:
            for sub_mod in all_children(mod):
                yield sub_mod
        # no sub-children
        else:
            yield mod


def num_total_parameters(params):
    """Compute the total number of elements.

    Parameters:
    -----------
    params : iter(torch.Tensor)
        Iterator over tensors whose number of elements will be summed.
    """
    return sum([p.numel() for p in params])


def hbp_passed_hessians(model, loss_fn, input, max_size=10000):
    """Yield the Hessians passed backward through the model during HBP.

    Skip Hessians with dimension exceeding ``max_size``.

    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch model.
    loss_fn : torch.nn.Module
        Loss function applied on the outputs of the model.
    input : tuple(torch.Tensor, torch.Tensor)
        Network inputs and labels
    max_size : int
        Maximum size of the Hessian, will be skipped if exceeded.

    Yields:
    -------
    torch.Tensor, str
        Hessian with respect to the module input and name of the module.
    """
    (x, y) = input
    intermediates = []
    # forward pass
    for mod in all_children(model):
        intermediates.append((x, mod.__class__.__name__))
        x = mod(x)
    intermediates.append((x, loss_fn.__class__.__name__))
    loss = loss_fn(x, y)
    # compute the Hessians
    for x, name in intermediates:
        if x.numel() < max_size:
            yield exact_hessian(loss, [x], show_progress=False).detach(), name


def parameter_hessians_layerwise(model, loss_fn, input, max_size=10000):
    """Yield the Hessian blocks for parameters within one layer.

    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model.
    loss_fn : torch.nn.Module
        Loss function applied to the model outputs.
    input : tuple(torch.Tensor, torch.Tensor)
        Input and labels.
    max_size : int
        Maximum size of the Hessians that will be computed.

    Yields:
    -------
    torch.Tensor, str
        Hessian of the parameters in one layer, along with the layer name
    """
    (x, y) = input
    out = model(x)
    loss = loss_fn(out, y)
    for mod in all_children(model):
        if len(list(mod.parameters())) != 0:
            name = mod.__class__.__name__
            if num_total_parameters(mod.parameters()) < max_size:
                yield exact_hessian(
                    loss, mod.parameters(), show_progress=False).detach(), name


def full_parameter_hessian(model, loss_fn, input):
    """Compute the full parameter Hessian of ``loss_fn``.

    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model.
    loss_fn : torch.nn.Module
        Loss function that will be applied on the model outputs.
    input : tuple(torch.Tensor, torch.Tensor)
        Input and labels.

    Returns:
    --------
    torch.Tensor
        Hessian of vectorized concatenated parameters.
    """
    (x, y) = input
    out = model(x)
    loss = loss_fn(out, y)
    return exact_hessian(
        loss, model.parameters(), show_progress=False).detach(), name


def imshow_hessian(hessian, log_abs=True, magnitudes=3):
    """Plot the Hessian.

    Optionally show the absolute values on a logarithmic clipped scale.
    Clip to the first ``show_magnitudes`` orders.

    Parameters:
    -----------
    hessian : 2d torch.Tensor
        The Hessian matrix to be shown.
    log_abs : bool
        Plot absolute values of the Hessian in logarithmic scale.
    magnitudes : float/int
        (Only if using ``log_abs``) Range of the logarithmic axis.
    """

    def preprocess(h):
        """Take absolute log-values and perform clipping if desired."""
        if log_abs == True:
            h_abs = h.abs()
            max_val = h.max()
            low = max_val / 10**magnitudes
            h_clip = h_abs.clamp(low)
            return h_clip.log10()
        else:
            return h

    hessian_plot = preprocess(hessian)
    plt.imshow(hessian_plot.numpy())


def c1d1_model():
    """Create the model used in this experiment."""
    return c1d1(
        input_size=(1, 28, 28),
        num_outputs=10,
        conv_channels=8,
        kernel_size=5,
        padding=0,
        stride=1)


def trained_c1d1_model():
    """Return trained c1d1 model on MNIST."""
    # clean data from previous runs
    old_logs = glob.glob(path.join(data_dir, "events.out.tfevents.*"))
    for file in old_logs:
        print(boxed_message("Remove old log {}".format(file)))
        remove(file)

    # set up training function
    model = c1d1_model()
    loss_function = CrossEntropyLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_loader = MNISTLoader(500, 500)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
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


def save_input_images(x, directory, prefix="input_image"):
    """Save pdf images of each batch instance in ``x``.

    Parameters:
    -----------
    x : 3d or 4d torch.Tensor
        Input data, first dimension is batch size. The image itself
        can be a 2d or 3d tensor.
    directory : str
        String to the directory where the figures will be saved.
    prefix : str
        Prefix for the filename of the images.
    """
    for idx in range(x.size()[0]):
        filename = path.join(directory, '{}_{}.pdf'.format(prefix, idx))
        input_image = x[idx, :].clone().detach().squeeze()
        if len(input_image.size()) == 3:
            input_image = input_image.numpy().transpose(1, 2, 0)
        # show and save
        plt.figure()
        plt.imshow(input_image)
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


def save_passed_hessians(model,
                         loss_fn,
                         input,
                         directory,
                         prefix="passed_hessian",
                         max_size=10000,
                         log_abs=True,
                         magnitudes=3,
                         colorbar=True):
    """Compute Hessians backpropagated during HBP and save as .pdf.

    Parameters:
    -----------
    directory : str
        String to the directory where the figures will be stored.
    prefix : str
        Prefix for the filenames of the figures.
    colorbar : True
        Show a colorbar.
    """
    for idx, (hessian, name) in enumerate(
            hbp_passed_hessians(model, loss_fn, input, max_size=max_size)):

        filename = path.join(directory, '{}_{}_{}.pdf'.format(
            prefix, idx, name))

        plt.figure()
        imshow_hessian(hessian, log_abs=log_abs, magnitudes=magnitudes)
        if colorbar == True:
            plt.colorbar()
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


def save_parameter_hessians_layerwise(model,
                                      loss_fn,
                                      input,
                                      directory,
                                      prefix="parameter_hessian",
                                      max_size=10000,
                                      log_abs=True,
                                      magnitudes=3,
                                      colorbar=True):
    """Compute Parameter Hessians and save as .pdf.

    Parameters:
    -----------
    directory : str
        String to the directory where the figures will be stored.
    prefix : str
        Prefix for the filenames of the figures.
    colorbar : True
        Show a colorbar.
    """
    for idx, (hessian, name) in enumerate(
            parameter_hessians_layerwise(model, loss_fn, input)):
        filename = path.join(directory, '{}_{}_{}.pdf'.format(
            prefix, idx, name))
        plt.figure()
        imshow_hessian(hessian, log_abs=log_abs, magnitudes=magnitudes)
        plt.colorbar()
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


def main():
    """Create figures of quantities involved in the HBP of ``c1d1```."""
    # create figure directory
    print(boxed_message("Creating figures in {}".format(fig_dir)))
    makedirs(fig_dir, exist_ok=True)

    # load inputs
    set_seeds(0)
    model = c1d1_model()
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
    trained_model = trained_c1d1_model()

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
