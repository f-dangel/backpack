import inspect
from backpack import BackpropExtension


def _check_extensions(func_name, extensions):
    for ext in extensions:
        if not isinstance(ext, BackpropExtension):
            if inspect.isclass(ext) and issubclass(ext, BackpropExtension):
                raise ValueError(
                    "{} expect instances of BackpropExtension,".format(func_name)
                    + " but received a class instead [{}].".format(ext)
                    + " Instantiate it before passing it to backpack."
                )
            else:
                raise ValueError(
                    "{} expects instances of BackpropExtension,".format(func_name)
                    + " but received [{}].".format(ext)
                )


def zero_backpack(model, *extensions):
    """Clears the backpack-computed quantities of the model.

    Removes references from the model to quantities computed by BackPACK
    using the given extensions. Can be used to free memory or remove additional
    tensors from model parameters for saving data with pickle or deepcopy.

    Args:
        model: A :py:class:`Module <torch.nn.Module>`
        *extensions: Instances of BackpropExtensions that have been called
    """
    _check_extensions("zero_backpack", extensions)

    for p in model.parameters():
        for ext in extensions:
            if hasattr(p, ext.savefield):
                delattr(p, ext.savefield)
