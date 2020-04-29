'''
This setup can be used to define the problems which need testing.

SETUPS: A list which contains a dictonary for each instance we want to test.
Format:
[
	{
        "module_cls": A torch.nn module,
        "module_kwargs": Inputs which are required by the torch.nn module
        				 type: dict
        "input_shape": Input shape respectively
        			   type: Tuple
        "N": Batch_size
        	 type: int
		
	}]

'''



import torch
from test.core.derivatives.utils import get_available_devices


class DerivativesModule:
    """ 
    Information required to test a class inheriting from
    `backpack.core.derivatives.BaseDerivatives`.

    batch_size : N
    input_shape : [C_in, H_in, W_in, ...]
    module: Torch module used (Eg: torch.nn.Linear)

    Shape of module input → output:
      [N, C_in, H_in, W_in, ...] → [N, C_out, H_out, W_out, ...]

    """

    def __init__(self, module, N, input_shape, device):
        self.module = module
        self.derivative = None
        self.N = N
        self.input_shape = input_shape
        self.device = device


def set_up_derivatives_module(setup, device):
    """Create a DerivativesModule object from setup.
	
	Input:
		setup: Dictonary containing information pertaining torch.nn module to be tested
		device: cpu/gpu

	Return: 
		DerivativesModule: Returns a class with all the information pertaining torch.nn 
						   module obtained from setup
	
    """
    module_cls = setup["module_cls"]
    module_kwargs = setup["module_kwargs"]
    N = setup["N"]
    input_shape = setup["input_shape"]

    module = module_cls(**module_kwargs)
    derivatives_module_id = "{}-{}-N={}-input_shape={}".format(
        device, module, N, input_shape
    )

    return DerivativesModule(module, N, input_shape, device), derivatives_module_id


SETUPS = [
    {
        "module_cls": torch.nn.Linear,
        "module_kwargs": {"in_features": 7, "out_features": 3, "bias": True,},
        "input_shape": (7,),
        "N": 10,
    },
    
    {
        "module_cls": torch.nn.ReLU,
        "module_kwargs": {},
        "input_shape": (10,5),
        "N": 1,
    },

    {
        "module_cls": torch.nn.Tanh,
        "module_kwargs": {},
        "input_shape": (1,5,6),
         "N": 1,
    },

    {
        "module_cls": torch.nn.Sigmoid,
        "module_kwargs": {},
        "input_shape": (1,5),
        "N": 1,
    },

    {
        "module_cls": torch.nn.Conv2d,
        "module_kwargs": {"in_channels":3,"out_channels":3,"kernel_size":4, "stride":1},
        "input_shape": (3,32,32),
        "N": 1,
    },

    {
        "module_cls": torch.nn.MaxPool2d,
        "module_kwargs": {"kernel_size":2, "stride":2},
        "input_shape": (5,32,32),
        "N": 1,
    },

    {
        "module_cls": torch.nn.AvgPool2d,
        "module_kwargs": {"kernel_size":2},
        "input_shape": (3,32,32),
        "N": 1,
    },

]


DEVICES = get_available_devices()

ALL_CONFIGURATIONS = []
CONFIGURATION_IDS = []

for setup in SETUPS:
    for device in DEVICES:
        derivatives_module, derivatives_module_id = set_up_derivatives_module(
            setup, device
        )

        ALL_CONFIGURATIONS.append(derivatives_module)
        CONFIGURATION_IDS.append(derivatives_module_id)

