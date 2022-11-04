"""Test cases for BackPACK extensions for the GGN diagonal.

Includes
- ``DiagGGNExact``
- ``DiagGGNMC``
- ``BatchDiagGGNExact``
- ``BatchDiagGGNMC``

Shared settings are taken from `test.extensions.secondorder.secondorder_settings`.
Additional local cases can be defined here through ``LOCAL_SETTINGS``.
"""
from test.converter.resnet_cases import ResNet1, ResNet2
from test.core.derivatives.utils import classification_targets, regression_targets
from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS
from test.utils.evaluation_mode import initialize_training_false_recursive

from torch import rand, randint
from torch.nn import (
    LSTM,
    RNN,
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Conv2d,
    CrossEntropyLoss,
    Embedding,
    Flatten,
    Identity,
    Linear,
    MaxPool2d,
    MSELoss,
    ReLU,
    Sequential,
    Sigmoid,
)

from backpack import convert_module_to_backpack
from backpack.custom_module.branching import Parallel
from backpack.custom_module.permute import Permute
from backpack.custom_module.reduce_tuple import ReduceTuple

SHARED_SETTINGS = SECONDORDER_SETTINGS
LOCAL_SETTINGS = []
##################################################################
#                         RNN settings                           #
##################################################################
LOCAL_SETTINGS += [
    # RNN settings
    {
        "input_fn": lambda: rand(8, 5, 6),
        "module_fn": lambda: Sequential(
            RNN(input_size=6, hidden_size=3, batch_first=True),
            ReduceTuple(index=0),
            Permute(0, 2, 1),
            Flatten(),
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((8, 3 * 5)),
    },
    {
        "input_fn": lambda: rand(4, 3, 5),
        "module_fn": lambda: Sequential(
            LSTM(input_size=5, hidden_size=4, batch_first=True),
            ReduceTuple(index=0),
            Flatten(),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((4,), 4 * 3),
    },
    {
        "input_fn": lambda: rand(8, 5, 6),
        "module_fn": lambda: Sequential(
            RNN(input_size=6, hidden_size=3, batch_first=True),
            ReduceTuple(index=0),
            Linear(3, 3),
            Permute(0, 2, 1),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((8, 5), 3),
    },
]
##################################################################
#                AdaptiveAvgPool settings                        #
##################################################################
LOCAL_SETTINGS += [
    {
        "input_fn": lambda: rand(2, 2, 9),
        "module_fn": lambda: Sequential(
            Linear(9, 9), AdaptiveAvgPool1d((3,)), Flatten()
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((2, 2 * 3)),
    },
    {
        "input_fn": lambda: rand(2, 2, 6, 8),
        "module_fn": lambda: Sequential(
            Linear(8, 8), AdaptiveAvgPool2d((3, 4)), Flatten()
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((2, 2 * 3 * 4)),
    },
    {
        "input_fn": lambda: rand(2, 2, 9, 5, 4),
        "module_fn": lambda: Sequential(
            Linear(4, 4), AdaptiveAvgPool3d((3, 5, 2)), Flatten()
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((2, 2 * 3 * 5 * 2)),
    },
]
##################################################################
#                      BatchNorm settings                        #
##################################################################
LOCAL_SETTINGS += [
    {
        "input_fn": lambda: rand(2, 3, 4),
        "module_fn": lambda: initialize_training_false_recursive(
            Sequential(BatchNorm1d(num_features=3), Flatten())
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((2, 4 * 3)),
    },
    {
        "input_fn": lambda: rand(3, 2, 4, 3),
        "module_fn": lambda: initialize_training_false_recursive(
            Sequential(BatchNorm2d(num_features=2), Flatten())
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((3, 2 * 4 * 3)),
    },
    {
        "input_fn": lambda: rand(3, 3, 4, 1, 2),
        "module_fn": lambda: initialize_training_false_recursive(
            Sequential(BatchNorm3d(num_features=3), Flatten())
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((3, 3 * 4 * 1 * 2)),
    },
    {
        "input_fn": lambda: rand(3, 3, 4, 1, 2),
        "module_fn": lambda: initialize_training_false_recursive(
            Sequential(
                BatchNorm3d(num_features=3),
                Linear(2, 3),
                BatchNorm3d(num_features=3),
                ReLU(),
                BatchNorm3d(num_features=3),
                Flatten(),
            )
        ),
        "loss_function_fn": lambda: MSELoss(),
        "target_fn": lambda: regression_targets((3, 4 * 1 * 3 * 3)),
    },
]
###############################################################################
#                               Embedding                                     #
###############################################################################
LOCAL_SETTINGS += [
    {
        "input_fn": lambda: randint(0, 5, (6,)),
        "module_fn": lambda: Sequential(
            Embedding(5, 3),
            Linear(3, 4),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((6,), 4),
    },
    {
        "input_fn": lambda: randint(0, 3, (3, 2, 2)),
        "module_fn": lambda: Sequential(
            Embedding(3, 2),
            Flatten(),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(reduction="mean"),
        "target_fn": lambda: classification_targets((3,), 2 * 2),
        "seed": 2,
    },
]


###############################################################################
#                               Branched models                               #
###############################################################################
LOCAL_SETTINGS += [
    {
        "input_fn": lambda: rand(3, 10),
        "module_fn": lambda: Sequential(
            Linear(10, 5),
            ReLU(),
            # skip connection
            Parallel(
                Identity(),
                Linear(5, 5),
            ),
            # end of skip connection
            Sigmoid(),
            Linear(5, 4),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((3,), 4),
        "id_prefix": "branching-linear",
    },
    {
        "input_fn": lambda: rand(4, 2, 6, 6),
        "module_fn": lambda: Sequential(
            Conv2d(2, 3, kernel_size=3, stride=1, padding=1),
            ReLU(),
            # skip connection
            Parallel(
                Identity(),
                Sequential(
                    Conv2d(3, 5, kernel_size=3, stride=1, padding=1),
                    ReLU(),
                    Conv2d(5, 3, kernel_size=3, stride=1, padding=1),
                ),
            ),
            # end of skip connection
            MaxPool2d(kernel_size=3, stride=2),
            Flatten(),
            Linear(12, 5),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((4,), 5),
        "id_prefix": "branching-convolution",
    },
    {
        "input_fn": lambda: rand(4, 3, 6, 6),
        "module_fn": lambda: Sequential(
            Conv2d(3, 2, kernel_size=3, stride=1, padding=1),
            ReLU(),
            # skip connection
            Parallel(
                Identity(),
                Sequential(
                    Conv2d(2, 4, kernel_size=3, stride=1, padding=1),
                    Sigmoid(),
                    Conv2d(4, 2, kernel_size=3, stride=1, padding=1),
                    Parallel(
                        Identity(),
                        Sequential(
                            Conv2d(2, 4, kernel_size=3, stride=1, padding=1),
                            ReLU(),
                            Conv2d(4, 2, kernel_size=3, stride=1, padding=1),
                        ),
                    ),
                ),
            ),
            # end of skip connection
            MaxPool2d(kernel_size=3, stride=2),
            Flatten(),
            Linear(8, 5),
        ),
        "loss_function_fn": lambda: CrossEntropyLoss(),
        "target_fn": lambda: classification_targets((4,), 5),
        "id_prefix": "nested-branching-convolution",
    },
]

###############################################################################
#                      Branched models - converter                            #
###############################################################################
LOCAL_SETTINGS += [
    {
        "input_fn": lambda: ResNet1.input_test,
        "module_fn": lambda: convert_module_to_backpack(ResNet1(), True),
        "loss_function_fn": lambda: ResNet1.loss_test,
        "target_fn": lambda: ResNet1.target_test,
        "id_prefix": "ResNet1",
    },
    {
        "input_fn": lambda: rand(ResNet2.input_test),
        "module_fn": lambda: convert_module_to_backpack(ResNet2().eval(), True),
        "loss_function_fn": lambda: ResNet2.loss_test,
        "target_fn": lambda: rand(ResNet2.target_test),
        "id_prefix": "ResNet2",
    },
]

DiagGGN_SETTINGS = SHARED_SETTINGS + LOCAL_SETTINGS
