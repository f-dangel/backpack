
from test.automated_test import check_sizes, check_values
import torch


def get_available_devices():
    devices = [torch.device("cpu")]

    if torch.cuda.is_available():
        devices.add(torch.device("cuda"))

    return devices