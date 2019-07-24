import torch
import torchvision
from models import VGG16Sequential, VGG19Sequential

x = torch.randn(1, 3, 224, 224)


def compare_vgg16():
    vgg_torch = torchvision.models.vgg16()
    vgg_deepobs = VGG16Sequential(1000)
    print("\nVGG16")
    print(78 * "=")

    # forward pass to set padding value correctly
    out = vgg_deepobs(x)

    print_models(vgg_torch, vgg_torch)


def compare_vgg19():
    vgg_torch = torchvision.models.vgg19()
    vgg_deepobs = VGG19Sequential(1000)
    print("\nVGG19")
    print(78 * "=")
    # forward pass to set padding value correctly
    out = vgg_deepobs(x)

    print_models(vgg_torch, vgg_torch)


def print_models(model_deepobs, model_pytorch):
    print("Sequential inspired from DeepOBS\n")
    print(model_deepobs)

    print(78 * "=")
    print("Model from torchvision\n")
    print(model_pytorch)


if __name__ == "__main__":
    compare_vgg16()
    compare_vgg19()
