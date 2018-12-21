from torchvision import datasets, transforms
import torch


def make_loader(dataset, batch_size, use_cuda):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, *kwargs)


def load_MNIST(batch_size=128, use_cuda=False, path='../dat'):
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(path, train=True, download=True, transform=normalize)
    return make_loader(dataset, batch_size, use_cuda)
