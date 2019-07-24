"""Test batch gradient computation of linear layer."""
from backpack import extend
from torch import allclose, randn, randint, manual_seed, cat
from torch.autograd import grad
from torch.nn import Linear, Sequential, CrossEntropyLoss, Conv2d
from backpack.core.layers import LinearConcat, Conv2dConcat, Flatten

# Linear


def data():
    N = 5
    Ds = [20, 10, 3]

    X = randn(N, Ds[0])
    Y = randint(high=Ds[-1], size=(N, ))

    manual_seed(0)
    model1 = Sequential(
        extend(Linear(Ds[0], Ds[1])), extend(Linear(Ds[1], Ds[2])))

    manual_seed(0)
    model2 = Sequential(
        extend(LinearConcat(Ds[0], Ds[1])), extend(LinearConcat(Ds[1], Ds[2])))

    loss = CrossEntropyLoss()

    return X, Y, model1, model2, loss


def test_LinearConcat_forward():
    X, Y, model1, model2, loss = data()
    assert allclose(model1(X), model2(X))


def test_LinearConcat_backward():
    X, Y, model1, model2, loss = data()

    d1 = grad(loss(model1(X), Y), model1.parameters())
    d2 = grad(loss(model2(X), Y), model2.parameters())

    d1 = list(d1)
    d2 = list(d2)

    d1_cat = list()

    # take grad of separated parameters and concat them
    for i in range(len(d2)):
        d1_cat.append(cat([
            d1[2 * i],
            d1[2 * i + 1].unsqueeze(-1),
        ], dim=1))

    for p1, p2 in zip(d1_cat, d2):
        assert allclose(p1, p2)


# Conv
TEST_SETTINGS = {
    "in_features": (3, 4, 5),
    "out_channels": 3,
    "kernel_size": (3, 2),
    "padding": (1, 1),
    "bias": True,
    "batch": 3,
    "rtol": 1e-5,
    "atol": 5e-4
}


def convlayer(join_params):
    conv_cls = Conv2dConcat if join_params else Conv2d
    return extend(
        conv_cls(
            in_channels=TEST_SETTINGS["in_features"][0],
            out_channels=TEST_SETTINGS["out_channels"],
            kernel_size=TEST_SETTINGS["kernel_size"],
            padding=TEST_SETTINGS["padding"],
            bias=TEST_SETTINGS["bias"]))


def convlayer2(join_params):
    conv_cls = Conv2dConcat if join_params else Conv2d
    return extend(
        conv_cls(
            in_channels=TEST_SETTINGS["in_features"][0],
            out_channels=TEST_SETTINGS["out_channels"],
            kernel_size=TEST_SETTINGS["kernel_size"],
            padding=TEST_SETTINGS["padding"],
            bias=TEST_SETTINGS["bias"]))


def data_conv():
    input_size = (TEST_SETTINGS["batch"], ) + TEST_SETTINGS["in_features"]

    temp_model = Sequential(convlayer(False), convlayer2(False), Flatten())

    X = randn(size=input_size)
    Y = randint(high=X.shape[1], size=(temp_model(X).shape[0], ))

    del temp_model

    manual_seed(0)
    model1 = Sequential(convlayer(False), convlayer2(False), Flatten())

    manual_seed(0)
    model2 = Sequential(convlayer(True), convlayer2(True), Flatten())

    loss = CrossEntropyLoss()

    return X, Y, model1, model2, loss


def test_Conv2dConcat_forward():
    X, Y, model1, model2, loss = data_conv()
    assert allclose(model1(X), model2(X))


def test_Conv2dConcat_backward():
    X, Y, model1, model2, loss = data_conv()

    d1 = grad(loss(model1(X), Y), model1.parameters())
    d2 = grad(loss(model2(X), Y), model2.parameters())

    d1 = list(d1)
    d2 = list(d2)

    d1_cat = list()

    # take grad of separated parameters and concat them
    for i in range(len(d2)):
        d1_cat.append(
            cat(
                [
                    # require view because concat stores kernel as 2d tensor
                    d1[2 * i].view(d1[2 * i].shape[0], -1),
                    d1[2 * i + 1].unsqueeze(-1),
                ],
                dim=1))

    for p1, p2 in zip(d1_cat, d2):
        assert allclose(p1, p2)
