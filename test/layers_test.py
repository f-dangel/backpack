"""Test batch gradient computation of linear layer."""
from backpack import extend
from torch import allclose, randn, randint, manual_seed, cat
from torch.autograd import grad
from torch.nn import Linear, Sequential, CrossEntropyLoss
from backpack.core.layers import LinearConcat


def data():
    N = 5
    Ds = [20, 10, 3]

    X = randn(N, Ds[0])
    Y = randint(high=Ds[-1], size=(N,))

    manual_seed(0)
    model1 = Sequential(
        extend(Linear(Ds[0], Ds[1])),
        extend(Linear(Ds[1], Ds[2]))
    )

    manual_seed(0)
    model2 = Sequential(
        extend(LinearConcat(Ds[0], Ds[1])),
        extend(LinearConcat(Ds[1], Ds[2]))
    )

    loss = CrossEntropyLoss()

    return X, Y, model1, model2, loss


def test_LinearCat_forward():
    X, Y, model1, model2, loss = data()
    assert allclose(model1(X), model2(X))


def test_LinearCat_backward():
    X, Y, model1, model2, loss = data()

    d1 = grad(loss(model1(X), Y), model1.parameters())
    d2 = grad(loss(model2(X), Y), model2.parameters())

    d1 = list(d1)
    d2 = list(d2)

    d1_cat = list()

    for i in range(len(d2)):
        d1_cat.append(cat([
            d1[2 * i],
            d1[2 * i + 1].unsqueeze(-1),
        ], dim=1))

    for p1, p2 in zip(d1_cat, d2):
        assert allclose(p1, p2)
