# <img alt="BackPACK" src="./logo/backpack_logo_no_torch.svg" height="90"> BackPACK: Packing more into backprop

[![Travis](https://travis-ci.org/f-dangel/backpack.svg?branch=master)](https://travis-ci.org/f-dangel/backpack)
[![Coveralls](https://coveralls.io/repos/github/f-dangel/backpack/badge.svg?branch=master)](https://coveralls.io/github/f-dangel/backpack)

BackPACK is built on top of [PyTorch](https://github.com/pytorch/pytorch). It efficiently computes quantities other than the gradient.

- **Website:** https://backpack.pt
- **Documentation:** https://readthedocs.org/projects/backpack/
- **Bug reports & feature requests:** https://github.com/f-dangel/backpack/issues

Provided quantities include:
- Individual gradients from a mini-batch
- Estimates of the gradient variance or second moment
- Approximate second-order information (diagonal and Kronecker approximations)

## Installation
```bash
pip install backpack-for-pytorch
```

## Getting started

- Check out the [cheatsheet](examples/cheatsheet.pdf) for an overview of quantities.
- Check out the [examples](https://f-dangel.github.io/backpack/) on how to use the code.

## Contributing

BackPACK is actively being developed. 
We are appreciating any help.
If you are considering to contribute, do not hesitate to contact us.
An overview of the development procedure is provided in the [developer `README`](https://github.com/f-dangel/backpack/blob/master/README-dev.md).

## How to cite
If you are using BackPACK, consider citing the [paper](https://openreview.net/forum?id=BJlrF24twB) 
```
@inproceedings{dangel2020backpack,
    title     = {Back{PACK}: Packing more into Backprop},
    author    = {Felix Dangel and Frederik Kunstner and Philipp Hennig},
    booktitle = {International Conference on Learning Representations},
    year      = {2020},
    url       = {https://openreview.net/forum?id=BJlrF24twB}
}
```

## Other interesting projects
Here is an alphabetical overview of interesting projects at the intersection of automatic differentiation and machine learning (without claiming completeness):
- [`autograd-hacks`](https://github.com/cybertronai/autograd-hacks) (PyTorch)
- [DiffSharp](http://diffsharp.github.io/DiffSharp/) (F#)
- [`grad-cnns`](https://github.com/owkin/grad-cnns) (PyTorch)
- [`higher`](https://github.com/facebookresearch/higher) (PyTorch)
- [JAX](https://github.com/google/jax) (Python)
- [`pyhessian`](https://github.com/amirgholami/PyHessian) (PyTorch)
- [`pytorch-hessian-eigenthings`](https://github.com/noahgolmant/pytorch-hessian-eigenthings) (PyTorch)
- [Zygote](https://github.com/FluxML/Zygote.jl) (Julia)

