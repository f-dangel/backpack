# <img alt="BackPACK" src="./logo/backpack_logo_torch.svg" height="90"> BackPACK: Packing more into backprop

[![Travis](https://travis-ci.org/f-dangel/backpack.svg?branch=master)](https://travis-ci.org/f-dangel/backpack)
[![Coveralls](https://coveralls.io/repos/github/f-dangel/backpack/badge.svg?branch=master)](https://coveralls.io/github/f-dangel/backpack)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)

BackPACK is built on top of [PyTorch](https://github.com/pytorch/pytorch). It efficiently computes quantities other than the gradient.

- **Website:** https://backpack.pt
- **Documentation:** https://docs.backpack.pt/en/master/
- **Bug reports & feature requests:** https://github.com/f-dangel/backpack/issues

Provided quantities include:
- Individual gradients from a mini-batch
- Estimates of the gradient variance or second moment
- Approximate second-order information (diagonal and Kronecker approximations)

**Motivation:** Computation of most quantities is not necessarily expensive (often just a small modification of the existing backward pass where backpropagated information can be reused). But it is difficult to do in the current software environment.


## Installation
```bash
pip install backpack-for-pytorch
```

## Examples
- [Basic usage](https://docs.backpack.pt/en/master/basic_usage/example_all_in_one.html)
- [Some use cases](https://docs.backpack.pt/en/master/use_cases/index.html)

#
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

###### _BackPACK is not endorsed by or affiliated with Facebook, Inc. PyTorch, the PyTorch logo and any related marks are trademarks of Facebook, Inc._

