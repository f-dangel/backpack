# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [1.2.0] - 2020-10-26

Thanks to [@sbharadwajj](https://github.com/sbharadwajj) for
co-authoring many PRs shipped in this release.

### Added
- Deprecated `python3.5`, tested compatibility with PyTorch 1.6.0
  [[PR](https://github.com/f-dangel/backpack/pull/88)]
- Support first-order extensions for `Conv1d`, `Conv3d`,
  `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d`
  - `extensions.BatchGrad`
    [[PR](https://github.com/f-dangel/backpack/pull/92)]
  - `extensions.BatchL2Grad`
    [[PR](https://github.com/f-dangel/backpack/pull/100)]
  - `extensions.SumGradSquared` and `extensions.Variance`
    [[PR](https://github.com/f-dangel/backpack/pull/105)]
  - Raise exceptions for unsupported exotic hyperparameters
    [[PR1](https://github.com/f-dangel/backpack/pull/108),
    [PR2](https://github.com/f-dangel/backpack/pull/109)]
- New example: Backpropagating through BackPACK quantities
  [[commit](https://github.com/f-dangel/backpack/commit/8ef33a42badded9a1d9b5013f8686bfa7feec6e7)]
- New extensions in API: Block-diagonal curvature products
  - Exposed via `extensions.HMP`, `extensions.GGNMP`,
    `extensions.PCHMP`
    [[PR](https://github.com/f-dangel/backpack/pull/73)]
  - Examples: Hutchinson trace estimation
    [[PR](https://github.com/f-dangel/backpack/pull/98)]
    and  Hessian-free optimization with CG
    [[PR](https://github.com/f-dangel/backpack/pull/99)]
### Fixed

- Add missing `zero_grad` in the diagonal GGN second-order
  optimization example
  [[PR](https://github.com/f-dangel/backpack/pull/101)]

### Internal
- Increased test coverage
  - New test suite for `backpack.extensions`
    [[PR](https://github.com/f-dangel/backpack/pull/90)]
  - New test suite for `backpack.core`
    [[PR](https://github.com/f-dangel/backpack/pull/75)]
- Implemented derivatives of the following operations in
  `backpack.core`
  - More activation functions
    [[PR](https://github.com/f-dangel/backpack/pull/76)]
  - `Conv1d`, `Conv3d`
    [[PR](https://github.com/f-dangel/backpack/pull/79)]
  - `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d`
    [[PR](https://github.com/f-dangel/backpack/pull/84)]
- Refactor `firstorder` extensions to share more code
  [[PR1](https://github.com/f-dangel/backpack/pull/105),
  [PR2](https://github.com/f-dangel/backpack/pull/105)]
- Removed `detach`s to support differentiating through
  quantities
  [[PR](https://github.com/f-dangel/backpack/pull/70)]


## [1.1.1] - 2020-04-29

### Added
- Improved documentation, moved to [ReadTheDocs](https://docs.backpack.pt) 
  [[PR1](https://github.com/f-dangel/backpack/pull/57), 
  [PR2](https://github.com/f-dangel/backpack/pull/58),
  [PR3](https://github.com/f-dangel/backpack/pull/66)]
- Tested compatibility with PyTorch 1.5.0.
- Support 2nd-order backprop for vectors in `MSELoss` 
  [[PR](https://github.com/f-dangel/backpack/pull/61)]
- Sanity checks to raise warnings if the following are used.
  `inplace` modification 
  [[PR](https://github.com/f-dangel/backpack/pull/59)],
  unsupported loss parameters 
  [[PR](https://github.com/f-dangel/backpack/pull/60)],
  custom losses in 2nd-order backpropagation 
  [[PR](https://github.com/f-dangel/backpack/pull/60)]

### Fixed
- Removed `opt_einsum` dependency 
  [[PR](https://github.com/f-dangel/backpack/pull/54)]
- Missing implementations and wrong backpropagation of KFRA 
  for `Conv2d`, `MaxPool2d`, and `AvgPool2d` 
  [[PR](https://github.com/f-dangel/backpack/pull/53)]
- Remove `try_view` and use `reshape` to use PyTorch 1.4.0 improvements 
  [[PR](https://github.com/f-dangel/backpack/pull/50)]

### Internal
- Docstring style [[PR](https://github.com/f-dangel/backpack/pull/52)]


## [1.1.0] - 2020-02-11

### Added
- Support MC sampling 
  [[Issue](https://github.com/f-dangel/backpack/issues/21),
  [PR](https://github.com/f-dangel/backpack/pull/36)]
- Utilities to handle Kronecker factors 
  [[PR](https://github.com/f-dangel/backpack/pull/17)]
- Examples 
  [[PR](https://github.com/f-dangel/backpack/pull/34)]
  
### Fixed
- Fixed documentation issue in `Batch l2` 
  [[PR](https://github.com/f-dangel/backpack/pull/33)]
- Added support for stride parameter in Conv2d 
  [[Issue](https://github.com/f-dangel/backpack/issues/30), 
  [PR](https://github.com/f-dangel/backpack/pull/31)]
- Pytorch `1.3.0` compatibility 
  [[PR](https://github.com/f-dangel/backpack/pull/8), 
  [PR](https://github.com/f-dangel/backpack/pull/9)]
  
### Internal
- Added 
  continuous integration [[PR](https://github.com/f-dangel/backpack/pull/19)],
  test coverage [[PR](https://github.com/f-dangel/backpack/pull/25)],
  style guide enforcement [[PR](https://github.com/f-dangel/backpack/pull/27)]
- Changed internal shape conventions of backpropagated quantities for performance improvements 
  [[PR](https://github.com/f-dangel/backpack/pull/37)]

## [1.0.1] - 2019-09-05

### Fixed
- Fixed PyPI installaton 

## [1.0.0] - 2019-10-03 

Initial release

[Unreleased]: https://github.com/f-dangel/backpack/compare/v1.1.0...HEAD
[1.2.0]: https://github.com/f-dangel/backpack/compare/1.2.0...1.1.1
[1.1.1]: https://github.com/f-dangel/backpack/compare/1.1.0...1.1.1
[1.1.0]: https://github.com/f-dangel/backpack/compare/1.0.1...1.1.0
[1.0.1]: https://github.com/f-dangel/backpack/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/f-dangel/backpack/releases/tag/1.0.0
