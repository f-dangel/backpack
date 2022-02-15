# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.5.0] - 2022-02-15

This small release improves ResNet support of some second-order extensions and
adds some more custom layers.

Thanks to [@jabader97](https://github.com/jabader97) for contributing to `KFAC`.

### Added/New
- `KFAC` for ResNets modules
  ([commit](https://github.com/f-dangel/backpack/commit/69f561f9f827115d2075faafe384592eff5c0158))
- Custom modules for
  - Slicing
   ([commit](https://github.com/f-dangel/backpack/commit/fc9a97ef0180a840437367b941493cc840d6d564))
  - Padding
   ([commit](https://github.com/f-dangel/backpack/commit/175db2abf9d1be9c02796fbafdcb0ad4d81d1300))
- `{Batch}DiagGGN{Exact,MC}`, `{Batch}DiagHessian` support for
  - Custom slicing module
    ([commit](https://github.com/f-dangel/backpack/commit/fc9a97ef0180a840437367b941493cc840d6d564))
  - Custom padding module
    ([commit](https://github.com/f-dangel/backpack/commit/175db2abf9d1be9c02796fbafdcb0ad4d81d1300))
- `SqrtGGN{Exact,MC}` support for
  - `BatchNormNd` (eval mode)
    ([commit](https://github.com/f-dangel/backpack/commit/7e634e3a78884f1cc4676e00991143aaed2ab56f))
  - ResNet modules (`ScaleModule`, `SumModule`, `Identity`)
    ([commit](https://github.com/f-dangel/backpack/commit/98e7420368ef56b830b9251c067ed6b8f0d43502))
  - Custom slicing module
    ([commit](https://github.com/f-dangel/backpack/commit/fc9a97ef0180a840437367b941493cc840d6d564))
  - Custom padding module
    ([commit](https://github.com/f-dangel/backpack/commit/175db2abf9d1be9c02796fbafdcb0ad4d81d1300))

### Fixed/Removed

### Internal
- Automate PyPI release through GitHub action
  ([commit](https://github.com/f-dangel/backpack/commit/cf50703aba72d8e208e2abba9a5b11365b0fd9aa))
- Add `KFAC` tests for linear layers to test suite
  ([commit](https://github.com/f-dangel/backpack/commit/26352070fae41d2f5ded10bf704c89de7a7416fb))
- Update formatting to `black==22.1.0`
  ([commit](https://github.com/f-dangel/backpack/commit/cbf7955dd1812d5fcbab544d0d8a04d8ed102448))
- Add BackPACK logo to RTD
  ([commit](https://github.com/f-dangel/backpack/commit/70441b6f8889136b3630f15cfde2174d90e809cc))

## [1.4.0] - 2021-11-12

This release ships many new features. Some rely on recent PyTorch functionality.
We now require `torch>=1.9.0`.

**Highlights:**

- *ResNets & RNNs:* Thanks to [@schaefertim](https://github.com/schaefertim) for
  bringing basic support for RNNs
  ([#16](https://github.com/f-dangel/backpack/issues/16),
  [tutorial](https://docs.backpack.pt/en/1.4.0/use_cases/example_rnn.html#sphx-glr-use-cases-example-rnn-py)) and
  ResNets ([#14](https://github.com/f-dangel/backpack/issues/14),
  [tutorial](https://docs.backpack.pt/en/1.4.0/use_cases/example_resnet_all_in_one.html#sphx-glr-use-cases-example-resnet-all-in-one-py))
- *`SqrtGGN{Exact,MC}` extension:* Symmetric factorization of the generalized
  Gauss-Newton/Fisher (see
  [arXiv:2106.02624](https://arxiv.org/abs/2106.02624))
- *Sub-sampling:* Allows for restricting BackPACK extensions to a sub-set of
  samples in the mini-batch
  ([#12](https://github.com/f-dangel/backpack/issues/12),
   [tutorial](https://docs.backpack.pt/en/1.4.0/use_cases/example_subsampling.html#sphx-glr-use-cases-example-subsampling-py))

### Added/New
- Converter functionality for basic support of ResNets and RNNs
  [[PR1](https://github.com/f-dangel/backpack/pull/202),
   [PR2](https://github.com/f-dangel/backpack/pull/221),
   [PR3](https://github.com/f-dangel/backpack/pull/229)]
- New extensions:
    - `SqrtGGNExact`: Symmetric factorization of the exact GGN/Fisher
      [[PR](https://github.com/f-dangel/backpack/pull/180)]
    - `SqrtGGNMC`: Symmetric factorization of the MC-approximated GGN/Fisher
      [[PR](https://github.com/f-dangel/backpack/pull/182)]
- Module support:
    - `Linear`: Support additional (more than 2) input dimensions
      [[PR1](https://github.com/f-dangel/backpack/pull/185),
       [PR2](https://github.com/f-dangel/backpack/pull/186)]
    - `BatchNormNd`: Distinguish evaluation and training mode, support
      first-order extensions and `DiagGGN{Exact,MC}`
      [[#160](https://github.com/f-dangel/backpack/issues/160),
      [PR1](https://github.com/f-dangel/backpack/pull/179),
      [PR2](https://github.com/f-dangel/backpack/pull/201)]
    - `AdaptiveAvgPoolND`: Support first-order extensions and
      `DiagGGN{Exact,MC}`
      [[PR](https://github.com/f-dangel/backpack/pull/201)]
    - `RNN`: Support first-order extensions and `DiagGGN{MC,Exact}`
      [[PR1](https://github.com/f-dangel/backpack/pull/159)
       [PR2](https://github.com/f-dangel/backpack/pull/158)
       [PR3](https://github.com/f-dangel/backpack/pull/156)]
    - `LSTM`: Support first-order extensions and `DiagGGN{MC,Exact}`
      [[PR](https://github.com/f-dangel/backpack/pull/215)]
    - `CrossEntropyLoss`: Support additional (more than 2) input dimensions.
      [[PR](https://github.com/f-dangel/backpack/pull/211)]
    - `Embedding`: Support first-order extensions and `DiagGGN{MC,Exact}`
      [[PR](https://github.com/f-dangel/backpack/pull/216)]
- Mini-batch sub-sampling
  - `BatchGrad`
    [[PR1](https://github.com/f-dangel/backpack/pull/200),
     [PR2](https://github.com/f-dangel/backpack/pull/210)]
  - `SqrtGGN{Exact,MC}`
    [[PR](https://github.com/f-dangel/backpack/pull/200)]
- `retain_graph` option for `backpack` context
  [[PR](https://github.com/f-dangel/backpack/pull/217)]
- Assume batch axis always first
  [[PR](https://github.com/f-dangel/backpack/pull/227)]

### Fixed/Removed
- Deprecate `python3.6`, require at least `python3.7`
  [[PR](https://github.com/f-dangel/backpack/pull/190)]

### Internal
- Use `full_backward_hook` for `torch>=1.9.0`
  [[PR](https://github.com/f-dangel/backpack/pull/194)]
- Core
    - Implement derivatives for `LSTM`
      [[PR](https://github.com/f-dangel/backpack/pull/169)]
    - Implement derivatives for `AdaptiveAvgPoolNd`
      [[PR](https://github.com/f-dangel/backpack/pull/165)]
    - Sub-sampling
      - `weight_jac_t_mat_prod`
        [[PR](https://github.com/f-dangel/backpack/pull/195)]
      - `bias_jac_t_mat_prod`
        [[PR](https://github.com/f-dangel/backpack/pull/196)]
      - `*_jac_t_mat_prod` of `RNN` and `LSTM` parameters
        [[PR](https://github.com/f-dangel/backpack/pull/197)]
      - `jac_t_mat_prod`
        [[PR](https://github.com/f-dangel/backpack/pull/205)]
      - Hessian square root decomposition (exact and MC)
        [[PR](https://github.com/f-dangel/backpack/pull/207)]
    - Refactor: Share code for `*_jac_t_mat_prod`
      [[PR](https://github.com/f-dangel/backpack/pull/203)]
- Extensions
    - Refactor `BatchL2Grad`, introducing a base class
      [[PR](https://github.com/f-dangel/backpack/pull/175)]
    - Automate parameter functions for `BatchGrad` and `Grad`
      [[PR](https://github.com/f-dangel/backpack/pull/150)]
    - Introduce interface to check module hyperparameters
      [[PR](https://github.com/f-dangel/backpack/pull/206)]
- Tests
    - Check if module Hessian is zero
      [[PR](https://github.com/f-dangel/backpack/pull/183)]
    - Reduce run time
      [[PR](https://github.com/f-dangel/backpack/pull/199)]

## [1.3.0] - 2021-06-16

Thanks to [@sbharadwajj](https://github.com/sbharadwajj)
and [@schaefertim](https://github.com/schaefertim) for
co-authoring many PRs shipped in this release.

### Added
- New extensions
  - `BatchDiagGGN{Exact,MC}`: Per sample diagonal of the GGN/Fisher,
    exact or with a Monte-Carlo approximation
    [[PR1](https://github.com/f-dangel/backpack/pull/135),
     [PR2](https://github.com/f-dangel/backpack/pull/139),
     [PR3](https://github.com/f-dangel/backpack/pull/170),
     [example](https://docs.backpack.pt/en/1.3.0/basic_usage/example_all_in_one.html)]
  - `BatchDiagHessian`: Per sample diagonal of the Hessian
    [[PR1](https://github.com/f-dangel/backpack/pull/137),
     [PR2](https://github.com/f-dangel/backpack/pull/170),
     [example](https://docs.backpack.pt/en/1.3.0/basic_usage/example_all_in_one.html)]
- Support for more layers
  ([[PR](https://github.com/f-dangel/backpack/pull/171),
  [overview](https://docs.backpack.pt/en/1.3.0/supported-layers.html)])
  - `DiagGGN{Exact,MC}` extensions
    - `Conv{1,3}d`, `ConvTranspose{1,2,3}d`, `LeakyReLU`,
      `LogSigmoid`, `ELU`, `SELU`
      [[PR](https://github.com/f-dangel/backpack/pull/113)]
    - `MaxPool{1,3}d`
      [[PR](https://github.com/f-dangel/backpack/pull/125)]
    - `AvgPool{1,3}d`
      [[PR](https://github.com/f-dangel/backpack/pull/128)]
  - `DiagHessian` extension
    - `Conv{1,3}d`, `ConvTranspose{1,2,3}d`, `LeakyReLU`,
      `LogSigmoid`
      [[PR](https://github.com/f-dangel/backpack/pull/115)]
    - `MaxPool{1,3}d`
      [[PR](https://github.com/f-dangel/backpack/pull/124)]
    - `AvgPool{1,3}d`
      [[PR](https://github.com/f-dangel/backpack/pull/127)]
    - `ELU`, `SELU`
      [[PR](https://github.com/f-dangel/backpack/pull/168)]
  - `group` argument of (transpose) convolutions
    - Full support for first-order diagonal curvature extensions
      [[PR1](https://github.com/f-dangel/backpack/pull/151),
       [PR2](https://github.com/f-dangel/backpack/pull/161),
       [PR3](https://github.com/f-dangel/backpack/pull/162),
       [PR4](https://github.com/f-dangel/backpack/pull/163)]
    - No support (yet) for `KFAC`, `KFLR` and `KFRA` extensions
      [[PR](https://github.com/f-dangel/backpack/pull/167)]
- Extension hook which allows to run code right after a BackPACK extension
  [[PR](https://github.com/f-dangel/backpack/pull/120),
   [example](https://docs.backpack.pt/en/development/use_cases/example_extension_hook.html)]
- Context to disable BackPACK
  [[PR](https://github.com/f-dangel/backpack/pull/119)]
- Tutorial how to extend custom modules
  [[PR](https://github.com/f-dangel/backpack/pull/152),
  [example](https://docs.backpack.pt/en/development/use_cases/example_custom_module.html)]
- (Experimental) Alternative convolution weight Jacobian with option to save memory
  [[PR](https://github.com/f-dangel/backpack/pull/142),
   [example](https://docs.backpack.pt/en/development/use_cases/example_save_memory_convolutions.html)]

### Fixed/Removed
- Remove hooks that save input/output shapes. This probably
  resolves [#97](https://github.com/f-dangel/backpack/issues/97)
  [[PR](https://github.com/f-dangel/backpack/pull/118)]
- Remove `DiagGGN` from API (use `DiagGGNExact` instead). It was
  indented as abstract parent class for `DiagGGNExact` and `DiagGGNMC`
  [[PR](https://github.com/f-dangel/backpack/pull/138)]

### Internal
- CI
  - Move tests from Travis to GitHub actions
    [[PR](https://github.com/f-dangel/backpack/pull/118),
     [small fix](https://github.com/f-dangel/backpack/pull/130)]
  - Test `DiagHessian` with new test suite
    [[PR](https://github.com/f-dangel/backpack/pull/114)]
  - Test `DiagGGN` with new test suite, introduce 'light' and
    'full' tests
    [[PR1](https://github.com/f-dangel/backpack/pull/112),
     [PR2](https://github.com/f-dangel/backpack/pull/140)]
  - Fix `isort`
    [[PR](https://github.com/f-dangel/backpack/pull/144)]
  - Add partial docstring checks
    [[PR](https://github.com/f-dangel/backpack/pull/157)]
  - Add docstrings to contribution guide lines
    [[commit](https://github.com/f-dangel/backpack/commit/42897ca6dff1a5cd4a4d17d78dc9e309fa3ee178)]
  - Auto-format and lint examples
    [[PR](https://github.com/f-dangel/backpack/pull/167)]
- Refactoring
  - Share code between `Conv{Transpose}{1,2,3}d` in `BatchL2Grad`
    [[PR](https://github.com/f-dangel/backpack/pull/111)]
  - Use `eingroup` package, remove custom `eingroup` utility
    [[PR](https://github.com/f-dangel/backpack/pull/133)]
- Core
  - Implement derivatives for `MaxPool{1,3}d`
    [[PR](https://github.com/f-dangel/backpack/pull/129)]
  - Implement derivatives for `AvgPool{1,3}d`
    [[PR](https://github.com/f-dangel/backpack/pull/126)]
  - Support for `groups` in (transpose) convolutions
    [[PR1](https://github.com/f-dangel/backpack/pull/151),
     [PR2](https://github.com/f-dangel/backpack/pull/161),
     [PR3](https://github.com/f-dangel/backpack/pull/162),
     [PR4](https://github.com/f-dangel/backpack/pull/163)]

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

[Unreleased]: https://github.com/f-dangel/backpack/compare/v1.5.0...HEAD
[1.4.0]: https://github.com/f-dangel/backpack/compare/1.5.0...1.4.0
[1.4.0]: https://github.com/f-dangel/backpack/compare/1.4.0...1.3.0
[1.3.0]: https://github.com/f-dangel/backpack/compare/1.3.0...1.2.0
[1.2.0]: https://github.com/f-dangel/backpack/compare/1.2.0...1.1.1
[1.1.1]: https://github.com/f-dangel/backpack/compare/1.1.0...1.1.1
[1.1.0]: https://github.com/f-dangel/backpack/compare/1.0.1...1.1.0
[1.0.1]: https://github.com/f-dangel/backpack/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/f-dangel/backpack/releases/tag/1.0.0
