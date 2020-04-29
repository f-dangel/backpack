# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.1] - 2020-04-27

### Added
- Support 2nd-order backpropagation for vectors in `MSELoss` [[PR](https://github.com/f-dangel/backpack/pull/61)]
- Sanity checks
  - No inplace modification [[PR](https://github.com/f-dangel/backpack/pull/59)]
  - Unsupported loss function hyperparameters [[PR](https://github.com/f-dangel/backpack/pull/60)]
  - 2nd-order backpropagation [[PR](https://github.com/f-dangel/backpack/pull/60)]
- Tests for KFRA backpropagation [[PR](https://github.com/f-dangel/backpack/pull/53)]
- Documentation on readthedocs [[PR1](https://github.com/f-dangel/backpack/pull/57), [PR2](https://github.com/f-dangel/backpack/pull/58)]

### Fixed
- Remove `opt_einsum` dependency [[PR](https://github.com/f-dangel/backpack/pull/54)]
- Missing implementations in and wrong backpropagation of KFRA for `Conv2d`, `MaxPool2d`, and `AvgPool2d` [[PR](https://github.com/f-dangel/backpack/pull/53)]

### Internal
- Docstring style [[PR](https://github.com/f-dangel/backpack/pull/52)]
- Remove `try_view` by `reshape` [[PR](https://github.com/f-dangel/backpack/pull/50)]


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
[1.1.1]: https://github.com/f-dangel/backpack/compare/1.1.0...1.1.1
[1.1.0]: https://github.com/f-dangel/backpack/compare/1.0.1...1.1.0
[1.0.1]: https://github.com/f-dangel/backpack/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/f-dangel/backpack/releases/tag/1.0.0
