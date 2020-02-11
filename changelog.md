# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
[1.1.0]: https://github.com/f-dangel/backpack/compare/1.0.1...1.1.0
[1.0.1]: https://github.com/f-dangel/backpack/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/f-dangel/backpack/releases/tag/1.0.0
