Basics about the development setup 

|-|-|
|-|-|
| Python version | The subset of Python 3 and Pytorch (`3.5, 3.6, 3.7`) and use `3.7` for development |
| Tooling management | [`make`](https://www.gnu.org/software/make/) as an interface to the dev tools ([makefile](makefile)) |
| Testing | [`pytest`](https://docs.pytest.org) ([testing readme](test/readme.md))
| Style | [`black`](https://black.readthedocs.io) ([rules](black.toml)) for formatting and [`flake8`](http://flake8.pycqa.org/) ([rules](.flake8)) for linting |
| CI/QA | [`Travis`](https://travis-ci.org/f-dangel/backpack) ([config](.travis.yaml)) to run tests and [`Github workflows`](https://github.com/f-dangel/backpack/actions) ([config](.github/workflows)) to check formatting and linting |

