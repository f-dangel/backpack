# <img alt="BackPACK" src="./logo/backpack_logo_torch.svg" height="90"> BackPACK developer manual

## General standards 
- Python version: support 3.6+, use 3.7 for development
- `git` [branching model](https://nvie.com/posts/a-successful-git-branching-model/)
- Docstring style:  [Google](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Test runner: [`pytest`](https://docs.pytest.org/en/latest/)
- Formatting: [`black`](https://black.readthedocs.io) ([`black` config](black.toml))
- Linting: [`flake8`](http://flake8.pycqa.org/) ([`flake8` config](.flake8))

---

The development tools are managed using [`make`](https://www.gnu.org/software/make/) as an interface ([`makefile`](makefile)). For an overview, call
```bash
make help 
```
  
## Suggested workflow with [Anaconda](https://docs.anaconda.com/anaconda/install/)
1. Clone the repository. Check out the `development` branch
```bash
git clone https://github.com/f-dangel/backpack.git ~/backpack
cd ~/backpack
git checkout development
```
2. Create `conda` environment `backpack` with the [environment file](.conda_env.yml). It comes with all dependencies installed, and BackPACK installed with the [--editable](http://codumentary.blogspot.com/2014/11/python-tip-of-year-pip-install-editable.html) option. Activate it.
```bash
make conda-env
conda activate backpack
```
3. Install the development dependencies and `pre-commit` hooks
```bash
make install-dev
```
4. **You're set up!** Here are some useful commands for developing
  - Run the tests
    ```bash
    make test
    ```
  - Lint code
    ```bash
    make flake8
    ```
  - Check format (code, imports, and docstrings)
    ```bash
    make format-check
    ```

## Documentation

### Build
- Use `make build-docs`
- To use the RTD theme, uncomment the line `html_theme = "sphinx_rtd_theme"` in `docs/rtd/conf.py` (this line needs to be uncommented for automatic deployment to RTD)

### View
- Go to `docs_src/rtd_output/html`, open `index.html`

### Edit
- Content in `docs_src/rtd/*.rst`
- Docstrings in code
- Examples in `examples/rtd_examples` (compiled automatically)


## Details

- Running quick/extensive tests: ([testing readme](test/readme.md))
- Continuous Integration (CI)/Quality Assurance (QA)
  - [`Travis`](https://travis-ci.org/f-dangel/backpack) ([`Travis` config](.travis.yaml))
    - Run tests: [`pytest`](https://docs.pytest.org/en/latest/)
    - Report test coverage: [`coveralls`](https://coveralls.io)
    - Run examples
  - [`Github workflows`](https://github.com/f-dangel/backpack/actions) ([config](.github/workflows))
    - Check code formatting: [`black`](https://black.readthedocs.io) ([`black` config](black.toml))
    - Lint code: [`flake8`](http://flake8.pycqa.org/) ([`flake8` config](.flake8))
    - Check docstring style: [`pydocstyle`](https://github.com/PyCQA/pydocstyle) ([`pydocstyle` config](.pydocstyle))
    - Check docstring description matches definition: [`darglint`](https://github.com/terrencepreilly/darglint) ([`darglint` config](.darglint))
- Optional [`pre-commit`](https://github.com/pre-commit/pre-commit) hooks [ `pre-commit` config ](.pre-commit-config.yaml)

###### _BackPACK is not endorsed by or affiliated with Facebook, Inc. PyTorch, the PyTorch logo and any related marks are trademarks of Facebook, Inc._
