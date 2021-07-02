.PHONY: help
.PHONY: black black-check flake8
.PHONY: install install-dev install-devtools install-test install-lint install-docs
.PHONY: test test-no-gpu
.PHONY: test-light test-light-no-gpu
.PHONY: conda-env
.PHONY: black isort format
.PHONY: black-check isort-check format-check format-check-partial
.PHONY: flake8
.PHONY: pydocstyle-check pydocstyle-check-partial
.PHONY: darglint-check darglint-check-partial
.PHONY: build-docs

.DEFAULT: help
help:
	@echo "test"
	@echo "        Run pytest on the project and report coverage"
	@echo "test-no-gpu"
	@echo "        Exclude GPU tests, run pytest on the project and report coverage."
	@echo "test-light"
	@echo "        Run pytest on the light part of project and report coverage"
	@echo "test-light-no-gpu"
	@echo "        Exclude GPU tests, run pytest on the light part of project and report coverage"
	@echo "black"
	@echo "        Run black on the project"
	@echo "black-check"
	@echo "        Check if black would change files"
	@echo "flake8"
	@echo "        Run flake8 on the project"
	@echo "pydocstyle-check"
	@echo "        Run pydocstyle on the project"
	@echo "pydocstyle-check-partial"
	@echo "        Run pydocstyle on documented part of the project"
	@echo "darglint-check"
	@echo "        Run darglint on the project"
	@echo "darglint-check-partial"
	@echo "        Run darglint on documented part of the project"
	@echo "install"
	@echo "        Install backpack and dependencies"
	@echo "isort"
	@echo "        Run isort (sort imports) on the project"
	@echo "isort-check"
	@echo "        Check if isort (sort imports) would change files"
	@echo "install-dev"
	@echo "        Install backpack and all development tools"
	@echo "install-lint"
	@echo "        Install backpack and linter tools (included in install-dev)"
	@echo "install-test"
	@echo "        Install backpack and testing tools (included in install-dev)"
	@echo "install-docs"
	@echo "        Install backpack and tools to build/view the docs (included in install-dev)"
	@echo "conda-env"
	@echo "        Create conda environment 'backpack' with dev setup"
	@echo "build-docs"
	@echo "        Build the docs"
###
# Test coverage
test:
	@pytest -vx -rs --run-optional-tests=montecarlo --cov=backpack .

test-light:
	@pytest -vx -rs --cov=backpack .

test-no-gpu:
	@pytest -k "not cuda" -vx -rs --run-optional-tests=montecarlo --cov=backpack .

test-light-no-gpu:
	@pytest -k "not cuda" -vx -rs --cov=backpack .

###
# Linter and autoformatter

# Uses black.toml config instead of pyproject.toml to avoid pip issues. See
# - https://github.com/psf/black/issues/683
# - https://github.com/pypa/pip/pull/6370
# - https://pip.pypa.io/en/stable/reference/pip/#pep-517-and-518-support
black:
	@black . --config=black.toml

black-check:
	@black . --config=black.toml --check

flake8:
	@flake8 .

pydocstyle-check:
	@pydocstyle --count .

pydocstyle-check-partial:
	@pydocstyle --count $(shell grep -v '^#' fully_documented.txt )

darglint-check:
	@darglint --verbosity 2 .

darglint-check-partial:
	@darglint --verbosity 2 $(shell grep -v '^#' fully_documented.txt)

isort:
	@isort .

isort-check:
	@isort . --check --diff

format:
	@make black
	@make isort
	@make black-check

format-check: black-check isort-check flake8 pydocstyle-check darglint-check

format-check-partial: black-check isort-check flake8 pydocstyle-check-partial darglint-check-partial

###
# Installation

install:
	@pip install -e .

install-lint:
	@pip install -e .[lint]

install-test:
	@pip install -e .[test]
	@pip install git+https://git@github.com/Stonesjtu/pytorch_memlab.git@6ab5fab#egg=pytorch_memlab

install-docs:
	@pip install -e .[docs]

install-dev: install-lint install-test install-docs

###
# Conda environment
conda-env:
	@conda env create --file .conda_env.yml

###
# Documentation
build-docs:
	@cd docs_src/rtd && make html
