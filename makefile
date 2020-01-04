.PHONY: help black black-check flake8 install install-dev

.DEFAULT: help
help:
	@echo "black"
	@echo "        Run black on the project"
	@echo "black-check"
	@echo "        Check if black would change files"
	@echo "flake8"
	@echo "        Run flake8 on the project"
	@echo "install"
	@echo "        Install backpack and dependencies"
	@echo "install-dev"
	@echo "        Install development tools"

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

install:
	@pip install -r requirements.txt
	@pip install .

install-dev:
	@echo "Install dependencies..."
	@pip install -r requirements.txt
	@echo "Uninstall existing version of backpack..."
	@pip uninstall backpack-for-pytorch
	@echo "Install backpack in editable mode..."
	@pip install -e .
	@echo "Install dev tools..."
	@pip install -r requirements-dev.txt
	@echo "Install pre-commit hooks..."
	@pre-commit install
