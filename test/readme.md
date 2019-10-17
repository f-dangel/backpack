# Testing
Automated testing based on [`pytest`](https://docs.pytest.org/en/latest/).
Install with `pip install pytest`, run tests with `pytest` from this directory.

Useful options:
```
-v          verbose output
-k text     select tests containing text in their name
-x          stop if a test fails
--tb=no     disable trace output
--help
```

## Optional tests
Uses [`pytest-optional-tests`](https://pypi.org/project/pytest-optional-tests) f
or optional tests. Install with `pip install pytest-optional-tests`.

Optional test categories are defined in `pytest.ini`
and tests are marked with `@pytest.mark.OPTIONAL_TEST_CATEGORY`.

To run the optional tests, use 
`pytest --run-optional-tests=OPTIONAL_TEST_CATEGORY`

## Run all tests for BackPACK
In working directory `tests/`, run
```bash
pytest -vx --run-optional-tests=montecarlo .
```
