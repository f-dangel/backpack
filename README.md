# backprop-extended

PyTorch extension to compute additional quantities such as
* Hessian blocks
* batch gradients

by backpropagation for employment in 
* 2nd-order and
* variance-adapted

optimization methods

# developer notes
* Please copy the `pre-commit` file to your `.git/hooks/` directory. It will run tests before accepting commits.


# virtualenv notes
* Change into the repository directory and set up a virtualenv with _Python 3_:
```console
virtualenv --python=/usr/bin/python3 .venv
```

* Once you have set up the virtual environment, it can be activated by
```console
source .venv/bin/activate
```

* After activating the environment, install dependencies
```console
pip3 install -r ./requirements.txt
```

* (Optional) run tests using the `pre-commit` script
```console
chmod u+x ./pre-commit
./pre-commit
```

* Deactivate the virtual environment by typing
```console
deactivate
```