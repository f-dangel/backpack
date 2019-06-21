- [Backpropagation extensions (`bpexts`) for `PyTorch`](#org4e4c156)
  - [Overview](#org944ac5c)
  - [Applications](#org55e811f)
  - [Getting started](#orgce0d72b)
    - [Installation](#orgfb9990c)
    - [(Optional) Verify installation](#org557271b)
    - [Check out the tutorials](#org02838fc)
  - [Functionality](#org2842971)
  - [Related work](#org6019d4f)



<a id="org4e4c156"></a>

# Backpropagation extensions (`bpexts`) for `PyTorch`

`bpexts` provides extended `PyTorch` layers capable of computing different quantities than just the gradient. Currently, feedforward (fully-connected and convolutional) neural networks are supported.

Modules are designed in such a way that they can be used as drop-in replacement to the corresponding `torch.nn.Module` class.


<a id="org944ac5c"></a>

## Overview

Very briefly, the following quantities can be computed:

-   First-order information
    -   Batchwise gradients (`bpexts.gradient`)
    -   Sum of squared gradients (`bpexts.sumgradsquared`)
-   Second-order information (for more details, consult [1](#org28e897d))
    -   Approximate block-diagonal curvature matrices obtained by Hessian backpropagation (`bpexts.hbp`)
    -   Exact block-diagonal curvature matrix-vector products (`bpexts.cvp`)


<a id="org55e811f"></a>

## Applications

The main motivation of `bpexts` is to preserve `PyTorch`'s modular structure. Algorithms that require more information than just the gradient, for instance

-   Variance-adapted first-order optimization methods
-   Second-order methods that use a block-diagonal curvature estimate

can thus be formulated in a more elegant fashion.


<a id="orgce0d72b"></a>

## Getting started


<a id="orgfb9990c"></a>

### Installation

1.  Clone the repository
    
    ```bash:
    git clone https://github.com/f-dangel/bpexts.git
    ```
2.  Change into the directory
    
    ```bash:
    cd bpexts/
    ```
3.  Install dependencies and `bpexts`
    
    ```bash:
    pip3 install -r ./requirements.txt
    pip3 install .
    ```
4.  (**Optional**, for reproducing experiments of [1](#org28e897d)) Install requirements to run our experiments
    
    ```bash:
    pip3 install -r ./requirements_exp.txt
    ```

Congratulations! You should now be able to call `import bpexts` in a `python3` session.


<a id="org557271b"></a>

### (Optional) Verify installation

If `pytest` is installed on your system, you can run the tests in the repository directory by

```bash:
pytest -v tests/
```


<a id="org02838fc"></a>

### Check out the tutorials

One main goal of `bpexts` is to simplify the computation of quantities in the backward pass. You can find explanations on how to use the code in the [tutorials](./tutorials) directory.

Available tutorials:

-   First-order information
    -   TODO
-   Second-order information
    -   TODO
-   Auxiliary
    -   TODO

We plan to add further illustrative examples in the future.


<a id="org2842971"></a>

## Functionality

-   **`bpexts.gradient`:** Batch gradients
    -   For a mini-batch over N samples, compute the individual gradients g<sub>n</sub> of the gradient g = &sum;<sub>n</sub> g<sub>n</sub>
-   **`bpexts.sumgradsquared`:** Sum of squared gradients
    -   For a mini-batch over N samples, instead of computing the gradient (the sum of the gradient of individual samples g = &sum;<sub>n</sub> g<sub>n</sub>), computes the sum of the individual gradients, squared element-wise g<sub>2</sub> = &sum;<sub>n</sub> g<sub>n</sub> o g<sub>n</sub> (where o indicates element-wise multiplication)
    -   Given the gradient g and the sgs g<sub>2</sub> this makes it easy to compute the element-wise variance of the gradient over the mini-batch as g<sub>2</sub> - g o g
-   **`bpexts.hbp`:** Hessian backpropagation (see [1](#org28e897d) for details)
    -   Approximate the block-diagonal of different curvature matrices
    -   Provides multiplication with the approximate blocks of
        -   Hessian (H)
        -   Generalized Gauss-Newton matrix (GGN)
        -   Positive-curvature Hessian (PCH)
    -   Different approximation modes
-   **`bpexts.cvp`:** Curvature matrix-vector products (see [1](#org28e897d) for details)
    -   Provides **exact** multiplication with the diagonal blocks of
        -   Hessian (H)
        -   Generalized Gauss-Newton matrix (GGN)
        -   Positive-curvature Hessian (PCH)
-   **`bpexts.optim`:** Optimizers
    -   Implements conjugate gradients and the Newton-style optimizer used in [1](#org28e897d)


<a id="org6019d4f"></a>

## Related work

-   <a id="org28e897d"></a> [[1](#org28e897d)] Dangel, F. and Hennig, P.: [A Modular Approach to Block-diagonal Hessian Approximations for Second-order Optimization](https://arxiv.org/abs/1902.01813) (2019)
    -   The work presents an extended backpropagation procedure, referred to as **Hessian backpropagation (HBP)**, for computing curvature approximations of feedforward neural networks.
    -   To ****reproduce the experiment**** (Figure 5) in the paper, we recommend using our script. A step-by-step instruction is given in the [README](examples/2019_02_dangel_hbp/README.rst) file in [`examples/2019_02_dangel_hbp/`](examples/2019_02_dangel_hbp/).
