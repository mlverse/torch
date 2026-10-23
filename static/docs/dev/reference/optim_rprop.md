# Implements the resilient backpropagation algorithm.

Proposed first in [RPROP - A Fast Adaptive Learning
Algorithm](https://ieeexplore.ieee.org/document/298623)

## Usage

``` r
optim_rprop(params, lr = 0.01, etas = c(0.5, 1.2), step_sizes = c(1e-06, 50))
```

## Arguments

- params:

  (iterable): iterable of parameters to optimize or lists defining
  parameter groups

- lr:

  (float, optional): learning rate (default: 1e-2)

- etas:

  (Tuple(float, float), optional): pair of (etaminus, etaplis), that are
  multiplicative increase and decrease factors (default: (0.5, 1.2))

- step_sizes:

  (vector(float, float), optional): a pair of minimal and maximal
  allowed step sizes (default: (1e-6, 50))

## Warning

If you need to move a model to GPU via `$cuda()`, please do so before
constructing optimizers for it. Parameters of a model after `$cuda()`
will be different objects from those before the call. In general, you
should make sure that the objects pointed to by model parameters subject
to optimization remain the same over the whole lifecycle of optimizer
creation and usage.

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
optimizer <- optim_rprop(model$parameters(), lr = 0.1)
optimizer$zero_grad()
loss_fn(model(input), target)$backward()
optimizer$step()
} # }
}
```
