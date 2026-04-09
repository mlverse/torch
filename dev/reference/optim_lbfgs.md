# LBFGS optimizer

Implements L-BFGS algorithm, heavily inspired by
[minFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)

## Usage

``` r
optim_lbfgs(
  params,
  lr = 1,
  max_iter = 20,
  max_eval = NULL,
  tolerance_grad = 1e-07,
  tolerance_change = 1e-09,
  history_size = 100,
  line_search_fn = NULL
)
```

## Arguments

- params:

  (iterable): iterable of parameters to optimize or dicts defining
  parameter groups

- lr:

  (float): learning rate (default: 1)

- max_iter:

  (int): maximal number of iterations per optimization step (default:
  20)

- max_eval:

  (int): maximal number of function evaluations per optimization step
  (default: max_iter \* 1.25).

- tolerance_grad:

  (float): termination tolerance on first order optimality (default:
  1e-5).

- tolerance_change:

  (float): termination tolerance on function value/parameter changes
  (default: 1e-9).

- history_size:

  (int): update history size (default: 100).

- line_search_fn:

  (str): either 'strong_wolfe' or None (default: None).

## Details

This optimizer is different from the others in that in
`optimizer$step()`, it needs to be passed a closure that (1) calculates
the loss, (2) calls `backward()` on it, and (3) returns it. See example
below.

## Note

This is a very memory intensive optimizer (it requires additional
`param_bytes * (history_size + 1)` bytes). If it doesn't fit in memory
try reducing the history size, or use a different algorithm.

## Warning

This optimizer doesn't support per-parameter options and parameter
groups (there can be only one).

Right now all parameters have to be on a single device. This will be
improved in the future.

If you need to move a model to GPU via `$cuda()`, please do so before
constructing optimizers for it. Parameters of a model after `$cuda()`
will be different objects from those before the call. In general, you
should make sure that the objects pointed to by model parameters subject
to optimization remain the same over the whole lifecycle of optimizer
creation and usage.

## Examples

``` r
if (torch_is_installed()) {
a <- 1
b <- 5
rosenbrock <- function(x) {
  x1 <- x[1]
  x2 <- x[2]
  (a - x1)^2 + b * (x2 - x1^2)^2
}

x <- torch_tensor(c(-1, 1), requires_grad = TRUE)

optimizer <- optim_lbfgs(x)
calc_loss <- function() {
  optimizer$zero_grad()
  value <- rosenbrock(x)
  value$backward()
  value
}

num_iterations <- 2
for (i in 1:num_iterations) {
  optimizer$step(calc_loss)
}

rosenbrock(x)

}
#> torch_tensor
#> 1e-12 *
#>  4.5475
#> [ CPUFloatType{1} ][ grad_fn = <AddBackward0> ]
```
