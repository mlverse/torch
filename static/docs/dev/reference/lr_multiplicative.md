# Multiply the learning rate of each parameter group by the factor given in the specified function. When last_epoch=-1, sets initial lr as lr.

Multiply the learning rate of each parameter group by the factor given
in the specified function. When last_epoch=-1, sets initial lr as lr.

## Usage

``` r
lr_multiplicative(optimizer, lr_lambda, last_epoch = -1, verbose = FALSE)
```

## Arguments

- optimizer:

  (Optimizer): Wrapped optimizer.

- lr_lambda:

  (function or list): A function which computes a multiplicative factor
  given an integer parameter epoch, or a list of such functions, one for
  each group in optimizer.param_groups.

- last_epoch:

  (int): The index of last epoch. Default: -1.

- verbose:

  (bool): If `TRUE`, prints a message to stdout for each update.
  Default: `FALSE`.

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
lmbda <- function(epoch) 0.95
scheduler <- lr_multiplicative(optimizer, lr_lambda = lmbda)
for (epoch in 1:100) {
  train(...)
  validate(...)
  scheduler$step()
}
} # }

}
```
