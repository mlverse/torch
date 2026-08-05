# Sets the learning rate of each parameter group to the initial lr times a given function. When last_epoch=-1, sets initial lr as lr.

Sets the learning rate of each parameter group to the initial lr times a
given function. When last_epoch=-1, sets initial lr as lr.

## Usage

``` r
lr_lambda(optimizer, lr_lambda, last_epoch = -1, verbose = FALSE)
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
# Assuming optimizer has two groups.
lambda1 <- function(epoch) epoch %/% 30
lambda2 <- function(epoch) 0.95^epoch
if (FALSE) { # \dontrun{
scheduler <- lr_lambda(optimizer, lr_lambda = list(lambda1, lambda2))
for (epoch in 1:100) {
  train(...)
  validate(...)
  scheduler$step()
}
} # }

}
```
