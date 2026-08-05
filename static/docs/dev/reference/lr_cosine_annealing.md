# Set the learning rate of each parameter group using a cosine annealing schedule

Set the learning rate of each parameter group using a cosine annealing
schedule

## Usage

``` r
lr_cosine_annealing(
  optimizer,
  T_max,
  eta_min = 0,
  last_epoch = -1,
  verbose = FALSE
)
```

## Arguments

- optimizer:

  (Optimizer): Wrapped optimizer.

- T_max:

  Maximum number of iterations

- eta_min:

  Minimum learning rate. Default: 0.

- last_epoch:

  The index of the last epoch

- verbose:

  (bool): If `TRUE`, prints a message to stdout for each update.
  Default: `FALSE`.
