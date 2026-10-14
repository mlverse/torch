# Creates learning rate schedulers

Creates learning rate schedulers

## Usage

``` r
lr_scheduler(
  classname = NULL,
  inherit = LRScheduler,
  ...,
  parent_env = parent.frame()
)
```

## Arguments

- classname:

  optional name for the learning rate scheduler

- inherit:

  an optional learning rate scheduler to inherit from

- ...:

  named list of methods. You must implement the `get_lr()` method that
  doesn't take any argument and returns learning rates for each
  `param_group` in the optimizer.

- parent_env:

  passed to
  [`R6::R6Class()`](https://r6.r-lib.org/reference/R6Class.html).
