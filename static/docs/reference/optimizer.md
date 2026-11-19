# Creates a custom optimizer

When implementing custom optimizers you will usually need to implement
the `initialize` and `step` methods. See the example section below for a
full example.

## Usage

``` r
optimizer(
  name = NULL,
  inherit = Optimizer,
  ...,
  private = NULL,
  active = NULL,
  parent_env = parent.frame()
)
```

## Arguments

- name:

  (optional) name of the optimizer

- inherit:

  (optional) you can inherit from other optimizers to re-use some
  methods.

- ...:

  Pass any number of fields or methods. You should at least define the
  `initialize` and `step` methods. See the examples section.

- private:

  (optional) a list of private methods for the optimizer.

- active:

  (optional) a list of active methods for the optimizer.

- parent_env:

  used to capture the right environment to define the class. The default
  is fine for most situations.

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

# In this example we will create a custom optimizer
# that's just a simplified version of the `optim_sgd` function.

optim_sgd2 <- optimizer(
  initialize = function(params, learning_rate) {
    defaults <- list(
      learning_rate = learning_rate
    )
    super$initialize(params, defaults)
  },
  step = function() {
    with_no_grad({
      for (g in seq_along(self$param_groups)) {
        group <- self$param_groups[[g]]
        for (p in seq_along(group$params)) {
          param <- group$params[[p]]

          if (is.null(param$grad) || is_undefined_tensor(param$grad)) {
            next
          }

          param$add_(param$grad, alpha = -group$learning_rate)
        }
      }
    })
  }
)

x <- torch_randn(1, requires_grad = TRUE)
opt <- optim_sgd2(x, learning_rate = 0.1)
for (i in 1:100) {
  opt$zero_grad()
  y <- x^2
  y$backward()
  opt$step()
}
all.equal(x$item(), 0, tolerance = 1e-9)
}
#> [1] TRUE
```
