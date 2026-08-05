# Averaged Stochastic Gradient Descent optimizer

Proposed in Acceleration of stochastic approximation by averaging,
[doi:10.1137/0330046](https://doi.org/10.1137/0330046)

## Usage

``` r
optim_asgd(
  params,
  lr = 0.01,
  lambda = 1e-04,
  alpha = 0.75,
  t0 = 1e+06,
  weight_decay = 0
)
```

## Arguments

- params:

  (iterable): iterable of parameters to optimize or lists defining
  parameter groups

- lr:

  (float): learning rate

- lambda:

  (float, optional): decay term (default: 1e-4)

- alpha:

  (float, optional): power for eta update (default: 0.75)

- t0:

  (float, optional): point at which to start averaging (default: 1e6)

- weight_decay:

  (float, optional): weight decay (L2 penalty) (default: 0)

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
optimizer <- optim_asgd(model$parameters(), lr = 0.1)
optimizer$zero_grad()
loss_fn(model(input), target)$backward()
optimizer$step()
} # }

}
```
