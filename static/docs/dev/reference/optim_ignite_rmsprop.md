# LibTorch implementation of RMSprop

Proposed by G. Hinton in his course.

## Usage

``` r
optim_ignite_rmsprop(
  params,
  lr = 0.01,
  alpha = 0.99,
  eps = 1e-08,
  weight_decay = 0,
  momentum = 0,
  centered = FALSE
)
```

## Arguments

- params:

  (iterable): iterable of parameters to optimize or list defining
  parameter groups

- lr:

  (float, optional): learning rate (default: 1e-2)

- alpha:

  (float, optional): smoothing constant (default: 0.99)

- eps:

  (float, optional): term added to the denominator to improve numerical
  stability (default: 1e-8)

- weight_decay:

  optional weight decay penalty. (default: 0)

- momentum:

  (float, optional): momentum factor (default: 0)

- centered:

  (bool, optional) : if `TRUE`, compute the centered RMSProp, the
  gradient is normalized by an estimation of its variance weight_decay
  (float, optional): weight decay (L2 penalty) (default: 0)

## Fields and Methods

See
[`OptimizerIgnite`](https://torch.mlverse.org/docs/dev/reference/OptimizerIgnite.md).

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
optimizer <- optim_ignite_rmsprop(model$parameters(), lr = 0.1)
optimizer$zero_grad()
loss_fn(model(input), target)$backward()
optimizer$step()
} # }
}
```
