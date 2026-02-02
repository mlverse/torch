# LibTorch implementation of SGD

Implements stochastic gradient descent (optionally with momentum).
Nesterov momentum is based on the formula from On the importance of
initialization and momentum in deep learning.

## Usage

``` r
optim_ignite_sgd(
  params,
  lr = optim_required(),
  momentum = 0,
  dampening = 0,
  weight_decay = 0,
  nesterov = FALSE
)
```

## Arguments

- params:

  (iterable): iterable of parameters to optimize or dicts defining
  parameter groups

- lr:

  (float): learning rate

- momentum:

  (float, optional): momentum factor (default: 0)

- dampening:

  (float, optional): dampening for momentum (default: 0)

- weight_decay:

  (float, optional): weight decay (L2 penalty) (default: 0)

- nesterov:

  (bool, optional): enables Nesterov momentum (default: FALSE)

## Fields and Methods

See
[`OptimizerIgnite`](https://torch.mlverse.org/docs/dev/reference/OptimizerIgnite.md).

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
optimizer <- optim_ignite_sgd(model$parameters(), lr = 0.1)
optimizer$zero_grad()
loss_fn(model(input), target)$backward()
optimizer$step()
} # }
}
```
