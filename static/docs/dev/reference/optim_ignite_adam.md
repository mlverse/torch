# LibTorch implementation of Adam

It has been proposed in [Adam: A Method for Stochastic
Optimization](https://arxiv.org/abs/1412.6980).

## Usage

``` r
optim_ignite_adam(
  params,
  lr = 0.001,
  betas = c(0.9, 0.999),
  eps = 1e-08,
  weight_decay = 0,
  amsgrad = FALSE
)
```

## Arguments

- params:

  (iterable): iterable of parameters to optimize or dicts defining
  parameter groups

- lr:

  (float, optional): learning rate (default: 1e-3)

- betas:

  (`Tuple[float, float]`, optional): coefficients used for computing
  running averages of gradient and its square (default: (0.9, 0.999))

- eps:

  (float, optional): term added to the denominator to improve numerical
  stability (default: 1e-8)

- weight_decay:

  (float, optional): weight decay (L2 penalty) (default: 0)

- amsgrad:

  (boolean, optional): whether to use the AMSGrad variant of this
  algorithm from the paper [On the Convergence of Adam and
  Beyond](https://openreview.net/forum?id=ryQu7f-RZ) (default: FALSE)

## Fields and Methods

See
[`OptimizerIgnite`](https://torch.mlverse.org/docs/dev/reference/OptimizerIgnite.md).

## Examples

``` r
if (torch_is_installed()) {
if (FALSE) { # \dontrun{
optimizer <- optim_ignite_adam(model$parameters(), lr = 0.1)
optimizer$zero_grad()
loss_fn(model(input), target)$backward()
optimizer$step()
} # }
}
```
