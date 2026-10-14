# Adadelta optimizer

It has been proposed in [ADADELTA: An Adaptive Learning Rate
Method](https://arxiv.org/pdf/1212.5701)

## Usage

``` r
optim_adadelta(params, lr = 1, rho = 0.9, eps = 1e-06, weight_decay = 0)
```

## Arguments

- params:

  (iterable): list of parameters to optimize or list defining parameter
  groups

- lr:

  (float, optional): learning rate (default: 1e-3)

- rho:

  (float, optional): coefficient used for computing a running average of
  squared gradients (default: 0.9)

- eps:

  (float, optional): term added to the denominator to improve numerical
  stability (default: 1e-6)

- weight_decay:

  (float, optional): weight decay (L2 penalty) (default: 0)

## Note

According to the original paper, decaying average of the squared
gradients is computed as follows: \$\$ E\[g^2\]\_{t} = \rho
E\[g^2\]\_{t- 1} + (1 - \rho){g\_{t}}^2 \$\$

RMS of previous squared gradients up to time t: \$\$ RMS\[g\_{t}\] =
\sqrt{E\[g^2\]\_{t} + \epsilon } \$\$

Adadelta update rule: \$\$ \begin{array}{ll} \Delta \theta\_{t} = -
\frac{RMS \[\Delta \theta\]\_{t - 1} }{RMS\[g\]\_{t}} \theta\_{t+1} =
\theta\_{t} + \Delta \theta\_{t} \end{array} \$\$

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
optimizer <- optim_adadelta(model$parameters, lr = 0.1)
optimizer$zero_grad()
loss_fn(model(input), target)$backward()
optimizer$step()
} # }
}
```
