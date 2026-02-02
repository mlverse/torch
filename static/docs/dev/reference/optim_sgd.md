# SGD optimizer

Implements stochastic gradient descent (optionally with momentum).
Nesterov momentum is based on the formula from On the importance of
initialization and momentum in deep learning.

## Usage

``` r
optim_sgd(
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

## Note

The implementation of SGD with Momentum-Nesterov subtly differs from
Sutskever et. al. and implementations in some other frameworks.

Considering the specific case of Momentum, the update can be written as
\$\$ \begin{array}{ll} v\_{t+1} & = \mu \* v\_{t} + g\_{t+1}, \\
p\_{t+1} & = p\_{t} - \mbox{lr} \* v\_{t+1}, \end{array} \$\$

where \\p\\, \\g\\, \\v\\ and \\\mu\\ denote the parameters, gradient,
velocity, and momentum respectively.

This is in contrast to Sutskever et. al. and other frameworks which
employ an update of the form

\$\$ \begin{array}{ll} v\_{t+1} & = \mu \* v\_{t} + \mbox{lr} \*
g\_{t+1}, \\ p\_{t+1} & = p\_{t} - v\_{t+1}. \end{array} \$\$ The
Nesterov version is analogously modified.

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
optimizer <- optim_sgd(model$parameters(), lr = 0.1, momentum = 0.9)
optimizer$zero_grad()
loss_fn(model(input), target)$backward()
optimizer$step()
} # }

}
```
