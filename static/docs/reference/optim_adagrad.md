# Adagrad optimizer

Proposed in [Adaptive Subgradient Methods for Online Learning and
Stochastic Optimization](https://jmlr.org/papers/v12/duchi11a.html)

## Usage

``` r
optim_adagrad(
  params,
  lr = 0.01,
  lr_decay = 0,
  weight_decay = 0,
  initial_accumulator_value = 0,
  eps = 1e-10
)
```

## Arguments

- params:

  (iterable): list of parameters to optimize or list parameter groups

- lr:

  (float, optional): learning rate (default: 1e-2)

- lr_decay:

  (float, optional): learning rate decay (default: 0)

- weight_decay:

  (float, optional): weight decay (L2 penalty) (default: 0)

- initial_accumulator_value:

  the initial value for the accumulator. (default: 0)

  Adagrad is an especially good optimizer for sparse data. It
  individually modifies learning rate for every single parameter,
  dividing the original learning rate value by sum of the squares of the
  gradients. It causes that the rarely occurring features get greater
  learning rates. The main downside of this method is the fact that
  learning rate may be getting small too fast, so that at some point a
  model cannot learn anymore.

- eps:

  (float, optional): term added to the denominator to improve numerical
  stability (default: 1e-10)

## Note

Update rule: \$\$ \theta\_{t+1} = \theta\_{t} - \frac{\eta
}{\sqrt{G\_{t} + \epsilon}} \odot g\_{t} \$\$ The equation above and
some remarks quoted after [*An overview of gradient descent optimization
algorithms*](https://web.archive.org/web/20220810011734/https://ruder.io/optimizing-gradient-descent/index.html)
by Sebastian Ruder.

## Warning

If you need to move a model to GPU via `$cuda()`, please do so before
constructing optimizers for it. Parameters of a model after `$cuda()`
will be different objects from those before the call. In general, you
should make sure that the objects pointed to by model parameters subject
to optimization remain the same over the whole lifecycle of optimizer
creation and usage.
