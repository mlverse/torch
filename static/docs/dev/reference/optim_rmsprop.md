# RMSprop optimizer

Proposed by G. Hinton in his course.

## Usage

``` r
optim_rmsprop(
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

## Note

The centered version first appears in [Generating Sequences With
Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850v5). The
implementation here takes the square root of the gradient average before
adding epsilon (note that TensorFlow interchanges these two operations).
The effective learning rate is thus \\\alpha/(\sqrt{v} + \epsilon)\\
where \\\alpha\\ is the scheduled learning rate and \\v\\ is the
weighted moving average of the squared gradient.

Update rule:

\$\$ \theta\_{t+1} = \theta\_{t} - \frac{\eta }{\sqrt{{E\[g^2\]}\_{t} +
\epsilon}} \* g\_{t} \$\$

## Warning

If you need to move a model to GPU via `$cuda()`, please do so before
constructing optimizers for it. Parameters of a model after `$cuda()`
will be different objects from those before the call. In general, you
should make sure that the objects pointed to by model parameters subject
to optimization remain the same over the whole lifecycle of optimizer
creation and usage.
