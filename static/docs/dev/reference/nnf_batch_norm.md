# Batch_norm

Applies Batch Normalization for each channel across a batch of data.

## Usage

``` r
nnf_batch_norm(
  input,
  running_mean,
  running_var,
  weight = NULL,
  bias = NULL,
  training = FALSE,
  momentum = 0.1,
  eps = 1e-05
)
```

## Arguments

- input:

  input tensor

- running_mean:

  the running_mean tensor

- running_var:

  the running_var tensor

- weight:

  the weight tensor

- bias:

  the bias tensor

- training:

  bool wether it's training. Default: FALSE

- momentum:

  the value used for the `running_mean` and `running_var` computation.
  Can be set to None for cumulative moving average (i.e. simple
  average). Default: 0.1

- eps:

  a value added to the denominator for numerical stability. Default:
  1e-5
