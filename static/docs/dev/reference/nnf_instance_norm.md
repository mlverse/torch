# Instance_norm

Applies Instance Normalization for each channel in each data sample in a
batch.

## Usage

``` r
nnf_instance_norm(
  input,
  running_mean = NULL,
  running_var = NULL,
  weight = NULL,
  bias = NULL,
  use_input_stats = TRUE,
  momentum = 0.1,
  eps = 1e-05
)
```

## Arguments

- input:

  the input tensor

- running_mean:

  the running_mean tensor

- running_var:

  the running var tensor

- weight:

  the weight tensor

- bias:

  the bias tensor

- use_input_stats:

  whether to use input stats

- momentum:

  a double for the momentum

- eps:

  an eps double for numerical stability
