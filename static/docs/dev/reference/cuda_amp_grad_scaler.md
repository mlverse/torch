# Creates a gradient scaler

A gradient scaler instance is used to perform dynamic gradient scaling
to avoid gradient underflow when training with mixed precision.

## Usage

``` r
cuda_amp_grad_scaler(
  init_scale = 2^16,
  growth_factor = 2,
  backoff_factor = 0.5,
  growth_interval = 2000,
  enabled = TRUE
)
```

## Arguments

- init_scale:

  a numeric value indicating the initial scale factor.

- growth_factor:

  a numeric value indicating the growth factor.

- backoff_factor:

  a numeric value indicating the backoff factor.

- growth_interval:

  a numeric value indicating the growth interval.

- enabled:

  a logical value indicating whether the gradient scaler should be
  enabled.

## Value

A gradient scaler object.
