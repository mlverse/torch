# Layer_norm

Applies Layer Normalization for last certain number of dimensions.

## Usage

``` r
nnf_layer_norm(
  input,
  normalized_shape,
  weight = NULL,
  bias = NULL,
  eps = 1e-05
)
```

## Arguments

- input:

  the input tensor

- normalized_shape:

  input shape from an expected input of size. If a single integer is
  used, it is treated as a singleton list, and this module will
  normalize over the last dimension which is expected to be of that
  specific size.

- weight:

  the weight tensor

- bias:

  the bias tensor

- eps:

  a value added to the denominator for numerical stability. Default:
  1e-5
