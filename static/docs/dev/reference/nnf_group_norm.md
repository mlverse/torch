# Group_norm

Applies Group Normalization for last certain number of dimensions.

## Usage

``` r
nnf_group_norm(input, num_groups, weight = NULL, bias = NULL, eps = 1e-05)
```

## Arguments

- input:

  the input tensor

- num_groups:

  number of groups to separate the channels into

- weight:

  the weight tensor

- bias:

  the bias tensor

- eps:

  a value added to the denominator for numerical stability. Default:
  1e-5
