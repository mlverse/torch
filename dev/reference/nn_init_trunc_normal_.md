# Truncated normal initialization

Fills the input Tensor with values drawn from a truncated normal
distribution.

## Usage

``` r
nn_init_trunc_normal_(tensor, mean = 0, std = 1, a = -2, b = 2)
```

## Arguments

- tensor:

  an n-dimensional Tensor

- mean:

  the mean of the normal distribution

- std:

  the standard deviation of the normal distribution

- a:

  the minimum cutoff value

- b:

  the maximum cutoff value

## Examples

``` r
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_trunc_normal_(w)
}
#> torch_tensor
#>  0.7169  0.5156 -0.5015  0.8152  0.3892
#>  1.6825  0.5823  0.1249 -0.2365  0.3972
#> -0.6086  0.9432 -1.1624 -0.2687 -0.5962
#> [ CPUFloatType{3,5} ]
```
