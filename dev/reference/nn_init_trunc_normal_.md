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
#>  1.5662 -1.3474 -1.1102 -0.7457 -0.3233
#>  0.2069 -1.5232 -0.9163  1.8009  1.7400
#>  0.5360  0.9780  1.7332 -0.5184  1.1558
#> [ CPUFloatType{3,5} ]
```
