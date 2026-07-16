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
#> -1.1210 -0.7495  0.0130  0.1111  1.7640
#> -1.6053 -1.1973  1.3898  0.3765 -0.1361
#>  0.0745  0.1940  0.2705  1.2692 -0.1377
#> [ CPUFloatType{3,5} ]
```
