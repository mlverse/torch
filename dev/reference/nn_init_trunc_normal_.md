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
#>  0.7086 -1.0445 -0.6015  0.6182 -0.6275
#> -0.4756  0.7693 -0.9486 -0.3432 -0.9843
#>  0.5205  1.7394  0.3731  0.8029 -0.3389
#> [ CPUFloatType{3,5} ]
```
