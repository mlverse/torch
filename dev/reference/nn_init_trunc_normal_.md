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
#>  0.8115  1.4305  0.0245 -1.1453  0.8699
#>  1.2152 -0.2800  1.6122 -0.3110 -1.7480
#>  1.4146  1.5102 -0.2070  0.6434  0.9544
#> [ CPUFloatType{3,5} ]
```
