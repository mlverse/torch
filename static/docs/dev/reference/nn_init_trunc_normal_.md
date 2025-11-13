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
#>  0.3621  0.1921  0.0964 -1.0150  0.0398
#>  0.1984 -1.3909 -0.2787 -0.7068  0.3592
#>  1.0506 -0.2084 -0.4470  0.6451 -0.6166
#> [ CPUFloatType{3,5} ]
```
