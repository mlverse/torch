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
#>  0.9654  0.4623 -0.9739 -1.2712  0.1685
#>  0.7285  0.1402 -1.0576 -0.8360  0.7709
#> -0.5137 -0.5875  0.7684  0.2338 -0.8943
#> [ CPUFloatType{3,5} ]
```
