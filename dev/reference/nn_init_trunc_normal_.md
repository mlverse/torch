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
#> -0.4164  1.2493 -1.3347 -0.8022 -1.2148
#>  1.1254  0.9762  0.4291  0.4399 -0.5257
#> -1.6849 -0.0392 -0.4106 -0.1953 -1.0331
#> [ CPUFloatType{3,5} ]
```
