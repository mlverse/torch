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
#>  1.2661  0.8331  0.4700 -1.0824 -0.4128
#> -1.1975 -0.7369 -0.4477 -1.7559 -0.3550
#>  0.2874  1.5776  0.8292 -0.3667  0.4331
#> [ CPUFloatType{3,5} ]
```
