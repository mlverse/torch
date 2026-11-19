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
#>  1.3460 -0.6413 -1.1683  0.5448 -0.1041
#> -1.3375 -0.3402  1.1042  0.4835 -1.7904
#> -0.3335 -1.0661  1.7343  0.8865  0.4278
#> [ CPUFloatType{3,5} ]
```
