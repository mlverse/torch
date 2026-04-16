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
#>  0.4510 -1.0503 -1.3696 -1.8613  0.2959
#> -0.7551  1.1548 -0.9453 -0.1223 -1.1160
#> -0.3356 -1.0809 -0.5123  0.5643 -1.6727
#> [ CPUFloatType{3,5} ]
```
