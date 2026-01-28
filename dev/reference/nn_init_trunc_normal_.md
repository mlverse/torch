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
#> -0.5975 -0.3844 -0.8930  0.9813 -0.5167
#> -0.4247 -0.2191 -0.6907  0.5502 -0.2155
#> -1.1785 -0.2040 -0.1254 -0.7202 -0.9808
#> [ CPUFloatType{3,5} ]
```
