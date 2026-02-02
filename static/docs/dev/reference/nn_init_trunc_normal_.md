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
#>  1.0816 -1.2216 -0.0507  1.1409  0.6800
#> -0.6637 -0.5771 -0.5185  0.5051 -0.0879
#> -0.0822  0.3712  0.8935  1.6615  0.4246
#> [ CPUFloatType{3,5} ]
```
