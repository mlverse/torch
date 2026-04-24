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
#>  1.8602  0.2155 -0.3611 -1.0735 -0.2765
#>  0.4878 -1.0964  1.3389 -0.5304  0.3610
#> -0.7943 -0.4201  0.5781  0.9015  0.5586
#> [ CPUFloatType{3,5} ]
```
