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
#> -0.2896  1.9248  0.0145  1.3186  0.4580
#>  0.1903  0.0603 -0.0546 -1.0487 -1.1237
#>  0.1190 -1.2237 -0.9170 -0.1892 -0.0545
#> [ CPUFloatType{3,5} ]
```
