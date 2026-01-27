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
#> -0.5529  0.3802 -1.9958  0.4782  1.5841
#>  0.7313 -0.7735  1.9824  0.9953  0.5627
#>  0.1355  0.4402  0.3757 -0.7396 -1.4701
#> [ CPUFloatType{3,5} ]
```
