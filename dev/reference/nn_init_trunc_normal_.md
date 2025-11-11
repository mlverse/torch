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
#> -0.0048  1.0488 -0.4861  0.2859  0.2477
#>  0.9399 -0.9988 -0.6488  0.4975  0.6242
#> -0.4597  1.5275 -0.9855 -0.4262  0.1292
#> [ CPUFloatType{3,5} ]
```
