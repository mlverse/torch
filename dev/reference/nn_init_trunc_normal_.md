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
#> -0.4298 -0.5779 -0.7127  0.5540  0.5080
#>  1.9787 -1.5926 -0.4574  1.0816  0.4359
#> -0.8585 -1.1857 -0.4822 -1.7102 -0.9749
#> [ CPUFloatType{3,5} ]
```
