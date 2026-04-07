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
#>  0.1446 -0.2434 -0.1181 -0.0170  1.4731
#> -1.0631 -0.0381 -1.6803 -1.1063  1.5604
#> -0.6788  0.7116  0.0444  0.5872  1.2746
#> [ CPUFloatType{3,5} ]
```
