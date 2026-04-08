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
#> -0.3709  1.1153  0.4780  0.0937  0.4895
#>  1.0823  0.6149 -1.5338  0.4720  0.9347
#>  0.2802 -1.4567  0.2953 -0.6080 -0.2323
#> [ CPUFloatType{3,5} ]
```
