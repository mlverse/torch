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
#> -1.0113 -0.7531  0.2675  0.8207  0.5375
#> -0.6100 -0.3650 -0.9122 -0.2912 -0.6103
#>  0.4293  0.5251 -0.9281 -0.7702  1.8528
#> [ CPUFloatType{3,5} ]
```
