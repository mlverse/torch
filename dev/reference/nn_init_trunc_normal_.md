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
#>  0.5116  1.4981 -1.4706 -1.3452 -0.0104
#> -0.0261 -0.9899 -1.4326 -0.1749  0.5367
#> -1.3575  1.8987  0.3437  1.7883 -0.6254
#> [ CPUFloatType{3,5} ]
```
