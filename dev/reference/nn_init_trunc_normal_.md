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
#>  1.3116 -1.1773  0.8660 -0.9797 -0.0340
#>  0.1488  0.7650  0.5221 -1.4854 -0.5884
#>  0.1042 -1.0536 -0.8125  0.9350 -1.4366
#> [ CPUFloatType{3,5} ]
```
