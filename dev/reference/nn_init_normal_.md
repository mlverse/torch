# Normal initialization

Fills the input Tensor with values drawn from the normal distribution

## Usage

``` r
nn_init_normal_(tensor, mean = 0, std = 1)
```

## Arguments

- tensor:

  an n-dimensional Tensor

- mean:

  the mean of the normal distribution

- std:

  the standard deviation of the normal distribution

## Examples

``` r
if (torch_is_installed()) {
w <- torch_empty(3, 5)
nn_init_normal_(w)
}
#> torch_tensor
#> -1.9065  0.1094 -1.9814 -0.4104  0.2956
#>  0.4850 -0.6129 -1.8040 -1.0332  0.8786
#> -0.7913  0.1575  0.1857 -1.1765 -1.1846
#> [ CPUFloatType{3,5} ]
```
