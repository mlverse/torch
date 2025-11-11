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
#> -0.5497  0.8296 -1.5413 -1.1464  0.6016
#>  2.3187  0.6032 -1.1976  1.6051 -0.1369
#>  1.2985  1.2771  0.3636 -0.2722 -0.4222
#> [ CPUFloatType{3,5} ]
```
