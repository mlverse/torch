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
#> -0.8941  0.7530 -1.2858  0.5816  0.0771
#>  2.0024 -0.3913  0.5016 -1.3994  0.3438
#> -0.7499 -1.6501  0.6724 -0.9632  0.3714
#> [ CPUFloatType{3,5} ]
```
