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
#>  0.6532  0.6921  1.0366  1.7076  0.4117
#> -0.0198  0.0132  0.2515  0.3228  0.7548
#> -0.9153  0.3978  0.9810  0.0862 -2.3269
#> [ CPUFloatType{3,5} ]
```
