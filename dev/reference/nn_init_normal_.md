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
#> -1.1407 -1.7101  0.8403  0.9681  1.9822
#> -0.4350 -0.0102  1.2152  1.4676  1.4075
#>  1.2783  0.2422 -0.4381 -0.5761  0.2169
#> [ CPUFloatType{3,5} ]
```
