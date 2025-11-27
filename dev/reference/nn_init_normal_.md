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
#> -0.6259  0.2003 -0.6395 -0.2001 -0.7780
#> -1.3656  0.0494  0.4675 -0.2251 -1.0573
#>  1.3344  0.4105 -0.3582 -0.4319  0.3922
#> [ CPUFloatType{3,5} ]
```
