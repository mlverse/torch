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
#> -0.2230 -1.2051  0.4319 -1.5788 -1.6219
#>  0.4358 -0.3579 -0.0622 -1.0191  0.1040
#>  1.6002  0.7186 -0.5494 -0.2249 -0.5106
#> [ CPUFloatType{3,5} ]
```
