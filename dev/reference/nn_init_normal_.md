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
#> -1.3029 -0.2395  1.0845  1.2790 -0.7240
#> -1.7979  0.6137  0.2373 -0.5532  0.1299
#>  0.1481 -0.0913 -2.3648 -0.3336  0.4456
#> [ CPUFloatType{3,5} ]
```
