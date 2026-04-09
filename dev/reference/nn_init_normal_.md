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
#> -1.5488 -1.5608 -2.1554 -2.3414 -1.3613
#> -0.1092 -1.1444 -0.2914  0.9151 -1.2944
#>  0.5716 -0.3215 -0.5169  0.8957  0.3971
#> [ CPUFloatType{3,5} ]
```
