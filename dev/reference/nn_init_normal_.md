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
#>  1.1629  0.2945 -0.3629 -0.3969 -0.5999
#> -1.0858 -0.3479  0.4309  0.9315  0.4120
#>  1.0615 -0.4045  1.5608  0.2569  1.6301
#> [ CPUFloatType{3,5} ]
```
