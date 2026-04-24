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
#> -0.4931 -0.1101 -0.9741 -1.0564  0.0780
#> -0.4101  0.1917  0.7714  0.0073 -0.2134
#> -0.3038 -0.5674 -0.3091  0.0876 -0.3761
#> [ CPUFloatType{3,5} ]
```
