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
#> -0.0836 -1.6139  0.3001  1.3758  0.7116
#> -0.7018  1.0888 -0.7420  0.1758 -1.6328
#> -0.7859 -1.2901 -0.1401  0.0340 -0.3396
#> [ CPUFloatType{3,5} ]
```
