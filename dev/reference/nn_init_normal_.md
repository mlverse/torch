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
#> -0.7995  1.4412 -0.6692 -0.2127  0.4484
#>  1.4948 -0.7845  0.6259 -0.9800 -0.9426
#>  2.4440  0.9775  1.1990 -0.0045  0.4999
#> [ CPUFloatType{3,5} ]
```
