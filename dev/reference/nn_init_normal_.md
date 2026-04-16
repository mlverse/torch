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
#>  0.5617  0.1968 -0.2907 -0.3272  0.7886
#>  0.9866 -0.1888 -1.5012 -0.6633  1.1335
#> -0.4427  1.3110  1.6325 -0.1081 -1.8882
#> [ CPUFloatType{3,5} ]
```
