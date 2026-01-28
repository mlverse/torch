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
#> -0.3946  0.8631 -0.1614 -1.1183  1.1930
#> -1.3802  0.5602 -0.2788 -0.3976 -0.8241
#> -1.7753  0.0143  0.4665 -0.0569 -1.1691
#> [ CPUFloatType{3,5} ]
```
