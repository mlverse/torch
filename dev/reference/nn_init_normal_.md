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
#> -0.7721  1.0758  0.2869 -1.3399 -0.0852
#> -0.6568 -1.5529 -0.6987  0.2203  0.8089
#>  0.5112  1.3160 -0.4895 -0.9196  0.5069
#> [ CPUFloatType{3,5} ]
```
