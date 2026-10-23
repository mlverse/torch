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
#>  0.1520  1.1519  0.1460 -0.3228 -0.0583
#>  0.4962  0.5166 -0.1890  1.3581  0.7674
#>  0.0507 -0.7886 -1.1823 -0.0621 -0.6020
#> [ CPUFloatType{3,5} ]
```
