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
#> -1.3395  0.2480 -0.4615 -0.0368  0.1954
#>  0.0740  0.1617 -1.2034  1.8706 -2.0828
#>  0.2252  1.1080  1.8845 -0.7172 -0.8349
#> [ CPUFloatType{3,5} ]
```
