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
#>  1.6103 -0.1816 -2.0193  2.5682  0.4408
#>  1.1688 -0.9732  0.5268  0.5444  0.2263
#> -0.1593  0.5838  0.0684 -0.1065  1.0064
#> [ CPUFloatType{3,5} ]
```
