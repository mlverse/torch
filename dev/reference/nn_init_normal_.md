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
#>  1.4000  1.1336 -0.5434  0.1504  1.2381
#>  0.1115  0.4195 -1.5681  1.6227  0.8242
#>  1.7017  0.6412  0.2661  2.4525 -0.0157
#> [ CPUFloatType{3,5} ]
```
