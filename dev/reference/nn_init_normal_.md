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
#> -0.9509  0.5450  0.3533  1.0295  1.7525
#>  0.4416 -0.5272  0.5385 -0.0403 -1.3089
#>  1.9203 -1.3828 -0.3535  1.2634  0.8875
#> [ CPUFloatType{3,5} ]
```
